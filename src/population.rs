use crate::environment::Environment;
use crate::genome::Genome;
use crate::organism::Organism;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::cmp::Ordering;
#[cfg(feature = "telemetry")]
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(feature = "telemetry")]
use rusty_dashed;

#[cfg(feature = "telemetry")]
use serde_json;

use crate::mutation_config::MutationConfig;
use crate::specie::Specie;
use crate::species_evaluator::SpeciesEvaluator;

/// All species in the network
#[derive(Debug)]
pub struct Population {
    /// container of species
    pub species: Vec<Specie>,
    champion_fitness: f64,
    epochs_without_improvements: usize,
    /// champion of the population
    pub champion: Option<Organism>,
    /// Mutation configuration
    pub mutation_config: MutationConfig,
    /// Source of every random decision taken by the population (initial
    /// weights, selection, crossover, mutation, representatives).
    rng: ChaCha8Rng,
    /// How the initial organisms were built, so `with_seed` can rebuild them.
    initial: InitialPopulation,
}

/// Shape of the initial (minimal) population: inputs and outputs only.
#[derive(Debug, Clone, Copy)]
struct InitialPopulation {
    size: usize,
    input_neurons: usize,
    output_neurons: usize,
    /// true: every input connected to every output; false: no connections.
    connected: bool,
}

const MAX_EPOCHS_WITHOUT_IMPROVEMENTS: usize = 50;
const STAGNATION_THRESHOLD: usize = 15; // Remove species after 15 generations without improvement
const SPECIES_ELITISM: usize = 2; // Protect top 2 species from stagnation removal

impl Population {
    /// Create a new population of size X whose genomes are a single neuron,
    /// without inputs or outputs.
    #[deprecated(
        note = "declare the inputs and outputs: use `create_population_initialized(size, inputs, outputs)`"
    )]
    pub fn create_population(population_size: usize) -> Population {
        Population::new(
            InitialPopulation {
                size: population_size,
                input_neurons: 0,
                output_neurons: 0,
                connected: true,
            },
            MutationConfig::default(),
        )
    }

    /// Create a population of size X with the minimal topology: every input
    /// connected directly to every output, no hidden neurons. Each organism
    /// draws its own initial weights.
    pub fn create_population_initialized(
        population_size: usize,
        input_neurons: usize,
        output_neurons: usize,
    ) -> Population {
        Population::create_population_initialized_with_config(
            population_size,
            input_neurons,
            output_neurons,
            MutationConfig::default(),
        )
    }

    /// Create a population with custom mutation configuration
    pub fn create_population_initialized_with_config(
        population_size: usize,
        input_neurons: usize,
        output_neurons: usize,
        config: MutationConfig,
    ) -> Population {
        Population::new(
            InitialPopulation {
                size: population_size,
                input_neurons,
                output_neurons,
                connected: true,
            },
            config,
        )
    }

    /// Create a population with unconnected genomes (no initial connections).
    /// NEAT will discover connections through structural mutation.
    pub fn create_population_unconnected_with_config(
        population_size: usize,
        input_neurons: usize,
        output_neurons: usize,
        config: MutationConfig,
    ) -> Population {
        Population::new(
            InitialPopulation {
                size: population_size,
                input_neurons,
                output_neurons,
                connected: false,
            },
            config,
        )
    }

    /// Make the population reproducible: reseed its random generator with
    /// `seed` and rebuild the initial organisms from it, with the same shape
    /// the constructor used and the current `mutation_config` (tau, step
    /// time, output activation). Call it right after a constructor; any
    /// evolution done before is discarded. The constructors without seed use a seed from entropy.
    ///
    /// The same seed and the same (deterministic) environment always produce
    /// the same genomes, whatever `Environment::threads` returns.
    ///
    /// ```
    /// use rustneat::{MutationConfig, Population};
    /// let population = Population::create_population_initialized_with_config(
    ///     150, 2, 1, MutationConfig::default(),
    /// )
    /// .with_seed(42);
    /// assert_eq!(population.size(), 150);
    /// ```
    pub fn with_seed(mut self, seed: u64) -> Population {
        self.rng = ChaCha8Rng::seed_from_u64(seed);
        self.champion = None;
        self.champion_fitness = 0f64;
        self.epochs_without_improvements = 0;
        self.create_initial_organisms();
        self
    }

    fn new(initial: InitialPopulation, config: MutationConfig) -> Population {
        let mut population = Population {
            species: vec![],
            champion_fitness: 0f64,
            champion: None,
            epochs_without_improvements: 0usize,
            mutation_config: config,
            rng: ChaCha8Rng::from_entropy(),
            initial,
        };
        population.create_initial_organisms();
        population
    }

    /// Number of organisms the population keeps in every generation: the
    /// size given to the constructor.
    pub fn target_size(&self) -> usize {
        self.initial.size
    }

    /// Find total of all organisms in the population
    pub fn size(&self) -> usize {
        self.species
            .iter()
            .fold(0usize, |total, specie| total + specie.organisms.len())
    }

    /// Create offspring by mutation and mating. May create new species.
    pub fn evolve(&mut self) {
        self.generate_offspring();
    }

    /// Evaluate all organisms in the population using the given environment.
    pub fn evaluate_in(&mut self, environment: &dyn Environment) {
        let champion = SpeciesEvaluator::new(environment).evaluate(&mut self.species);

        // Apply fitness sharing and update stagnation tracking. Stagnant
        // species are removed in the next `evolve`, where their share of the
        // offspring goes to the surviving species.
        for specie in &mut self.species {
            specie.adjust_fitness();
            specie.update_stagnation();
        }

        #[cfg(feature = "telemetry")]
        telemetry!("fitness1", 1.0, format!("{}", self.champion_fitness));

        if self.champion_fitness >= champion.fitness {
            self.epochs_without_improvements += 1;
        } else {
            #[cfg(feature = "telemetry")]
            telemetry!(
                "network1",
                1.0,
                serde_json::to_string(&champion.genome.get_genes()).unwrap()
            );
            self.epochs_without_improvements = 0usize;
            self.champion = Some(champion.clone());
            self.champion_fitness = champion.fitness;
        }
        self.champion_fitness = champion.fitness;
    }

    /// Remove species that haven't improved for too long
    fn remove_stagnant_species(&mut self, max_generations: usize, protect_top_n: usize) {
        if self.species.len() <= protect_top_n {
            return;
        }

        // Sort by champion fitness descending to identify top species
        self.species.sort_by(|a, b| {
            b.calculate_champion_fitness()
                .partial_cmp(&a.calculate_champion_fitness())
                .unwrap_or(Ordering::Equal)
        });

        // Mark which species to keep (top N are protected)
        let mut to_remove = vec![];
        for (i, specie) in self.species.iter().enumerate() {
            if i >= protect_top_n && specie.is_stagnant(max_generations) {
                to_remove.push(i);
            }
        }

        // Remove stagnant species (in reverse order to preserve indices)
        for i in to_remove.into_iter().rev() {
            self.species.remove(i);
        }
    }

    /// Return all organisms of the population
    pub fn get_organisms(&self) -> Vec<Organism> {
        self.species
            .iter()
            .flat_map(|specie| specie.organisms.clone())
            .collect::<Vec<Organism>>()
    }

    /// How many iterations without improvement
    pub fn epochs_without_improvements(&self) -> usize {
        self.epochs_without_improvements
    }

    fn generate_offspring(&mut self) {
        // Remove stagnant species (but protect top 2 by fitness); the
        // survivors fill the population back to its target size below.
        self.remove_stagnant_species(STAGNATION_THRESHOLD, SPECIES_ELITISM);
        self.speciate();

        let target_size = self.target_size();
        let organisms = self.get_organisms();
        let config = self.mutation_config;

        if self.epochs_without_improvements > MAX_EPOCHS_WITHOUT_IMPROVEMENTS {
            let mut best_species = self.get_best_species();
            let num_of_selected = best_species.len();
            for specie in &mut best_species {
                specie.generate_offspring_with_config_and_rng(
                    target_size.checked_div(num_of_selected).unwrap(),
                    &organisms,
                    &config,
                    &mut self.rng,
                );
            }
            self.epochs_without_improvements = 0;
            return;
        }

        let average_fitnesses: Vec<f64> = self
            .species
            .iter_mut()
            .map(|specie| specie.calculate_average_fitness())
            .collect();
        let total_average_fitness: f64 = average_fitnesses.iter().sum();
        // Without positive fitness, keep the current proportions.
        let weights: Vec<f64> = if total_average_fitness > 0f64 {
            average_fitnesses
        } else {
            self.species
                .iter()
                .map(|specie| specie.organisms.len() as f64)
                .collect()
        };
        let offspring_sizes = apportion(&weights, target_size);

        for (specie, offspring_size) in self.species.iter_mut().zip(offspring_sizes) {
            if offspring_size > 0 {
                specie.generate_offspring_with_config_and_rng(
                    offspring_size,
                    &organisms,
                    &config,
                    &mut self.rng,
                );
            } else {
                specie.remove_organisms();
            }
        }
    }

    fn get_best_species(&mut self) -> Vec<Specie> {
        if self.species.len() <= 2 {
            return self.species.clone();
        }

        // Sort by champion fitness descending (best first)
        self.species.sort_by(|specie1, specie2| {
            specie2
                .calculate_champion_fitness()
                .partial_cmp(&specie1.calculate_champion_fitness())
                .unwrap_or(Ordering::Equal)
        });

        // Return the top 2 species
        self.species[0..2].to_vec()
    }

    fn speciate(&mut self) {
        let organisms = &self.get_organisms();
        self.species.retain(|specie| !specie.is_empty());

        let mut next_specie_id = 0i64;

        #[cfg(feature = "telemetry")]
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();

        for specie in &mut self.species {
            #[cfg(feature = "telemetry")]
            telemetry!(
                "species1",
                1.0,
                format!(
                    "{{'id':{}, 'fitness':{}, 'organisms':{}, 'timestamp':'{:?}'}}",
                    specie.id,
                    specie.calculate_champion_fitness(),
                    specie.organisms.len(),
                    now
                )
            );

            specie.choose_new_representative_with_rng(&mut self.rng);

            specie.remove_organisms();

            specie.id = next_specie_id;
            next_specie_id += 1;
        }

        let threshold = self.mutation_config.compatibility_threshold;
        for organism in organisms {
            match self
                .species
                .iter_mut()
                .find(|specie| specie.match_genome_with_threshold(organism, threshold))
            {
                Some(specie) => {
                    specie.add(organism.clone());
                }
                None => {
                    let mut specie = Specie::new(organism.genome.clone());
                    specie.id = next_specie_id;
                    specie.add(organism.clone());
                    next_specie_id += 1;
                    self.species.push(specie);
                }
            };
        }
        self.species.retain(|specie| !specie.is_empty());
    }

    fn create_initial_organisms(&mut self) {
        let InitialPopulation {
            size,
            input_neurons,
            output_neurons,
            connected,
        } = self.initial;
        self.species = vec![];
        let mut organisms = vec![];

        while organisms.len() < size {
            let genome = if connected {
                Genome::new_initialized_with_rng(input_neurons, output_neurons, &mut self.rng)
            } else {
                Genome::new_unconnected(input_neurons, output_neurons)
            };
            let mut org = Organism::new(genome);
            org.tau = self.mutation_config.tau;
            org.step_time = self.mutation_config.step_time;
            org.output_activation = self.mutation_config.output_activation;
            organisms.push(org);
        }

        let mut specie = Specie::new(organisms.first().unwrap().genome.clone());
        specie.organisms = organisms;
        self.species.push(specie);
    }
}

/// Split `total` into integer shares proportional to `weights` by the
/// largest remainder method: every share gets the floor of its exact quota
/// and the units still missing go to the largest fractional parts (ties to
/// the first). The shares always add up to `total`. Negative or non finite
/// weights count as zero; if no weight is positive, `total` is split evenly.
fn apportion(weights: &[f64], total: usize) -> Vec<usize> {
    if weights.is_empty() {
        return vec![];
    }
    let weights: Vec<f64> = weights
        .iter()
        .map(|w| if w.is_finite() && *w > 0f64 { *w } else { 0f64 })
        .collect();
    let sum: f64 = weights.iter().sum();
    let quotas: Vec<f64> = if sum > 0f64 {
        weights.iter().map(|w| w * total as f64 / sum).collect()
    } else {
        vec![total as f64 / weights.len() as f64; weights.len()]
    };
    let mut shares: Vec<usize> = quotas.iter().map(|q| q.floor() as usize).collect();
    let assigned: usize = shares.iter().sum();
    // Floating point error could push the floors over `total`; never give
    // more than asked.
    let mut missing = total.saturating_sub(assigned);
    let mut by_remainder: Vec<usize> = (0..quotas.len()).collect();
    by_remainder.sort_by(|&a, &b| {
        let ra = quotas[a] - quotas[a].floor();
        let rb = quotas[b] - quotas[b].floor();
        rb.partial_cmp(&ra).unwrap_or(Ordering::Equal)
    });
    let mut i = 0;
    while missing > 0 {
        shares[by_remainder[i % by_remainder.len()]] += 1;
        missing -= 1;
        i += 1;
    }
    let mut excess = shares.iter().sum::<usize>().saturating_sub(total);
    for &j in by_remainder.iter().rev() {
        while excess > 0 && shares[j] > 0 {
            shares[j] -= 1;
            excess -= 1;
        }
    }
    shares
}

#[cfg(test)]
use crate::gene::Gene;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::Genome;
    use crate::organism::Organism;
    use crate::specie::Specie;

    #[test]
    fn population_should_be_able_to_speciate_genomes() {
        let mut genome1 = Genome::default();
        genome1.add_gene(Gene::new(0, 0, 1f64, true));
        genome1.add_gene(Gene::new(0, 1, 1f64, true));
        let mut genome2 = Genome::default();
        genome2.add_gene(Gene::new(1, 1, 1f64, true));
        genome2.add_gene(Gene::new(1, 0, 1f64, true));

        let mut population = Population::create_population_initialized(2, 1, 1);
        population.mutation_config.compatibility_threshold = 1.0;
        let organisms = vec![Organism::new(genome1), Organism::new(genome2)];
        let mut specie = Specie::new(organisms.first().unwrap().genome.clone());
        specie.organisms = organisms;
        population.species = vec![specie];
        population.speciate();
        assert_eq!(population.species.len(), 2usize);
    }

    #[test]
    fn apportion_by_largest_remainder_adds_up_exactly() {
        assert_eq!(apportion(&[1.0, 1.0, 1.0], 10), vec![4, 3, 3]);
        assert_eq!(apportion(&[1.0, 2.0], 150), vec![50, 100]);
        // 150 * (0.3, 0.3, 0.4) = 45, 45, 60
        assert_eq!(apportion(&[0.3, 0.3, 0.4], 150), vec![45, 45, 60]);
        // 7 * (0.25, 0.35, 0.4) = 1.75, 2.45, 2.8 -> floors 1, 2, 2; the two
        // largest remainders (0.8, 0.75) get the missing 2.
        assert_eq!(apportion(&[0.25, 0.35, 0.4], 7), vec![2, 2, 3]);
        let weights = [0.13, 7.0, 3.3, 0.001, 2.2, 9.9, 1.0];
        for total in [0usize, 1, 2, 7, 149, 150, 151, 1000] {
            assert_eq!(apportion(&weights, total).iter().sum::<usize>(), total);
        }
    }

    #[test]
    fn apportion_without_positive_weights_splits_evenly() {
        assert_eq!(apportion(&[0.0, 0.0, 0.0], 10), vec![4, 3, 3]);
        assert_eq!(apportion(&[0.0, -1.0], 3), vec![2, 1]);
    }

    #[test]
    fn after_population_evolve_population_should_be_the_same() {
        let mut population = Population::create_population_initialized(150, 2, 1);
        for _ in 0..150 {
            population.evolve();
        }
        assert!(population.size() == 150);
    }

    #[test]
    fn initial_population_is_minimal_with_different_weights_per_organism() {
        let population = Population::create_population_initialized(20, 2, 1).with_seed(1);
        let mut weights: Vec<Vec<f64>> = vec![];
        for organism in population.get_organisms() {
            let genome = &organism.genome;
            assert_eq!((genome.input_neurons(), genome.output_neurons()), (2, 1));
            assert_eq!(genome.len(), 3);
            let genes = genome.get_genes();
            let topology: Vec<(usize, usize)> = genes
                .iter()
                .map(|g| (g.in_neuron_id(), g.out_neuron_id()))
                .collect();
            assert_eq!(topology, vec![(0, 2), (1, 2)]);
            weights.push(genes.iter().map(|g| g.weight()).collect());
        }
        for (i, a) in weights.iter().enumerate() {
            for b in &weights[i + 1..] {
                assert_ne!(a, b);
            }
        }
    }

    #[test]
    #[allow(deprecated)]
    fn deprecated_create_population_still_works() {
        let mut population = Population::create_population(10);
        population.evolve();
        assert_eq!(population.size(), 10);
    }

    #[test]
    fn initial_organisms_take_output_activation_from_config() {
        use crate::organism::OutputActivation;
        let config = MutationConfig::new()
            .output_activation(OutputActivation::Linear)
            .build();
        let population = Population::create_population_initialized_with_config(10, 2, 1, config);
        assert!(population
            .get_organisms()
            .iter()
            .all(|o| o.output_activation == OutputActivation::Linear));
        let population = Population::create_population_initialized(10, 2, 1);
        assert!(population
            .get_organisms()
            .iter()
            .all(|o| o.output_activation == OutputActivation::Sigmoid));
    }
}
