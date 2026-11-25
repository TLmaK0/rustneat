use crate::environment::Environment;
use crate::genome::Genome;
use crate::innovation::InnovationRegistry;
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
    /// Fitness of `champion` (-inf before the first evaluation).
    champion_fitness: f64,
    epochs_without_improvements: usize,
    /// Id for the next new species: ids are never reused, so a species
    /// keeps its id for its whole life.
    next_species_id: i64,
    /// Best organism ever evaluated in the population (the historical
    /// champion): it is only replaced by a strictly fitter one.
    pub champion: Option<Organism>,
    /// Mutation configuration
    pub mutation_config: MutationConfig,
    /// Source of every random decision taken by the population (initial
    /// weights, selection, crossover, mutation, representatives).
    rng: ChaCha8Rng,
    /// How the initial organisms were built, so `with_seed` can rebuild them.
    initial: InitialPopulation,
    /// Ids of the neurons created by splitting connections, shared by every
    /// genome of the population.
    innovations: InnovationRegistry,
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
        self.champion_fitness = f64::NEG_INFINITY;
        self.epochs_without_improvements = 0;
        self.create_initial_organisms();
        self
    }

    fn new(initial: InitialPopulation, config: MutationConfig) -> Population {
        let mut population = Population {
            species: vec![],
            champion_fitness: f64::NEG_INFINITY,
            champion: None,
            epochs_without_improvements: 0usize,
            next_species_id: 0,
            mutation_config: config,
            rng: ChaCha8Rng::from_entropy(),
            initial,
            innovations: InnovationRegistry::default(),
        };
        population.create_initial_organisms();
        population
    }

    /// Number of organisms the population keeps in every generation: the
    /// size given to the constructor.
    pub fn target_size(&self) -> usize {
        self.initial.size
    }

    /// The innovation registry: the neuron id of every connection split so
    /// far in the population.
    pub fn innovations(&self) -> &InnovationRegistry {
        &self.innovations
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

    /// Evaluate all organisms in the population using the given environment
    /// and update `champion`, the best organism ever evaluated. Fitness can
    /// be negative.
    ///
    /// # Errors
    ///
    /// If the environment gives a fitness that is not a finite number (NaN
    /// or an infinity) it returns a `FitnessError` and leaves the champion
    /// and the species statistics as they were. The population keeps its
    /// size: `evolve` still works, and treats those organisms as the worst
    /// ones.
    pub fn evaluate_in(&mut self, environment: &dyn Environment) -> Result<(), FitnessError> {
        let champion = SpeciesEvaluator::new(environment).evaluate(&mut self.species);

        let invalid: Vec<f64> = self
            .species
            .iter()
            .flat_map(|specie| specie.organisms.iter())
            .map(|organism| organism.fitness)
            .filter(|fitness| !fitness.is_finite())
            .collect();
        if let Some(&first) = invalid.first() {
            return Err(FitnessError {
                count: invalid.len(),
                first,
            });
        }

        // Apply fitness sharing and update stagnation tracking. Stagnant
        // species are removed in the next `evolve`, where their share of the
        // offspring goes to the surviving species.
        for specie in &mut self.species {
            specie.adjust_fitness();
            specie.update_stagnation();
        }

        #[cfg(feature = "telemetry")]
        telemetry!("fitness1", 1.0, format!("{}", self.champion_fitness));

        if champion.fitness <= self.champion_fitness {
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
        Ok(())
    }

    /// Remove the species that are stagnant (see
    /// `MutationConfig::species_stagnation_generations`), except the
    /// `species_elitism` species with the best current champion. The
    /// survivors fill the population back to its target size.
    fn remove_stagnant_species(&mut self) {
        let max_generations = self.mutation_config.species_stagnation_generations;
        let protected = self.protected_species(self.mutation_config.species_elitism);
        let mut index = 0;
        self.species.retain(|specie| {
            let keep = protected.contains(&index) || !specie.is_stagnant(max_generations);
            index += 1;
            keep
        });
    }

    /// Positions in `self.species` of the `n` species with the best current
    /// champion (ties: the first ones).
    fn protected_species(&self, n: usize) -> Vec<usize> {
        let mut ranking: Vec<usize> = (0..self.species.len()).collect();
        ranking.sort_by(|&a, &b| {
            let fa = self.species[a].calculate_champion_fitness();
            let fb = self.species[b].calculate_champion_fitness();
            fb.partial_cmp(&fa).unwrap_or(Ordering::Equal)
        });
        ranking.truncate(n);
        ranking
    }

    /// Add a species with a new id.
    fn add_species(&mut self, mut specie: Specie) {
        specie.id = self.next_species_id;
        self.next_species_id += 1;
        self.species.push(specie);
    }

    /// Return all organisms of the population
    pub fn get_organisms(&self) -> Vec<Organism> {
        self.species
            .iter()
            .flat_map(|specie| specie.organisms.clone())
            .collect::<Vec<Organism>>()
    }

    /// How many evaluations in a row did not beat the historical champion.
    pub fn epochs_without_improvements(&self) -> usize {
        self.epochs_without_improvements
    }

    fn generate_offspring(&mut self) {
        // There is no population-level "rescue" when the champion does not
        // improve: the stagnation of each species frees the offspring of the
        // species that do not progress, and the best ones are protected.
        self.remove_stagnant_species();
        self.speciate();

        let target_size = self.target_size();
        // Candidates for an interspecies crossover: the organisms each
        // species lets reproduce, taken before any species is replaced by
        // its offspring.
        let parents = species_parents(&self.species);
        let config = self.mutation_config;

        let average_fitnesses: Vec<f64> = self
            .species
            .iter()
            .map(|specie| specie.calculate_average_fitness())
            .collect();
        let average_fitnesses = shift_to_non_negative(&average_fitnesses);
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

        for (index, (specie, offspring_size)) in
            self.species.iter_mut().zip(offspring_sizes).enumerate()
        {
            if offspring_size > 0 {
                let other_species_parents = other_species_parents(&parents, index);
                specie.generate_offspring_with_mates(
                    offspring_size,
                    &other_species_parents,
                    &config,
                    Some(&mut self.innovations),
                    &mut self.rng,
                );
            } else {
                specie.remove_organisms();
            }
        }
    }

    fn speciate(&mut self) {
        self.species.retain(|specie| !specie.is_empty());
        // Every organism leaves its species (moved, not cloned) and is
        // assigned again, in the same order.
        let mut organisms = Vec::with_capacity(self.size());

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

            organisms.append(&mut specie.organisms);
        }

        let config = self.mutation_config;
        for organism in organisms {
            match self
                .species
                .iter_mut()
                .find(|specie| specie.match_genome_with_config(&organism, &config))
            {
                Some(specie) => {
                    specie.add(organism);
                }
                None => {
                    let mut specie = Specie::new(organism.genome.clone());
                    specie.add(organism);
                    self.add_species(specie);
                }
            };
        }
        self.species.retain(|specie| !specie.is_empty());
        self.adjust_compatibility_threshold();
    }

    /// Dynamic threshold: move `compatibility_threshold` one step towards
    /// the value that gives `target_species` species, for the next
    /// speciation.
    fn adjust_compatibility_threshold(&mut self) {
        let config = &mut self.mutation_config;
        let target = match config.target_species {
            Some(target) => target,
            None => return,
        };
        let species = self.species.len();
        if species > target {
            config.compatibility_threshold += config.compatibility_threshold_step;
        } else if species < target {
            config.compatibility_threshold = (config.compatibility_threshold
                - config.compatibility_threshold_step)
                .max(config.compatibility_threshold_min);
        }
    }

    fn create_initial_organisms(&mut self) {
        let InitialPopulation {
            size,
            input_neurons,
            output_neurons,
            connected,
        } = self.initial;
        self.species = vec![];
        self.innovations = InnovationRegistry::new(input_neurons + output_neurons);
        let mut organisms = vec![];

        while organisms.len() < size {
            let genome = if connected {
                Genome::new_initialized_with_rng(input_neurons, output_neurons, &mut self.rng)
            } else {
                Genome::new_unconnected(input_neurons, output_neurons)
            };
            let mut org = Organism::new(genome);
            org.tau = self.mutation_config.tau;
            org.dt = self.mutation_config.dt;
            org.step_time = self.mutation_config.step_time;
            org.output_activation = self.mutation_config.output_activation;
            organisms.push(org);
        }

        let mut specie = Specie::new(organisms.first().unwrap().genome.clone());
        specie.organisms = organisms;
        self.next_species_id = 0;
        self.add_species(specie);
    }
}

/// A copy of the parents (`Specie::parents`) of each species.
fn species_parents(species: &[Specie]) -> Vec<Vec<Organism>> {
    species
        .iter()
        .map(|specie| specie.parents().cloned().collect())
        .collect()
}

/// The parents of every species except the one at `index`.
fn other_species_parents(parents: &[Vec<Organism>], index: usize) -> Vec<&Organism> {
    parents
        .iter()
        .enumerate()
        .filter(|&(other, _)| other != index)
        .flat_map(|(_, organisms)| organisms.iter())
        .collect()
}

/// Shift `values` so that none is negative: if the lowest finite value is
/// negative, subtract it from all of them (the lowest becomes 0); otherwise
/// leave them as they are, so non negative fitness keeps the proportions it
/// always had. Non finite values (only after an evaluation that returned a
/// `FitnessError`) become 0: those species get no share.
fn shift_to_non_negative(values: &[f64]) -> Vec<f64> {
    let lowest = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold(0f64, f64::min);
    values
        .iter()
        .map(|value| {
            if value.is_finite() {
                value - lowest
            } else {
                0f64
            }
        })
        .collect()
}

/// Error returned by `Population::evaluate_in` when the environment gave
/// a fitness that is not a finite number.
#[derive(Debug, Clone, PartialEq)]
pub struct FitnessError {
    /// Organisms with a non finite fitness.
    pub count: usize,
    /// First non finite fitness found (NaN or an infinity).
    pub first: f64,
}

impl std::fmt::Display for FitnessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} organism(s) got a fitness that is not a finite number (first: {}); \
             the environment must return finite fitness values",
            self.count, self.first
        )
    }
}

impl std::error::Error for FitnessError {}

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
    fn interspecies_mates_are_the_parents_of_the_other_species() {
        let organism = |fitness: f64| {
            let mut organism = Organism::new(Genome::default());
            organism.fitness = fitness;
            organism
        };
        let mut species = vec![];
        for fitnesses in [
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 20.0, 30.0, 40.0],
            [5.0, 6.0, 7.0, 8.0],
        ] {
            let mut specie = Specie::new(Genome::default());
            for fitness in fitnesses {
                specie.add(organism(fitness));
            }
            species.push(specie);
        }
        let parents = species_parents(&species);
        let fitnesses = |index| {
            let mut f: Vec<f64> = other_species_parents(&parents, index)
                .iter()
                .map(|o| o.fitness)
                .collect();
            f.sort_by(|a, b| a.partial_cmp(b).unwrap());
            f
        };
        assert_eq!(fitnesses(0), vec![7.0, 8.0, 30.0, 40.0]);
        assert_eq!(fitnesses(1), vec![3.0, 4.0, 7.0, 8.0]);
        assert_eq!(fitnesses(2), vec![3.0, 4.0, 30.0, 40.0]);
        assert!(other_species_parents(&parents[..1], 0).is_empty());
    }

    #[test]
    fn shift_to_non_negative_moves_negative_values_and_keeps_positive_ones() {
        assert_eq!(shift_to_non_negative(&[1.0, 3.0]), vec![1.0, 3.0]);
        assert_eq!(
            shift_to_non_negative(&[-100.0, -90.0, -95.0]),
            vec![0.0, 10.0, 5.0]
        );
        assert_eq!(shift_to_non_negative(&[-2.0, 3.0]), vec![0.0, 5.0]);
        assert_eq!(
            shift_to_non_negative(&[f64::NAN, -1.0, 1.0, f64::INFINITY]),
            vec![0.0, 0.0, 2.0, 0.0]
        );
    }

    #[test]
    fn species_with_negative_fitness_get_more_offspring_when_fitter() {
        let weights = shift_to_non_negative(&[-100.0, -90.0]);
        let shares = apportion(&weights, 10);
        assert_eq!(shares, vec![0, 10]);
    }

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

    fn threshold_after_one_speciation(config: MutationConfig) -> (f64, usize) {
        let mut population =
            Population::create_population_initialized_with_config(50, 2, 1, config).with_seed(1);
        population.speciate();
        (
            population.mutation_config.compatibility_threshold,
            population.species.len(),
        )
    }

    #[test]
    fn dynamic_threshold_rises_with_too_many_species() {
        let config = MutationConfig::new()
            .compatibility_threshold(0.01)
            .target_species(3)
            .compatibility_threshold_adjustment(0.3, 0.001)
            .build();
        let (threshold, species) = threshold_after_one_speciation(config);
        assert!(species > 3, "species = {}", species);
        assert!((threshold - 0.31).abs() < 1e-9, "threshold = {}", threshold);
    }

    #[test]
    fn dynamic_threshold_drops_with_too_few_species_down_to_its_minimum() {
        let config = MutationConfig::new()
            .compatibility_threshold(3.0)
            .target_species(10)
            .compatibility_threshold_adjustment(0.25, 0.3)
            .build();
        let (threshold, species) = threshold_after_one_speciation(config);
        assert_eq!(species, 1);
        assert!((threshold - 2.75).abs() < 1e-9, "threshold = {}", threshold);
        let config = MutationConfig::new()
            .compatibility_threshold(0.4)
            .target_species(1000)
            .compatibility_threshold_adjustment(0.25, 0.3)
            .build();
        let (threshold, _) = threshold_after_one_speciation(config);
        assert!((threshold - 0.3).abs() < 1e-9, "threshold = {}", threshold);
    }

    #[test]
    fn dynamic_threshold_stays_when_on_target_or_disabled() {
        let config = MutationConfig::new()
            .compatibility_threshold(3.0)
            .target_species(1)
            .build();
        assert_eq!(threshold_after_one_speciation(config), (3.0, 1));
        let config = MutationConfig::new()
            .compatibility_threshold(0.01)
            .fixed_compatibility_threshold()
            .build();
        let (threshold, species) = threshold_after_one_speciation(config);
        assert!(species > 1);
        assert_eq!(threshold, 0.01);
    }

    #[test]
    fn default_config_aims_at_a_number_of_species() {
        let config = MutationConfig::default();
        assert_eq!(config.target_species, Some(10));
        assert_eq!(config.compatibility_excess_coefficient, 1.0);
        assert_eq!(config.compatibility_disjoint_coefficient, 1.0);
        assert_eq!(config.compatibility_weight_coefficient, 0.4);
        assert_eq!(config.compatibility_small_genome_genes, usize::MAX);
        assert_eq!(config.compatibility_threshold, 2.0);
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

    /// Genome 2 inputs -> 1 output with both weights `w`: genomes with
    /// weights 20 apart are 0.4 * 20 = 8 apart, far over the threshold 2.
    fn genome_with_weights(w: f64) -> Genome {
        let mut genome = Genome::new_unconnected(2, 1);
        genome.add_gene(Gene::new(0, 2, w, true));
        genome.add_gene(Gene::new(1, 2, w, true));
        genome
    }

    /// Population of 30 split in three species (weights 20, -20, 0) whose
    /// fitness only depends on the species: 10, 5 and 1, constant. Asexual
    /// reproduction without any mutation (children are clones), so the
    /// species stay apart.
    fn three_species_population(protected: usize) -> (Population, [i64; 3]) {
        let config = MutationConfig::new()
            .fixed_compatibility_threshold()
            .compatibility_threshold(2.0)
            .weight_mutation_rate(0.0)
            .bias_mutation_rate(0.0)
            .add_connection_rate(0.0)
            .add_neuron_rate(0.0)
            .toggle_expression_rate(0.0)
            .mutation_probability(1.0)
            .species_stagnation(15, protected)
            .build();
        let mut population =
            Population::create_population_initialized_with_config(30, 2, 1, config).with_seed(1);
        population.species = vec![];
        for w in [20.0, -20.0, 0.0] {
            let genome = genome_with_weights(w);
            let mut specie = Specie::new(genome.clone());
            for _ in 0..10 {
                specie.add(Organism::new(genome.clone()));
            }
            population.add_species(specie);
        }
        let ids = [
            population.species[0].id,
            population.species[1].id,
            population.species[2].id,
        ];
        (population, ids)
    }

    /// Fitness by species: `fitness_of(id)`; then the same bookkeeping as
    /// `evaluate_in`.
    fn evaluate_by_species(population: &mut Population, fitness_of: &dyn Fn(i64) -> f64) {
        for specie in &mut population.species {
            let fitness = fitness_of(specie.id);
            for organism in &mut specie.organisms {
                organism.fitness = fitness;
            }
            specie.adjust_fitness();
            specie.update_stagnation();
        }
    }

    /// Generation (number of `evolve` calls) after which each of `ids` was
    /// no longer in the population, checking the size every generation.
    fn run_until_removed(
        population: &mut Population,
        ids: [i64; 3],
        generations: usize,
    ) -> [Option<usize>; 3] {
        let fitness_of = move |id: i64| {
            if id == ids[0] {
                10.0
            } else if id == ids[1] {
                5.0
            } else if id == ids[2] {
                1.0
            } else {
                0.0
            }
        };
        let mut removed = [None; 3];
        evaluate_by_species(population, &fitness_of);
        for generation in 1..=generations {
            population.evolve();
            assert_eq!(population.size(), 30, "generation {}", generation);
            for (k, id) in ids.iter().enumerate() {
                if removed[k].is_none() && population.species.iter().all(|s| s.id != *id) {
                    removed[k] = Some(generation);
                }
            }
            evaluate_by_species(population, &fitness_of);
        }
        removed
    }

    #[test]
    fn species_without_improvement_for_15_generations_are_removed() {
        let (mut population, ids) = three_species_population(1);
        let removed = run_until_removed(&mut population, ids, 25);
        // Best fitness found in the first evaluation, never improved after:
        // stagnant once 15 generations went by without improvement, removed
        // in the next evolve; the species with the best organism is kept.
        assert_eq!(removed, [None, Some(16), Some(16)]);
        assert_eq!(population.size(), 30);
        assert_eq!(population.species.len(), 1);
        assert_eq!(population.species[0].id, ids[0]);
    }

    #[test]
    fn the_n_best_species_are_protected_from_stagnation() {
        let (mut population, ids) = three_species_population(2);
        let removed = run_until_removed(&mut population, ids, 25);
        assert_eq!(removed, [None, None, Some(16)]);
        assert_eq!(population.size(), 30);
    }

    #[test]
    fn species_keep_their_id_across_generations() {
        let (mut population, ids) = three_species_population(3);
        let removed = run_until_removed(&mut population, ids, 10);
        assert_eq!(removed, [None, None, None]);
        let mut current: Vec<i64> = population.species.iter().map(|s| s.id).collect();
        current.sort();
        assert_eq!(current, ids.to_vec());
        // A new species never takes an id already used.
        let mut organism = Organism::new(genome_with_weights(-40.0));
        organism.fitness = 1.0;
        population.species[0].add(organism);
        population.speciate();
        let new_ids: Vec<i64> = population
            .species
            .iter()
            .map(|s| s.id)
            .filter(|id| !ids.contains(id))
            .collect();
        assert_eq!(new_ids.len(), 1);
        assert!(new_ids[0] > *ids.iter().max().unwrap());
    }

    #[test]
    fn default_config_species_stagnation() {
        let config = MutationConfig::default();
        assert_eq!(config.species_stagnation_generations, 15);
        assert_eq!(config.species_elitism, 2);
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

    #[test]
    fn initial_organisms_take_ctrnn_settings_from_config() {
        let config = MutationConfig::new()
            .tau(0.2)
            .dt(0.005)
            .step_time(0.3)
            .build();
        let population = Population::create_population_initialized_with_config(10, 2, 1, config);
        assert!(population
            .get_organisms()
            .iter()
            .all(|o| (o.tau, o.dt, o.step_time) == (0.2, 0.005, 0.3)));
    }

    #[test]
    fn tau_defaults_keep_tau_fixed_and_out_of_the_distance() {
        let config = MutationConfig::default();
        assert_eq!((config.tau, config.dt, config.step_time), (0.01, 0.01, 0.1));
        assert_eq!(config.tau_mutation_rate, 0.0);
        assert_eq!(config.compatibility_tau_coefficient, 0.0);
        assert_eq!((config.tau_min, config.tau_max), (0.001, 1.0));
        assert_eq!(config.tau_mutate_power, 0.25);
    }
}
