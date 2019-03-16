use crate::{Genome, Organism, Environment, Specie, NeuralNetwork, Params};
use rayon::prelude::*;
// use std::cmp::Ordering::*;
use rand::distributions::{Uniform, Distribution};

#[cfg(feature = "telemetry")]
use rusty_dashed;

#[cfg(feature = "telemetry")]
use serde_json;


/// Contains several species, and a way to evolve these to the next generation.
#[derive(Debug)]
pub struct Population<G = NeuralNetwork> {
    /// container of species
    pub species: Vec<Specie<G>>,
    target_size: usize,
    champion_fitness: f64,
    generations_without_improvements: usize,

    /// The latest innovation id
    innovation_id: usize,
}

impl<G: Genome> Population<G> {
    /// Create a new population with `population_size` organisms. Each organism will have only a single unconnected
    /// neuron.
    pub fn create_population(population_size: usize) -> Population<G> {
        Self::create_population_from(G::default(), population_size)
    }
    /// Create a new population with `population_size` organisms,
    /// where each organism has the same genome given in `genome`.
    pub fn create_population_from(genome: G, population_size: usize) -> Population<G> {
        let mut organisms = Vec::new();
        while organisms.len() < population_size {
            organisms.push(Organism::new(genome.clone()));
        }

        let mut specie = Specie::new(organisms.first().unwrap().clone());
        specie.organisms = organisms;

        Population {
            species: vec![specie],
            target_size: population_size,
            champion_fitness: 0.0,
            generations_without_improvements: 0,
            innovation_id: 0,
        }
    }

    /// Counts the number of organisms in the population
    pub fn size(&self) -> usize {
        self.species
            .iter()
            .fold(0, |total, specie| total + specie.organisms.len())
    }
    /// Collect all organisms of the population
    pub fn get_organisms<'a>(&'a self) -> impl Iterator<Item = &'a Organism<G>> {
        self.species
            .iter()
            .flat_map(|specie| specie.organisms.iter())
    }
    /// Get the best-performing organism of the entire population.
    /// Fitness is already calculated during the last call to `evolve()`
    pub fn get_champion(&self) -> Organism<G> {
        self.get_organisms().fold(None,
            |state: Option<&Organism<G>>, organism|
                Some(match state {
                    None => organism,
                    Some(state) => if organism.fitness > state.fitness {
                        organism
                    } else {
                        state
                    }
                })
        ).unwrap().clone()
    }
    /// How many generations have passed without improvement in peak fitness
    pub fn generations_without_improvements(&self) -> usize {
        self.generations_without_improvements
    }

    /// Evolve to the next generation. This includes, in order:
    /// * Collecting all organisms and dividing them into (new) species
    /// * Creating a number of offsprings in each species depending on that species' average
    /// fitness
    /// * Evaluating the fitness of all organisms
    ///
    /// Because of the last step, organisms will always have an up-to-date fitness value.
    pub fn evolve(&mut self, env: &mut Environment<G>, p: &Params) {
        // Collect all organisms
        let organisms = self.get_organisms().cloned().collect::<Vec<_>>();

        // Divide into species
        self.species = Population::speciate(&organisms, p);

        self.species.iter_mut().for_each(|specie| specie.calculate_champion_fitness());

        // Find champion, check if there is any improvement
        self.update_champion();

        if self.generations_without_improvements > p.prune_after_n_generations {
            // After a certain generations with no improvement, we prune all species except the two
            // best ones
            self.prune_species();
            let n_species = self.species.len();
            for specie in &mut self.species {
                specie.generate_offspring(organisms.len() / n_species, &organisms, &mut self.innovation_id, p);
            }
            self.generations_without_improvements = 0;
        } else {
            // Normal case: Give each species a number of offsprings related to that species'
            // average fitness

            // Gather the average shared fitness, and the calculated number of offsprings, per species
            let species_fitness = self.species.iter()
                .map(|species| species.average_fitness())
                .collect::<Vec<_>>();

            let sum_of_species_fitness: f64 = species_fitness.iter().sum();

            let elite_species = (0..self.species.len()).fold((0.0, 0), |(best_f, best_i), i| {
                if self.species[i].champion_fitness() > best_f {
                    (self.species[i].champion_fitness(), i)
                } else {
                    (best_f, best_i)
                }
            });
            let elite_species = elite_species.1;

            let n_offspring: Vec<_> = 
                if sum_of_species_fitness == 0.0 {
                    self.species.iter()
                        .map(|species| species.organisms.len()).collect()
                } else {
                    Self::partition(
                        organisms.len(),
                        &species_fitness.iter()
                            .map(|fitness| fitness / sum_of_species_fitness)
                            .collect::<Vec<_>>(),
                        elite_species)
                };

            for (species, n_offspring) in self.species.iter_mut().zip(n_offspring) {
                species.generate_offspring(n_offspring, &organisms, &mut self.innovation_id, p);
            }
        }

        // Evaluate the fitness of all organisms, in parallel
        self.species.par_iter_mut()
            .for_each(|species| species.organisms.par_iter_mut()
                .for_each(|organism| {
                    organism.fitness = env.test(&mut organism.genome);
                    if organism.fitness < 0.0 {
                        eprintln!("Fitness {} < 0.0", organism.fitness);
                        std::process::exit(-1);
                    }
                }))
    }

    // Helper of `evolve`. Partition `total` into partitions with size given by `fractions`.
    // We need this logic to ensure that the total stays the same after partitioning.
    // `elite` is the index of a partition that will be ensured one spot
    fn partition(total: usize, fractions: &[f64], elite: usize) -> Vec<usize> {
        let mut rng = rand::thread_rng();
        let mut partitions: Vec<usize> = fractions.iter().map(|x| ((total as f64 * x) as usize)).collect();
        let mut sum: usize = partitions.iter().sum();
        let range = Uniform::from(0..partitions.len());
        while sum != total {
            let residue = sum as i32 - total as i32;
            let selected = range.sample(&mut rng);
            if residue > 0 && partitions[selected] > 0 {
                partitions[selected] -= 1;
                sum -= 1;
            } else if residue < 0 {
                partitions[selected] += 1;
                sum += 1;
            }
        }
        // Ensure that the elite gets a spot
        while partitions[elite] == 0 {
            let selected = range.sample(&mut rng);
            if partitions[selected] > 0 {
                partitions[elite] = 1;
            }
        }
        partitions
    }

    /// Leaves only the best 2 species
    fn prune_species(&mut self) {
        const N: usize = 2;
        if self.species.len() < N {
            return;
        }
        self.species.sort_by(|a, b| {
            b.champion_fitness().partial_cmp(&a.champion_fitness()).unwrap()
        });
        self.species.truncate(N);
    }

    /// Helper of `evolve`
    fn speciate(organisms: &[Organism<G>], p: &Params) -> Vec<Specie<G>> {
        let mut species = Vec::<Specie<G>>::new();
        for organism in organisms {
            match species.iter_mut().find(|specie| specie.match_genome(&organism.genome, p)) {
                Some(specie) => {
                    specie.organisms.push(organism.clone());
                }
                None => {
                    species.push(Specie::new(organism.clone()));
                }
            }
        }
        species
    }
    /// Helper of `evolve`. Find champion, and check if there is any improvement
    fn update_champion(&mut self) {
        let champion_fitness = self.get_champion().fitness;
        if self.champion_fitness >= champion_fitness {
            self.generations_without_improvements += 1;
        } else {
            self.champion_fitness = champion_fitness;
            #[cfg(feature = "telemetry")]
            telemetry!("fitness1", 1.0, format!("{}", self.champion_fitness));
            #[cfg(feature = "telemetry")]
            telemetry!(
                "network1",
                1.0,
                serde_json::to_string(&champion.genome.get_genes()).unwrap()
            );
            self.generations_without_improvements = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{Organism, Specie, NeuralNetwork, Population, Environment, Params};

    #[test]
    fn population_should_be_able_to_speciate_genomes() {
        let p = Params {
            compatibility_threshold: 0.0,
            ..Default::default()
        };
        let mut genome1 = NeuralNetwork::with_neurons(2);
        genome1.add_connection(0, 0, 1.0);
        genome1.add_connection(0, 1, 1.0);
        let mut genome2 = NeuralNetwork::with_neurons(2);
        genome1.add_connection(0, 0, 1.0);
        genome1.add_connection(0, 1, 1.0);
        genome2.add_connection(1, 1, 1.0);
        genome2.add_connection(1, 0, 1.0);

        let mut population = Population::create_population(2);
        let mut specie = Specie::new(Organism::new(genome1));
        specie.organisms.push(Organism::new(genome2));
        population.species = vec![specie];
        // (note: there is only one species)
        let new_species = Population::speciate(&population.species[0].organisms, &p);

        assert_eq!(new_species.len(), 2);
    }

    #[test]
    fn after_population_evolve_population_should_be_the_same() {
        struct X;
        impl Environment<NeuralNetwork> for X {
            fn test(&self, _organism: &mut NeuralNetwork) -> f64 { 0.0 }
        }

        let p = Params::default();
        let mut population = Population::create_population(150);
        for _ in 0..150 {
            population.evolve(&mut X, &p);
        }
        assert!(population.size() == 150);
    }
}
