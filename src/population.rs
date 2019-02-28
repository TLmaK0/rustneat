use crate::{Genome, Organism, Environment, Specie, NeuralNetwork};
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
}

const MAX_EPOCHS_WITHOUT_IMPROVEMENTS: usize = 15;

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
            champion_fitness: 0f64,
            generations_without_improvements: 0usize,
        }
    }

    /// Counts the number of organisms in the population
    pub fn size(&self) -> usize {
        self.species
            .iter()
            .fold(0, |total, specie| total + specie.organisms.len())
    }
    /// Collect all organisms of the population
    pub fn get_organisms(&self) -> Vec<Organism<G>> {
        self.species
            .iter()
            .flat_map(|specie| specie.organisms.clone())
            .collect::<Vec<_>>()
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
    pub fn evolve(&mut self, env: &mut Environment<G>) {
        // Collect all organisms
        let organisms = self.get_organisms();

        // Divide into species
        self.species = Population::speciate(&organisms);

        // Find champion, check if there is any improvement
        self.update_champion(&organisms);

        let sum_of_species_fitness: f64 = self.species.iter()
            .map(|specie| specie.average_shared_fitness())
            .sum();

        if self.generations_without_improvements > MAX_EPOCHS_WITHOUT_IMPROVEMENTS {
            // After a certain generations with no improvement, we prune all species except the two
            // best ones
            let mut best_species = self.get_two_best_species();
            let n_species = best_species.len();
            for specie in &mut best_species {
                specie.generate_offspring(organisms.len() / n_species, &organisms);
            }
            self.generations_without_improvements = 0;
        } else {
            // Normal case: Give each species a number of offsprings related to that species'
            // average fitness

            // Gather the average shared fitness, and the calculated number of offsprings, per species
            let species_fitness = self.species.iter()
                .map(|species| species.average_shared_fitness())
                .collect::<Vec<_>>();

            let n_offspring: Vec<_> = 
                if sum_of_species_fitness == 0.0 {
                    print!("A");
                    self.species.iter()
                        .map(|species| species.organisms.len()).collect()
                } else {
                    print!("B -- {}", self.species.len());
                    Self::partition(
                        organisms.len(),
                        &species_fitness.iter()
                            .map(|fitness|
                                 fitness / sum_of_species_fitness
                            ).collect::<Vec<_>>())
                };
            println!("(n_offsprings ={:?})", n_offspring);

            for (species, n_offspring) in self.species.iter_mut().zip(n_offspring) {
                species.generate_offspring(n_offspring, &organisms);
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
    fn partition(total: usize, fractions: &[f64]) -> Vec<usize> {
        let mut rng = rand::thread_rng();
        let mut partitions: Vec<usize> = fractions.iter().map(|x| ((total as f64 * x) as usize)).collect();
        let mut sum: usize = partitions.iter().sum();
        let range = Uniform::from(0..partitions.len());

        while sum != total {
            let residue = sum as i32 - total as i32;
            let selected = range.sample(&mut rng);
            if residue > 0 {
                partitions[selected] -= 1;
                sum -= 1;
            } else if residue < 0 && partitions[selected] > 0 {
                partitions[selected] += 1;
                sum += 1;
            }
        }
        partitions
    }

    /// Returns a Vec that contains up to 2 species, which are the two species with the maximum
    /// champion fitness.
    fn get_two_best_species(&self) -> Vec<Specie<G>> {
        if self.species.len() < 2 {
            return self.species.clone();
        }
        let mut result = vec![];
        for specie in &self.species {
            if result.len() == 0 {
                result.push(specie.clone())
            } else if result.len() == 1 {
                if result[0].calculate_champion_fitness() < specie.calculate_champion_fitness() {
                    result.insert(0, specie.clone());
                } else {
                    result.push(specie.clone());
                }
            } else if result[0].calculate_champion_fitness() < specie.calculate_champion_fitness() {
                result[1] = result[0].clone();
                result[0] = specie.clone();
            } else if result[1].calculate_champion_fitness() < specie.calculate_champion_fitness() {
                result[1] = specie.clone();
            }
        }

        result
    }

    /// Helper of `evolve`
    fn speciate(organisms: &[Organism<G>]) -> Vec<Specie<G>> {
        let mut species = Vec::<Specie<G>>::new();
        for organism in organisms {
            match species.iter_mut().find(|specie| specie.match_genome(&organism.genome)) {
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
    fn update_champion(&mut self, organisms: &[Organism<G>]) {
        let champion_fitness = organisms.iter().fold(0.0, |max, organism| {
            if organism.fitness > max {
                organism.fitness
            } else {
                max
            }
        });

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
    use crate::{Organism, Specie, NeuralNetwork, Population, Environment};

    #[test]
    fn population_should_be_able_to_speciate_genomes() {
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
        let new_species = Population::speciate(&population.species[0].organisms);

        assert_eq!(new_species.len(), 2);
    }

    #[test]
    fn after_population_evolve_population_should_be_the_same() {
        struct X;
        impl Environment<NeuralNetwork> for X {
            fn test(&self, _organism: &mut NeuralNetwork) -> f64 { 0.0 }
        }

        let mut population = Population::create_population(150);
        for _ in 0..150 {
            population.evolve(&mut X);
        }
        assert!(population.size() == 150);
    }
}
