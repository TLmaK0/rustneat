use crate::{Environment, Genome, NeatParams, NeuralNetwork, Organism, Specie};
use rayon::prelude::*;
// use std::cmp::Ordering::*;
use rand::distributions::{Distribution, Uniform};
use std::f64;

#[cfg(feature = "telemetry")]
use rusty_dashed;

#[cfg(feature = "telemetry")]
use serde_json;

/// Contains several species, and a way to evolve these to the next generation.
#[derive(Debug)]
pub struct Population<G: Genome = NeuralNetwork> {
    /// container of species
    pub species: Vec<Specie<G>>,
    target_size: usize,
    generations_without_improvements: usize,

    /// The latest innovation id
    innovation_id: usize,
    /// To give each species a unique id. Useful for for example visualizing or
    /// processing the species.
    species_id: usize,
}

impl<G: Genome> Population<G> {
    /// Create a new population with `population_size` organisms. Each organism
    /// will have only a single unconnected neuron.
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

        let mut specie = Specie::new(organisms.first().unwrap().clone(), 0);
        specie.organisms = organisms;

        Population {
            species: vec![specie],
            target_size: population_size,
            generations_without_improvements: 0,
            innovation_id: 0,
            species_id: 1,
        }
    }

    /// Counts the number of organisms in the population
    pub fn size(&self) -> usize {
        self.species
            .iter()
            .fold(0, |total, specie| total + specie.organisms.len())
    }
    /// Collect all organisms of the population
    pub fn get_organisms(&self) -> impl Iterator<Item = &Organism<G>> {
        self.species
            .iter()
            .flat_map(|specie| specie.organisms.iter())
    }
    /// Get the best-performing organism of the entire population.
    /// Fitness is already calculated during the last call to `evolve()`
    pub fn get_champion(&self) -> Organism<G> {
        self.get_organisms()
            .fold(None, |state: Option<&Organism<G>>, organism| {
                Some(match state {
                    None => organism,
                    Some(state) => {
                        if organism.fitness > state.fitness {
                            organism
                        } else {
                            state
                        }
                    }
                })
            })
            .unwrap()
            .clone()
    }
    /// How many generations have passed without improvement in peak fitness
    pub fn generations_without_improvements(&self) -> usize {
        self.generations_without_improvements
    }

    /// Evolve to the next generation. This includes, in order:
    /// * Collecting all organisms and dividing them into (new) species
    /// * Creating a number of offsprings in each species depending on that
    ///   species' average
    /// fitness
    /// * Evaluating the fitness of all organisms
    ///
    /// Because of the last step, organisms will always have an up-to-date
    /// fitness value.
    pub fn evolve(&mut self, env: &mut dyn Environment<G>, p: &NeatParams, in_parallel: bool) {
        // Collect all organisms
        let organisms = self.get_organisms().cloned().collect::<Vec<_>>();

        // Divide into species
        self.speciate(&organisms, p);

        // Give each species a number of offsprings related to that species'
        // average fitness

        // Gather the average shared fitness, and the calculated number of offsprings,
        // per species
        let species_fitness = self
            .species
            .iter()
            .map(|species| species.average_fitness())
            .collect::<Vec<_>>();

        // let sum_of_species_fitness: f64 = species_fitness.iter().sum();

        let elite_species = (0..self.species.len()).fold((0.0, 0), |(best_f, best_i), i| {
            if self.species[i].champion_fitness() > best_f {
                (self.species[i].champion_fitness(), i)
            } else {
                (best_f, best_i)
            }
        });
        let elite_species = elite_species.1;

        let max_fitness = species_fitness.iter().cloned().fold(f64::NAN, f64::max);
        let min_fitness = species_fitness.iter().cloned().fold(f64::NAN, f64::min);
        let fitness_range = f64::max(1.0, max_fitness - min_fitness);
        let adjusted_fitness = species_fitness
            .iter()
            .map(|fitness| fitness - min_fitness + fitness_range * 0.2)
            .collect::<Vec<_>>();
        let total_adjusted_fitness = adjusted_fitness.iter().sum::<f64>();

        let n_offspring: Vec<_> = Self::partition(
            organisms.len(),
            &adjusted_fitness
                .iter()
                .map(|x| x / total_adjusted_fitness)
                .collect::<Vec<_>>(),
            elite_species,
        );

        for (species, n_offspring) in self.species.iter_mut().zip(n_offspring) {
            species.generate_offspring(n_offspring, &organisms, &mut self.innovation_id, p);
        }

        if in_parallel {
            // Evaluate the fitness of all organisms, in parallel
            self.species.par_iter_mut().for_each(|species| {
                species.organisms.par_iter_mut().for_each(|organism| {
                    organism.fitness = env.test(&mut organism.genome);
                    if organism.fitness < 0.0 {
                        eprintln!("Fitness {} < 0.0", organism.fitness);
                        std::process::exit(1);
                    }
                })
            })
        } else {
            // Evaluate the fitness of all organisms
            self.species.iter_mut().for_each(|species| {
                species.organisms.iter_mut().for_each(|organism| {
                    organism.fitness = env.test(&mut organism.genome);
                    if organism.fitness < 0.0 {
                        eprintln!("Fitness {} < 0.0", organism.fitness);
                        std::process::exit(1);
                    }
                })
            })
        }
    }

    // fn determine_new_species_sizes()

    // Helper of `evolve`. Partition `total` into partitions with size given by
    // `fractions`. We need this logic to ensure that the total stays the same
    // after partitioning. `elite` is the index of a partition that will be
    // ensured one spot
    fn partition(total: usize, fractions: &[f64], elite: usize) -> Vec<usize> {
        assert!(!fractions.is_empty());
        let mut rng = rand::thread_rng();
        let mut partitions: Vec<usize> = fractions
            .iter()
            .map(|x| ((total as f64 * x) as usize))
            .collect();
        let mut sum: usize = partitions.iter().sum();
        let range = Uniform::from(0..partitions.len());
        // println!("Fraction: {:?}", fractions);
        // println!("Partitions: {:?}", partitions);
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

    /// Helper of `evolve`
    fn speciate(&mut self, organisms: &[Organism<G>], p: &NeatParams) {
        for s in &mut self.species {
            if !s.organisms.is_empty() {
                // Pick random representative from the previous generation
                s.representative = s.organisms[rand::random::<usize>() % s.organisms.len()].clone();
                s.organisms = Vec::new();
            }
        }
        for organism in organisms {
            match self
                .species
                .iter_mut()
                .find(|specie| specie.match_genome(&organism.genome, p))
            {
                Some(specie) => {
                    specie.organisms.push(organism.clone());
                }
                None => {
                    self.species
                        .push(Specie::new(organism.clone(), self.species_id));
                    self.species_id += 1;
                }
            }
        }
        self.species.retain(|s| !s.organisms.is_empty());

        // Update champion
        self.species
            .iter_mut()
            .for_each(|specie| specie.update_champion());
        // Sort by descending fitness
        self.species.sort_by(|a, b| {
            b.champion_fitness()
                .partial_cmp(&a.champion_fitness())
                .unwrap()
        });
        // Fitness above which a species will not be removed.
        let safe_fitness = self.species[usize::min(p.species_elite - 1, self.species.len() - 1)]
            .champion_fitness();
        // Kill species that haven't improved in a awhile
        self.species.retain(|s| {
            s.age - s.age_last_improvement < p.remove_after_n_generations
                || s.champion_fitness() >= safe_fitness
        });
    }
}

#[cfg(test)]
mod tests {
    use crate::{Environment, NeatParams, NeuralNetwork, Organism, Population, Specie};

    #[test]
    fn population_should_be_able_to_speciate_genomes() {
        let p = NeatParams {
            compatibility_threshold: 0.0,
            ..NeatParams::default(1, 1)
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
        let mut specie = Specie::new(Organism::new(genome1), 0);
        specie.organisms.push(Organism::new(genome2));
        population.species = vec![specie];
        // (note: there is only one species)
        let organisms = population.get_organisms().cloned().collect::<Vec<_>>();
        population.speciate(&organisms, &p);

        assert_eq!(population.species.len(), 2);
    }

    #[test]
    fn after_population_evolve_population_should_be_the_same() {
        struct X;
        impl Environment<NeuralNetwork> for X {
            fn test(&self, _organism: &mut NeuralNetwork) -> f64 {
                0.0
            }
        }

        let p = NeatParams::default(0, 0);
        let mut population = Population::create_population(150);
        for _ in 0..150 {
            population.evolve(&mut X, &p, true);
        }
        assert!(population.size() == 150);
    }
}
