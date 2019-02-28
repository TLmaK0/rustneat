use crate::{Genome, Organism, Environment, Specie};
use rayon::prelude::*;

#[cfg(feature = "telemetry")]
use rusty_dashed;

#[cfg(feature = "telemetry")]
use serde_json;


/// Contains several species, and a way to evolve these to the next generation.
#[derive(Debug)]
pub struct Population<G> {
    /// container of species
    pub species: Vec<Specie<G>>,
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
            let mut best_species = self.get_best_species();
            let n_species = best_species.len();
            for specie in &mut best_species {
                specie.generate_offspring(organisms.len() / n_species, &organisms);
            }
            self.generations_without_improvements = 0;
        } else {
            // Normal case: Give each species a number of offsprings related to that species
            // average fitness

            let offspring_per_fitness = organisms.len() as f64 / sum_of_species_fitness;

            for specie in &mut self.species {
                let specie_fitness = specie.average_shared_fitness();
                let offspring_size = if sum_of_species_fitness == 0.0 {
                    specie.organisms.len()
                } else {
                    (specie_fitness * offspring_per_fitness).round() as usize
                };
                specie.generate_offspring(offspring_size, &organisms);
            }
        }

        // Evaluate the fitness of all organisms
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

    fn get_best_species(&self) -> Vec<Specie<G>> {
        // TODO rewrite
        let mut result = vec![];

        if self.species.len() < 2 {
            return self.species.clone();
        }

        for specie in &self.species {
            if result.len() < 1 {
                result.push(specie.clone())
            } else if result.len() < 2 {
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
    use crate::{nn::ConnectionGene, Organism, Specie, nn::NeuralNetwork, Population, Environment};

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
            fn test(&self, organism: &mut NeuralNetwork) -> f64 { 0.0 }
        }

        let mut population = Population::<NeuralNetwork>::create_population(150);
        for _ in 0..150 {
            population.evolve(&mut X);
        }
        assert!(population.size() == 150);
    }
}
