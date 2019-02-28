use conv::prelude::*;
use rand::{self, distributions::{Distribution, Uniform}};
use crate::{Genome, Organism};

/// A species (several organisms) and associated fitnesses
#[derive(Debug, Clone)]
pub struct Specie<G> {
    representative: Organism<G>,
    average_fitness: f64,
    champion_fitness: f64,
    /// All orgnamisms in this species
    pub organisms: Vec<Organism<G>>,
}

const MUTATION_PROBABILITY: f64 = 0.25;
const INTERSPECIE_MATE_PROBABILITY: f64 = 0.001;
/// The fraction of organisms in a species to cull (the worst ones)
const CULL_FRACTION: f64 = 0.1;
/// The fraction of organisms in a species that are definitely mated (the best ones)
const ELITE_FRACTION: f64 = 0.2;

impl<G: Genome> Specie<G> {
    /// Create a new species from a representative Organism. Adds this organism as the only member.
    pub fn new(genome: Organism<G>) -> Specie<G> {
        Specie {
            organisms: vec![genome.clone()],
            representative: genome,
            average_fitness: 0.0,
            champion_fitness: 0.0,
        }
    }
    /// Check if another organism is of the same species as this one.
    pub fn match_genome(&self, organism: &G) -> bool {
        self.representative.genome.is_same_specie(&organism)
    }
    /// Get the most performant organism
    pub fn calculate_champion_fitness(&self) -> f64 {
        self.organisms.iter().fold(0.0, |max, organism| {
            if organism.fitness > max {
                organism.fitness
            } else {
                max
            }
        })
    }
    /// Get the average shared fitness of the organisms in the species.
    /// This is the same as the average of the real fitness of the members,
    /// divided by the number of members.
    pub fn average_shared_fitness(&self) -> f64 {
        let n_organisms = self.organisms.len().value_as::<f64>().unwrap();
        if n_organisms == 0.0 {
            return 0.0;
        }

        let avg_fitness = self.organisms.iter().map(|o| o.fitness)
            .sum::<f64>() / n_organisms;
        avg_fitness  // TODO: make actually shared???
    }

    /// Generate the next generation of genomes, which will replace the old within this species.
    pub fn generate_offspring(&mut self, n_organisms: usize, population_organisms: &[Organism<G>]) {
        if n_organisms == 0 {
            self.organisms = Vec::new();
            return;
        }
        let mut rng = rand::thread_rng();

        self.organisms.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());

        // Organisms are split into 3 parts: Those that are culled, those that are guaranteed
        // offspring through elitism, and the rest which are amenable to random selection.
        let n_elite = std::cmp::min(n_organisms, (self.organisms.len() as f64 * ELITE_FRACTION) as usize);
        let n_elite = std::cmp::max(1, n_elite);
        let first_elite = self.organisms.len() - n_elite;

        let n_random = n_organisms - n_elite;

        let n_to_cull = std::cmp::min(first_elite,
                                      (self.organisms.len() as f64 * CULL_FRACTION) as usize);


        // println!("n_offspring={}, n_organisms={}, n_elite={}, first_elite={}, n_random={}, n_to_cull={}",
                 // n_organisms, self.organisms.len(), n_elite, first_elite, n_random, n_to_cull);
        let range = Uniform::from(n_to_cull..self.organisms.len());
        let offspring: Vec<Organism<G>> =
            Iterator::chain(
                range.sample_iter(&mut rng).take(n_random),     // take n_random random organisms
                first_elite..self.organisms.len())              // and all elite organisms
            .map(|i| self.create_child(&self.organisms[i], population_organisms))
            .collect();

        self.organisms = offspring;
    }

    /// Get the representative organism of this species.
    pub fn get_representative(&self) -> Organism<G> {
        self.representative.clone()
    }
    /// Clear existing organisms in this species.
    pub fn remove_organisms(&mut self) {
        self.organisms = vec![];
    }

    /// Create a new child by mutating and existing one or mating two genomes.
    fn create_child(&self, organism: &Organism<G>, population_organisms: &[Organism<G>]) -> Organism<G> {
        if rand::random::<f64>() < MUTATION_PROBABILITY || population_organisms.len() < 2 {
            self.create_child_by_mutation(organism)
        } else {
            self.create_child_by_mate(organism, population_organisms)
        }
    }

    fn create_child_by_mutation(&self, organism: &Organism<G>) -> Organism<G> {
        organism.mutate()
    }

    fn create_child_by_mate(&self, organism: &Organism<G>, population_organisms: &[Organism<G>]) -> Organism<G> {
        let mut rng = rand::thread_rng();
        if rand::random::<f64>() > INTERSPECIE_MATE_PROBABILITY {
            let selected_mate = Uniform::from(0..self.organisms.len()).sample(&mut rng);
            organism.mate(&self.organisms[selected_mate])
        } else {
            let selected_mate = Uniform::from(0..population_organisms.len()).sample(&mut rng);
            organism.mate(&population_organisms[selected_mate])
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{nn::NeuralNetwork, Organism, Specie};
    use std::f64::EPSILON;

    #[test]
    fn specie_should_return_correct_average_fitness() {
        let mut organism1 = Organism::new(NeuralNetwork::default());
        organism1.fitness = 10.0;

        let mut organism2 = Organism::new(NeuralNetwork::default());
        organism2.fitness = 15.0;

        let mut organism3 = Organism::new(NeuralNetwork::default());
        organism3.fitness = 20.0;
        
        let mut specie = Specie::new(Organism::default());
        specie.organisms = vec![organism1, organism2, organism3];

        assert!((specie.average_shared_fitness() - 15.0).abs() < EPSILON);
    }
}
