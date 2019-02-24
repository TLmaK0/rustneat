use conv::prelude::*;
use rand::{self, distributions::{Distribution, Uniform}};
use crate::{Genome, Organism};

/// A species (several organisms) and associated fitnesses
#[derive(Debug, Clone)]
pub struct Specie<G> {
    representative: Organism<G>,
    average_fitness: f64,
    champion_fitness: f64,
    age: usize,
    age_last_improvement: usize,
    /// All orgnamisms in this species
    pub organisms: Vec<Organism<G>>,
}

const MUTATION_PROBABILITY: f64 = 0.25;
const INTERSPECIE_MATE_PROBABILITY: f64 = 0.001;

impl<G: Genome> Specie<G> {
    /// Create a new species from a representative Genome
    pub fn new(genome: Organism<G>) -> Specie<G> {
        Specie {
            organisms: vec![],
            representative: genome,
            average_fitness: 0f64,
            champion_fitness: 0f64,
            age: 0,
            age_last_improvement: 0,
        }
    }
    /// Add an organism to the species
    pub fn add(&mut self, organism: Organism<G>) {
        self.organisms.push(organism);
    }
    /// Check if another organism is of the same species as this one.
    pub fn match_genome(&self, organism: &G) -> bool {
        self.representative.genome.is_same_specie(&organism)
    }
    /// Get the most performant organism
    pub fn calculate_champion_fitness(&self) -> f64 {
        self.organisms.iter().fold(0f64, |max, organism| {
            if organism.fitness > max {
                organism.fitness
            } else {
                max
            }
        })
    }
    /// Work out average fitness of this species
    pub fn calculate_average_fitness(&mut self) -> f64 {
        let organisms_count = self.organisms.len().value_as::<f64>().unwrap();
        if organisms_count == 0.0 {
            return 0.0;
        }

        let total_fitness = self.organisms
            .iter()
            .fold(0.0, |total, organism| total + organism.fitness);

        let new_fitness = total_fitness / organisms_count;

        if new_fitness > self.average_fitness {
            self.age_last_improvement = self.age;
        }

        self.average_fitness = new_fitness;
        self.average_fitness
    }

    /// Mate and generate offspring, delete old organisms and use the children
    /// as "new" species.
    pub fn generate_offspring(
        &mut self,
        num_of_organisms: usize,
        population_organisms: &[Organism<G>],
    ) {
        let mut rng = rand::thread_rng();
        self.age += 1;

        let copy_champion = if num_of_organisms > 5 { 1 } else { 0 };

        // Select `num_of_organisms` organisms in this specie, and make offspring from them.
        let mut offspring: Vec<Organism<G>> = {
            let mut selected_organisms = vec![];
            let uniform = Uniform::from(0..self.organisms.len());
            for _ in 0..num_of_organisms - copy_champion {
                selected_organisms.push(uniform.sample(&mut rng));
            }
            selected_organisms.iter()
                .map(|organism_pos| {
                    self.create_child(&self.organisms[*organism_pos], population_organisms)
                })
                .collect::<Vec<_>>()
        };

        if copy_champion == 1 {
            let champion: Option<Organism<G>> =
                self.organisms.iter().fold(None, |champion, organism| {
                    if champion.is_none() || champion.as_ref().unwrap().fitness < organism.fitness {
                        Some(organism.clone())
                    } else {
                        champion
                    }
                });

            offspring.push(champion.unwrap());
        }
        self.organisms = offspring;
    }

    /// Get the representative organism of this species.
    pub fn get_representative(&self) -> Organism<G> {
        self.representative.clone()
    }
    /// Clear existing organisms in this species.
    pub fn remove_organisms(&mut self) {
        self.adjust_fitness();
        self.organisms = vec![];
    }

    /// TODO
    pub fn adjust_fitness(&mut self) {
        // TODO: adjust fitness
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

    fn create_child_by_mate(
        &self,
        organism: &Organism<G>,
        population_organisms: &[Organism<G>],
    ) -> Organism<G> {
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
    use crate::{NeuralNetwork, Organism, Specie};
    use std::f64::EPSILON;

    #[test]
    fn specie_should_return_correct_average_fitness() {
        let mut specie = Specie::new(Organism::new(NeuralNetwork::default()));
        let mut organism1 = Organism::new(NeuralNetwork::default());
        organism1.fitness = 10f64;

        let mut organism2 = Organism::new(NeuralNetwork::default());
        organism2.fitness = 15f64;

        let mut organism3 = Organism::new(NeuralNetwork::default());
        organism3.fitness = 20f64;

        specie.organisms.push(organism1);
        specie.organisms.push(organism2);
        specie.organisms.push(organism3);

        assert!((specie.calculate_average_fitness() - 15f64).abs() < EPSILON);
    }
}
