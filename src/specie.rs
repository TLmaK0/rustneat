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
        avg_fitness / n_organisms // TODO: correct?
    }

    /// Mate and generate offspring, delete old organisms and use the children
    /// as "new" species.
    pub fn generate_offspring(&mut self, n_organisms: usize, population_organisms: &[Organism<G>]) {

        if n_organisms == 0 {
            self.organisms = Vec::new();
            return;
        }
        // TODO Review this.
        let mut rng = rand::thread_rng();

        let copy_champion = if n_organisms > 5 { 1 } else { 0 };

        // Select `n_organisms` organisms in this specie, and make offspring from them.
        let mut offspring: Vec<Organism<G>> = {
            let mut selected_organisms = vec![];
            let uniform = Uniform::from(0..self.organisms.len());
            for _ in 0..n_organisms - copy_champion {
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

        assert!((specie.average_shared_fitness() - 5.0).abs() < EPSILON);
    }
}
