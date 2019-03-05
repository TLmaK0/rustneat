use crate::{Genome, Organism, Params};
use conv::prelude::*;
use rand::{self, distributions::{Distribution, Uniform}};

/// A species (several organisms) and associated fitnesses
#[derive(Debug, Clone)]
pub struct Specie<G> {
    representative: Organism<G>,
    champion_fitness: Option<f64>,
    /// All orgnamisms in this species
    pub organisms: Vec<Organism<G>>,
}


impl<G: Genome> Specie<G> {
    /// Create a new species from a representative Organism. Adds this organism as the only member.
    pub fn new(genome: Organism<G>) -> Specie<G> {
        Specie {
            organisms: vec![genome.clone()],
            representative: genome,
            champion_fitness: None,
        }
    }
    /// Check if another organism is of the same species as this one.
    pub fn match_genome(&self, organism: &G, p: &Params) -> bool {
        self.representative.genome.is_same_specie(&organism, p)
    }
    /// Get the best fitness of this species. Stores the value internally, and uses it in
    /// subsequent calls to the function
    pub fn champion_fitness(&self) -> f64 {
        match self.champion_fitness {
            Some(fitness) => fitness,
            None => panic!("Calling Specie::champion_fitness requires that you first call calculate_champion_fitness!"),
        }
    }
    /// Calculate fitness of champion and store it internally, to be retrieved by `champion_fitness`
    pub fn calculate_champion_fitness(&mut self) {
        self.champion_fitness = Some(self.organisms.iter().fold(std::f64::NEG_INFINITY, |max, organism| {
                    if organism.fitness > max {
                        organism.fitness
                    } else {
                        max
                    }}));
    }
    /// Get the average fitness of the organisms in the species.
    pub fn average_fitness(&self) -> f64 {
        let n_organisms = self.organisms.len().value_as::<f64>().unwrap();
        if n_organisms == 0.0 {
            return 0.0;
        }

        let avg_fitness = self.organisms.iter().map(|o| o.fitness)
            .sum::<f64>() / n_organisms;
        avg_fitness
    }

    /// Generate the next generation of genomes, which will replace the old within this species.
    /// `champion_fitness`: the fitness of the population-wide champion. The reason for this
    /// parameter is that the species should see if it is the best-performing one.
    pub fn generate_offspring(&mut self, n_offspring: usize,
                                         population_offspring: &[Organism<G>],
                                         p: &Params) {
        if n_offspring == 0 {
            self.organisms = Vec::new();
            return;
        }
        let mut rng = rand::thread_rng();

        self.organisms.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());

        // Organisms are split into 3 parts: Those that are culled, those that are guaranteed
        // offspring through elitism, and the rest which are amenable to random selection.
        // NOTE: For now, we always have n_elite = 2 or 0.

        // let n_elite = std::cmp::min(n_offspring, (self.organisms.len() as f64 * ELITE_FRACTION) as usize);
        // let n_elite = std::cmp::max(1, n_elite);
        let n_elite = if self.organisms.len() > 5 {
            1
        } else {
            0
        };
        let first_elite = self.organisms.len() - n_elite;

        let n_random = n_offspring - n_elite;

        let n_to_cull = std::cmp::min(first_elite,
                                      (self.organisms.len() as f64 * p.cull_fraction) as usize);


        // println!("n_offspring={}, n_offspring={}, n_elite={}, first_elite={}, n_random={}, n_to_cull={}",
                 // n_offspring, self.organisms.len(), n_elite, first_elite, n_random, n_to_cull);
        let range = Uniform::from(n_to_cull..self.organisms.len());
        let offspring: Vec<Organism<G>> =
            Iterator::chain(
                // mate n_random random organisms
                range.sample_iter(&mut rng).take(n_random)
                    .map(|i| self.create_child(&self.organisms[i], population_offspring, p)),
                // copy elite organisms
                (first_elite..self.organisms.len())
                    .map(|i| self.organisms[i].clone())
            )
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
    fn create_child(&self, organism: &Organism<G>, population_organisms: &[Organism<G>], p: &Params) -> Organism<G> {
        let mut child = self.create_child_by_mate(organism, population_organisms, p);

        if rand::random::<f64>() < p.mutation_pr {
            child.mutate(p)
        }
        child
    }

    fn create_child_by_mate(&self, organism: &Organism<G>, population_organisms: &[Organism<G>], p: &Params) -> Organism<G> {
        let mut rng = rand::thread_rng();
        if rand::random::<f64>() > p.interspecie_mate_pr {
            let selected_mate = Uniform::from(0..self.organisms.len()).sample(&mut rng);
            organism.mate(&self.organisms[selected_mate], p)
        } else {
            let selected_mate = Uniform::from(0..population_organisms.len()).sample(&mut rng);
            organism.mate(&population_organisms[selected_mate], p)
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
