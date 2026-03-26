use crate::genome::Genome;
use crate::mutation_config::MutationConfig;
use crate::organism::Organism;
use conv::prelude::*;
use rand;
use rand::Rng;

/// A species (several organisms) and associated fitnesses
#[derive(Debug, Clone)]
pub struct Specie {
    representative: Genome,
    average_fitness: f64,
    champion_fitness: f64,
    age: usize,
    age_last_improvement: usize,
    /// All orgnamisms in this species
    pub organisms: Vec<Organism>,
    /// Allows to set an id to identify it
    pub id: i64,
}

const INTERSPECIE_MATE_PROBABILITY: f64 = 0.15f64; // Increased from 0.03 to escape local optima
const BEST_ORGANISMS_THRESHOLD: f64 = 0.5f64; // Only top 50% can reproduce

impl Specie {
    /// Create a new species from a Genome
    pub fn new(genome: Genome) -> Specie {
        Specie {
            organisms: vec![],
            representative: genome,
            average_fitness: 0f64,
            champion_fitness: 0f64,
            age: 0,
            age_last_improvement: 0,
            id: 0,
        }
    }

    /// Add an Organism
    pub fn add(&mut self, organism: Organism) {
        self.organisms.push(organism);
    }

    /// Check if another organism is of the same species as this one.
    pub fn match_genome(&self, organism: &Organism) -> bool {
        self.representative.is_same_specie(&organism.genome)
    }

    /// Check if another organism is of the same species using a custom threshold.
    pub fn match_genome_with_threshold(&self, organism: &Organism, threshold: f64) -> bool {
        self.representative
            .is_same_specie_with_threshold(&organism.genome, threshold)
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

    /// Update stagnation tracking. Call after evaluation.
    /// Returns true if species improved this generation.
    pub fn update_stagnation(&mut self) -> bool {
        let current_best = self.calculate_champion_fitness();
        if current_best > self.champion_fitness {
            self.champion_fitness = current_best;
            self.age_last_improvement = self.age;
            true
        } else {
            false
        }
    }

    /// Check if species has stagnated (no improvement for max_generations)
    pub fn is_stagnant(&self, max_generations: usize) -> bool {
        self.age > self.age_last_improvement + max_generations
    }

    /// Get generations since last improvement
    pub fn generations_without_improvement(&self) -> usize {
        self.age.saturating_sub(self.age_last_improvement)
    }

    /// Work out average fitness of this species
    pub fn calculate_average_fitness(&mut self) -> f64 {
        let organisms_count = self.organisms.len().value_as::<f64>().unwrap();
        if organisms_count == 0f64 {
            return 0f64;
        }

        let total_fitness = self
            .organisms
            .iter()
            .fold(0f64, |total, organism| total + organism.fitness);

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
        population_organisms: &[Organism],
    ) {
        self.generate_offspring_with_config(
            num_of_organisms,
            population_organisms,
            &MutationConfig::default(),
        );
    }

    /// Generate offspring with a specific mutation config (supports adaptive mutation)
    pub fn generate_offspring_with_config(
        &mut self,
        num_of_organisms: usize,
        population_organisms: &[Organism],
        base_config: &MutationConfig,
    ) {
        self.age += 1;

        // Always copy champion (elitism)
        let copy_champion = if num_of_organisms > 1 { 1 } else { 0 };

        let mut organisms_to_mate =
            (self.organisms.len() as f64 * BEST_ORGANISMS_THRESHOLD) as usize;
        if organisms_to_mate < 1 {
            organisms_to_mate = 1;
        }

        self.organisms.sort();
        self.organisms.truncate(organisms_to_mate);

        let mut offspring: Vec<Organism> = {
            // Fitness-proportionate selection (roulette wheel) using adjusted_fitness
            let mut rng = rand::thread_rng();

            // Calculate total adjusted fitness for roulette wheel
            let total_adjusted_fitness: f64 = self
                .organisms
                .iter()
                .map(|o| o.adjusted_fitness.max(0.0))
                .sum();

            let mut selected_organisms = vec![];
            for _ in 0..num_of_organisms - copy_champion {
                if total_adjusted_fitness <= 0.0 {
                    // Fallback to uniform selection if no positive fitness
                    selected_organisms.push(rng.gen_range(0, self.organisms.len()));
                } else {
                    // Roulette wheel selection
                    let spin = rng.gen_range(0.0, total_adjusted_fitness);
                    let mut cumulative = 0.0;
                    let mut selected = 0;
                    for (i, organism) in self.organisms.iter().enumerate() {
                        cumulative += organism.adjusted_fitness.max(0.0);
                        if cumulative >= spin {
                            selected = i;
                            break;
                        }
                    }
                    selected_organisms.push(selected);
                }
            }
            selected_organisms
                .iter()
                .map(|organism_pos| {
                    self.create_child(
                        &self.organisms[*organism_pos],
                        population_organisms,
                        base_config,
                    )
                })
                .collect::<Vec<Organism>>()
        };

        if copy_champion == 1 {
            let champion: Option<Organism> =
                self.organisms.iter().fold(None, |champion, organism| {
                    if champion.is_none() || champion.as_ref().unwrap().fitness < organism.fitness {
                        Some(organism.clone())
                    } else {
                        champion
                    }
                });

            // Mark champion copy to preserve its fitness (skip re-evaluation)
            let mut elite = champion.unwrap();
            elite.preserve_fitness = true;
            offspring.push(elite);
        }
        self.organisms = offspring;
    }

    /// Choice a new representative of the specie at random
    pub fn choose_new_representative(&mut self) {
        self.representative = rand::thread_rng()
            .choose(&self.organisms)
            .unwrap()
            .genome
            .clone();
    }

    /// Get a genome representitive of this species.
    pub fn get_representative_genome(&self) -> Genome {
        self.representative.clone()
    }

    /// Clear existing organisms in this species.
    pub fn remove_organisms(&mut self) {
        self.organisms = vec![];
    }

    /// Returns true if specie hasn't organisms
    pub fn is_empty(&self) -> bool {
        self.organisms.is_empty()
    }

    /// Apply fitness sharing: divide each organism's fitness by species size
    /// This prevents large species from dominating the population
    pub fn adjust_fitness(&mut self) {
        let species_size = self.organisms.len() as f64;
        if species_size == 0.0 {
            return;
        }
        for organism in &mut self.organisms {
            organism.adjusted_fitness = organism.fitness / species_size;
        }
    }

    /// Create a new child by crossover+mutation or mutation only.
    /// Per NEAT paper: 75% crossover (then mutate), 25% mutation only.
    fn create_child(
        &self,
        organism: &Organism,
        population_organisms: &[Organism],
        config: &MutationConfig,
    ) -> Organism {
        if rand::random::<f64>() < config.mutation_probability || population_organisms.len() < 2 {
            // 25%: mutation only (asexual reproduction)
            organism.mutate_with_config(config)
        } else {
            // 75%: crossover then mutate
            let child = self.create_child_by_mate(organism, population_organisms);
            child.mutate_with_config(config)
        }
    }

    fn create_child_by_mate(
        &self,
        organism: &Organism,
        population_organisms: &[Organism],
    ) -> Organism {
        let mut rng = rand::thread_rng();
        if rand::random::<f64>() > INTERSPECIE_MATE_PROBABILITY {
            let selected_mate =
                rand::seq::sample_iter(&mut rng, 0..self.organisms.len(), 1).unwrap()[0];
            organism.mate(&self.organisms[selected_mate])
        } else {
            let selected_mate =
                rand::seq::sample_iter(&mut rng, 0..population_organisms.len(), 1).unwrap()[0];
            organism.mate(&population_organisms[selected_mate])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::Genome;
    use crate::organism::Organism;
    use std::f64::EPSILON;

    #[test]
    fn specie_should_return_correct_average_fitness() {
        let mut specie = Specie::new(Genome::default());
        let mut organism1 = Organism::new(Genome::default());
        organism1.fitness = 10f64;

        let mut organism2 = Organism::new(Genome::default());
        organism2.fitness = 15f64;

        let mut organism3 = Organism::new(Genome::default());
        organism3.fitness = 20f64;

        specie.add(organism1);
        specie.add(organism2);
        specie.add(organism3);

        assert!((specie.calculate_average_fitness() - 15f64).abs() < EPSILON);
    }

    #[test]
    fn adjust_fitness_should_divide_by_species_size() {
        let mut specie = Specie::new(Genome::default());

        let mut organism1 = Organism::new(Genome::default());
        organism1.fitness = 30.0;

        let mut organism2 = Organism::new(Genome::default());
        organism2.fitness = 60.0;

        let mut organism3 = Organism::new(Genome::default());
        organism3.fitness = 90.0;

        specie.add(organism1);
        specie.add(organism2);
        specie.add(organism3);

        specie.adjust_fitness();

        // Each fitness should be divided by species size (3)
        assert!((specie.organisms[0].adjusted_fitness - 10.0).abs() < EPSILON);
        assert!((specie.organisms[1].adjusted_fitness - 20.0).abs() < EPSILON);
        assert!((specie.organisms[2].adjusted_fitness - 30.0).abs() < EPSILON);
    }
}
