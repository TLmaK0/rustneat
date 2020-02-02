use crate::{Environment, Genome, Organism, Specie, SpeciesEvaluator};
use conv::prelude::*;

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
    epochs_without_improvements: usize,
}

const MAX_EPOCHS_WITHOUT_IMPROVEMENTS: usize = 5;

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

        let mut specie = Specie::new(organisms.first().unwrap().clone());
        specie.organisms = organisms;

        Population {
            species: vec![specie],
            champion_fitness: 0f64,
            epochs_without_improvements: 0usize,
        }
    }

    /// Counts the number of organisms in the population
    pub fn size(&self) -> usize {
        self.species
            .iter()
            .fold(0, |total, specie| total + specie.organisms.len())
    }
    /// Create offspring by mutation and mating. May create new species.
    pub fn evolve(&mut self) {
        self.generate_offspring();
    }
    /// TODO
    pub fn evaluate_in(&mut self, environment: &mut Environment<G>) {
        let champion = SpeciesEvaluator::new(environment).evaluate(&mut self.species);

        if self.champion_fitness >= champion.fitness {
            self.epochs_without_improvements += 1;
            #[cfg(feature = "telemetry")]
            telemetry!("fitness1", 1.0, format!("{}", self.champion_fitness));
        } else {
            self.champion_fitness = champion.fitness;
            #[cfg(feature = "telemetry")]
            telemetry!("fitness1", 1.0, format!("{}", self.champion_fitness));
            #[cfg(feature = "telemetry")]
            telemetry!(
                "network1",
                1.0,
                serde_json::to_string(&champion.genome.get_genes()).unwrap()
            );
            self.epochs_without_improvements = 0usize;
        }
    }
    /// Collect all organisms of the population
    pub fn get_organisms(&self) -> Vec<Organism<G>> {
        self.species
            .iter()
            .flat_map(|specie| specie.organisms.clone())
            .collect::<Vec<_>>()
    }
    /// How many iterations without improvement
    pub fn epochs_without_improvements(&self) -> usize {
        self.epochs_without_improvements
    }

    fn generate_offspring(&mut self) {
        self.speciate();

        let total_average_fitness = self.species.iter_mut().fold(0f64, |total, specie| {
            total + specie.calculate_average_fitness()
        });

        let num_of_organisms = self.size();
        let organisms = self.get_organisms();

        if self.epochs_without_improvements > MAX_EPOCHS_WITHOUT_IMPROVEMENTS {
            let mut best_species = self.get_best_species();
            let num_of_selected = best_species.len();
            for specie in &mut best_species {
                specie.generate_offspring(
                    num_of_organisms.checked_div(num_of_selected).unwrap(),
                    &organisms,
                );
            }
            self.epochs_without_improvements = 0;
            return;
        }

        let organisms_by_average_fitness =
            num_of_organisms.value_as::<f64>().unwrap() / total_average_fitness;

        for specie in &mut self.species {
            let specie_fitness = specie.calculate_average_fitness();
            let offspring_size = if total_average_fitness <= 0f64 {
                specie.organisms.len()
            } else {
                (specie_fitness * organisms_by_average_fitness).round() as usize
            };
            if offspring_size > 0 {
                // TODO: check if offspring is for organisms fitness also, not only by specie
                specie.generate_offspring(offspring_size, &organisms);
            } else {
                specie.remove_organisms();
            }
        }
    }

    fn get_best_species(&self) -> Vec<Specie<G>> {
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

    fn speciate(&mut self) {
        let organisms = &self.get_organisms();
        for specie in &mut self.species {
            specie.remove_organisms();
        }

        for organism in organisms {
            let mut new_specie: Option<Specie<G>> = None;
            match self
                .species
                .iter_mut()
                .find(|specie| specie.match_genome(&organism.genome))
            {
                Some(specie) => {
                    specie.add(organism.clone());
                }
                None => {
                    let mut specie = Specie::new(organism.clone());
                    specie.add(organism.clone());
                    new_specie = Some(specie);
                }
            };
            if new_specie.is_some() {
                self.species.push(new_specie.unwrap());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{nn::Gene, nn::NeuralNetwork, Organism, Population, Specie};

    #[test]
    fn population_should_be_able_to_speciate_genomes() {
        let mut genome1 = NeuralNetwork::default();
        genome1.add_gene(Gene::new(0, 0, 1f64, true, false));
        genome1.add_gene(Gene::new(0, 1, 1f64, true, false));
        let mut genome2 = NeuralNetwork::default();
        genome1.add_gene(Gene::new(0, 0, 1f64, true, false));
        genome1.add_gene(Gene::new(0, 1, 1f64, true, false));
        genome2.add_gene(Gene::new(1, 1, 1f64, true, false));
        genome2.add_gene(Gene::new(1, 0, 1f64, true, false));

        let mut population = Population::create_population(2);
        let organisms = vec![Organism::new(genome1), Organism::new(genome2)];
        let mut specie = Specie::new(organisms.first().unwrap().clone());
        specie.organisms = organisms;
        population.species = vec![specie];
        population.speciate();
        assert_eq!(population.species.len(), 2usize);
    }

    #[test]
    fn after_population_evolve_population_should_be_the_same() {
        let mut population = Population::<NeuralNetwork>::create_population(150);
        for _ in 0..150 {
            population.evolve();
        }
        assert!(population.size() == 150);
    }
}
