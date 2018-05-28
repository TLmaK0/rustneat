use conv::prelude::*;
use environment::Environment;
use genome::Genome;
use organism::Organism;

#[cfg(feature = "telemetry")]
use rusty_dashed;

#[cfg(feature = "telemetry")]
use serde_json;

use specie::Specie;
use species_evaluator::SpeciesEvaluator;

/// All species in the network
#[derive(Debug)]
pub struct Population {
    /// container of species
    pub species: Vec<Specie>,
    champion_fitness: f64,
    epochs_without_improvements: usize,
}

const MAX_EPOCHS_WITHOUT_IMPROVEMENTS: usize = 5;

impl Population {
    /// Create a new population of size X.
    pub fn create_population(population_size: usize) -> Population {
        let mut population = Population {
            species: vec![],
            champion_fitness: 0f64,
            epochs_without_improvements: 0usize,
        };

        population.create_organisms(population_size);
        population
    }
    /// Find total of all orgnaisms in the population
    pub fn size(&self) -> usize {
        self.species
            .iter()
            .fold(0usize, |total, specie| total + specie.organisms.len())
    }
    /// Create offspring by mutation and mating. May create new species.
    pub fn evolve(&mut self) {
        self.generate_offspring();
    }
    /// TODO
    pub fn evaluate_in(&mut self, environment: &mut Environment) {
        let champion = SpeciesEvaluator::new(environment).evaluate(&mut self.species);

        if self.champion_fitness >= champion.fitness {
            self.epochs_without_improvements += 1;
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
    /// Return all organisms of the population
    pub fn get_organisms(&self) -> Vec<Organism> {
        self.species
            .iter()
            .flat_map(|specie| specie.organisms.clone())
            .collect::<Vec<Organism>>()
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
        } else {
            let organisms_by_average_fitness =
                num_of_organisms.value_as::<f64>().unwrap() / total_average_fitness;

            for specie in &mut self.species {
                let specie_fitness = specie.calculate_average_fitness();
                let offspring_size = if total_average_fitness == 0f64 {
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
        if self.epochs_without_improvements > MAX_EPOCHS_WITHOUT_IMPROVEMENTS {
            self.epochs_without_improvements = 0;
        }
    }

    fn get_best_species(&self) -> Vec<Specie> {
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
            let mut new_specie: Option<Specie> = None;
            match self.species
                .iter_mut()
                .find(|specie| specie.match_genome(organism))
            {
                Some(specie) => {
                    specie.add(organism.clone());
                }
                None => {
                    let mut specie = Specie::new(organism.genome.clone());
                    specie.add(organism.clone());
                    new_specie = Some(specie);
                }
            };
            if new_specie.is_some() {
                self.species.push(new_specie.unwrap());
            }
        }
    }

    fn create_organisms(&mut self, population_size: usize) {
        self.species = vec![];
        let mut organisms = vec![];

        while organisms.len() < population_size {
            organisms.push(Organism::new(Genome::default()));
        }

        let mut specie = Specie::new(organisms.first().unwrap().genome.clone());
        specie.organisms = organisms;
        self.species.push(specie);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use genome::Genome;
    use organism::Organism;
    use specie::Specie;

    #[test]
    fn population_should_be_able_to_speciate_genomes() {
        let mut genome1 = Genome::default();
        genome1.inject_gene(0, 0, 1f64);
        genome1.inject_gene(0, 1, 1f64);
        let mut genome2 = Genome::default();
        genome1.inject_gene(0, 0, 1f64);
        genome1.inject_gene(0, 1, 1f64);
        genome2.inject_gene(1, 1, 1f64);
        genome2.inject_gene(1, 0, 1f64);

        let mut population = Population::create_population(2);
        let organisms = vec![Organism::new(genome1), Organism::new(genome2)];
        let mut specie = Specie::new(organisms.first().unwrap().genome.clone());
        specie.organisms = organisms;
        population.species = vec![specie];
        population.speciate();
        assert_eq!(population.species.len(), 2usize);
    }

    #[test]
    fn after_population_evolve_population_should_be_the_same() {
        let mut population = Population::create_population(150);
        for _ in 0..150 {
            population.evolve();
        }
        assert!(population.size() == 150);
    }
}
