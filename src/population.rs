use conv::prelude::*;
use environment::Environment;
use genome::Genome;
use organism::Organism;
use std::cmp::Ordering;
#[cfg(feature = "telemetry")]
use std::time::{SystemTime, UNIX_EPOCH};

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
    /// champion of the population
    pub champion: Option<Organism>,
}

const MAX_EPOCHS_WITHOUT_IMPROVEMENTS: usize = 10;

impl Population {
    /// Create a new population of size X.
    pub fn create_population(population_size: usize) -> Population {
        let mut population = Population {
            species: vec![],
            champion_fitness: 0f64,
            champion: None,
            epochs_without_improvements: 0usize,
        };

        population.create_organisms(population_size);
        population
    }

    /// Create a population of size X with where every organisms has initial input and output neurons
    pub fn create_population_initialized(
        population_size: usize,
        input_neurons: usize,
        output_neurons: usize,
    ) -> Population {
        let mut population = Population {
            species: vec![],
            champion_fitness: 0f64,
            champion: None,
            epochs_without_improvements: 0usize,
        };

        population.create_organisms_initialized(population_size, input_neurons, output_neurons);
        population
    }

    /// Find total of all organisms in the population
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
    pub fn evaluate_in(&mut self, environment: &mut dyn Environment) {
        let champion = SpeciesEvaluator::new(environment).evaluate(&mut self.species);

        #[cfg(feature = "telemetry")]
        telemetry!("fitness1", 1.0, format!("{}", self.champion_fitness));

        if self.champion_fitness >= champion.fitness {
            self.epochs_without_improvements += 1;
        } else {
            #[cfg(feature = "telemetry")]
            telemetry!(
                "network1",
                1.0,
                serde_json::to_string(&champion.genome.get_genes()).unwrap()
            );
            self.epochs_without_improvements = 0usize;
            self.champion = Some(champion.clone());
        }
        self.champion_fitness = champion.fitness;
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
                specie.generate_offspring(offspring_size, &organisms);
            } else {
                specie.remove_organisms();
            }
        }
    }

    fn get_best_species(&mut self) -> Vec<Specie> {
        if self.species.len() <= 2 {
            return self.species.clone();
        }

        self.species.sort_by(|specie1, specie2| {
            if specie1.calculate_champion_fitness() > specie2.calculate_champion_fitness() {
                Ordering::Greater
            } else {
                Ordering::Less
            }
        });

        self.species[1..2].to_vec().clone()
    }

    fn speciate(&mut self) {
        let organisms = &self.get_organisms();
        self.species.retain(|specie| !specie.is_empty());

        let mut next_specie_id = 0i64;

        #[cfg(feature = "telemetry")]
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();

        for specie in &mut self.species {
            #[cfg(feature = "telemetry")]
            telemetry!(
                "species1",
                1.0,
                format!(
                    "{{'id':{}, 'fitness':{}, 'organisms':{}, 'timestamp':'{:?}'}}",
                    specie.id,
                    specie.calculate_champion_fitness(),
                    specie.organisms.len(),
                    now
                )
            );

            specie.choose_new_representative();

            specie.remove_organisms();

            specie.id = next_specie_id;
            next_specie_id += 1;
        }

        for organism in organisms {
            match self
                .species
                .iter_mut()
                .find(|specie| specie.match_genome(organism))
            {
                Some(specie) => {
                    specie.add(organism.clone());
                }
                None => {
                    let mut specie = Specie::new(organism.genome.clone());
                    specie.id = next_specie_id;
                    specie.add(organism.clone());
                    next_specie_id += 1;
                    self.species.push(specie);
                }
            };
        }
        self.species.retain(|specie| !specie.is_empty());
    }

    fn create_organisms_initialized(
        &mut self,
        population_size: usize,
        input_neurons: usize,
        output_neurons: usize,
    ) {
        self.species = vec![];
        let mut organisms = vec![];

        while organisms.len() < population_size {
            organisms.push(Organism::new(Genome::new_initialized(
                input_neurons,
                output_neurons,
            )));
        }

        let mut specie = Specie::new(organisms.first().unwrap().genome.clone());
        specie.organisms = organisms;
        self.species.push(specie);
    }

    fn create_organisms(&mut self, population_size: usize) {
        self.create_organisms_initialized(population_size, 0, 0);
    }
}

#[cfg(test)]
use gene::Gene;

#[cfg(test)]
mod tests {
    use super::*;
    use genome::Genome;
    use organism::Organism;
    use specie::Specie;

    #[test]
    fn population_should_be_able_to_speciate_genomes() {
        let mut genome1 = Genome::default();
        genome1.add_gene(Gene::new(0, 0, 1f64, true, false));
        genome1.add_gene(Gene::new(0, 1, 1f64, true, false));
        let mut genome2 = Genome::default();
        genome1.add_gene(Gene::new(0, 0, 1f64, true, false));
        genome1.add_gene(Gene::new(0, 1, 1f64, true, false));
        genome2.add_gene(Gene::new(1, 1, 1f64, true, false));
        genome2.add_gene(Gene::new(1, 0, 1f64, true, false));

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
