extern crate conv;

use self::conv::prelude::*;
use neat::*;

pub struct Population{
    organisms: Vec<Organism>
}

impl Population {
    pub fn create_population(initial_neurons: usize, population_size: usize) -> Population {
        let mut population = Population { organisms: vec![] };
        population.create_organisms(initial_neurons, population_size);
        population
    }

    pub fn size(&self) -> usize{
        self.organisms.len()
    }

    pub fn evolve(&mut self){
        self.organisms = self.generate_offspring();
    }

    fn generate_offspring(&self) -> Vec<Organism>{
        let species = self.speciate();
        //TODO: adjust species fitness to protect younger species
        let total_average_fitness = species.iter()
            .fold(0f64, |total, specie| total + specie.average_fitness());

        let num_of_organisms = self.organisms.len()
            .value_as::<f64>().unwrap();

        let organisms_by_average_fitness = num_of_organisms / total_average_fitness;

        let organisms = species.iter()
            .flat_map(|specie| specie.organisms.clone()).collect::<Vec<Organism>>();

        for specie in &species {
            specie.generate_offspring(
                    (specie.average_fitness() * organisms_by_average_fitness)
                        .round() as usize,
                     &organisms);
        }

        species.iter().flat_map(|specie| specie.organisms.clone()).collect::<Vec<Organism>>()
    }

    fn speciate(&self) -> Vec<Specie>{
        let mut species: Vec<Specie> = vec![];
        for organism in &self.organisms{
            let mut species_search = species.clone(); 
            match species_search.iter_mut().find(|specie| specie.match_genome(&organism)) {
                Some(specie) => {
                    specie.add(organism.clone());
                },
                None => {
                    let mut specie = Specie::new(organism.genome.clone());
                    specie.add(organism.clone());
                    species.push(specie);
                }
            };
        }

        species
    }

    fn create_organisms(&mut self, initial_neurons: usize, population_size: usize){
        let mut organisms = vec![];

        while organisms.len() < population_size {
            organisms.push(Organism::new(Genome::new(initial_neurons)));
        }

        self.organisms = organisms;
    }
}

#[cfg(test)]
mod tests {
    use neat::*;

    #[test]
    fn population_should_be_able_to_speciate_genomes(){
        let mut genome1 = Genome::new(10);
        genome1.create_gene(1, 1, 1f64);
        genome1.create_gene(1, 2, 1f64);
        let mut genome2 = Genome::new(10);
        genome2.create_gene(1, 3, 1f64);
        genome2.create_gene(1, 4, 1f64);

        let mut population = Population::create_population(10, 0);
        population.organisms = vec![Organism::new(genome1), Organism::new(genome2)];
        let species = population.speciate();
        assert!(species.len() == 2usize);
    }
}
