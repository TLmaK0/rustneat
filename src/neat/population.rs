extern crate conv;

use self::conv::prelude::*;
use neat::*;


#[derive(Debug)]
pub struct Population{
    pub organisms: Vec<Organism>
}

impl Population {
    pub fn create_population(population_size: usize) -> Population {
        let mut population = Population { organisms: vec![] };
        population.create_organisms(population_size);
        population
    }

    pub fn size(&self) -> usize{
        self.organisms.len()
    }

    pub fn evolve(&mut self){
        self.organisms = self.generate_offspring();
    }

    pub fn evaluate_in(&mut self, environment: &Environment){
        for organism in &mut self.organisms {
            organism.fitness = environment.test(organism);
        }
    }

    fn generate_offspring(&self) -> Vec<Organism>{
        let mut species = self.speciate();
        let total_average_fitness = species.iter()
            .fold(0f64, |total, specie| total + specie.average_fitness());

        let num_of_organisms = self.organisms.len()
            .value_as::<f64>().unwrap();

        let organisms_by_average_fitness = num_of_organisms / total_average_fitness;

        let organisms = species.iter()
            .flat_map(|specie| specie.organisms.clone()).collect::<Vec<Organism>>();
        for specie in &mut species {
            let mut offspring_size = (specie.average_fitness() * organisms_by_average_fitness).round() as usize;
            if total_average_fitness == 0f64 {
                offspring_size = specie.organisms.len();
            }
            specie.generate_offspring(offspring_size, &organisms);
        }
        species.iter().flat_map(|specie| specie.organisms.clone()).collect::<Vec<Organism>>()
    }

    fn speciate(&self) -> Vec<Specie>{
        let mut species: Vec<Specie> = vec![];
        for organism in &self.organisms{
            let mut new_specie: Option<Specie> = None;
            match species.iter_mut().find(|specie| specie.match_genome(&organism)) {
                Some(specie) => {
                    specie.add(organism.clone());
                },
                None => {
                    let mut specie = Specie::new(organism.genome.clone());
                    specie.add(organism.clone());
                    new_specie = Some(specie);
                }
            };
            if new_specie.is_some() {
                species.push(new_specie.unwrap());
            }
        }
        species
    }

    fn create_organisms(&mut self, population_size: usize){
        let mut organisms = vec![];

        while organisms.len() < population_size {
            organisms.push(Organism::new(Genome::new()));
        }

        self.organisms = organisms;
    }
}

#[cfg(test)]
mod tests {
    use neat::*;

    #[test]
    fn population_should_be_able_to_speciate_genomes(){
        let mut genome1 = Genome::new();
        genome1.inject_gene(0, 0, 1f64);
        genome1.inject_gene(0, 1, 1f64);
        let mut genome2 = Genome::new();
        genome1.inject_gene(0, 0, 1f64);
        genome1.inject_gene(0, 1, 1f64);
        genome2.inject_gene(1, 1, 1f64);
        genome2.inject_gene(1, 0, 1f64);

        let mut population = Population::create_population(0);
        population.organisms = vec![Organism::new(genome1), Organism::new(genome2)];
        let species = population.speciate();
        assert!(species.len() == 2usize);
    }

    #[test]
    fn after_population_evolve_population_should_be_the_same(){
        let mut population = Population::create_population(150);
        for _ in 0..150 {
            population.evolve();
        }
        assert!(population.organisms.len() == 150);
    }
}
