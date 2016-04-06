extern crate conv;
extern crate rand;

use self::conv::prelude::*;
use neat::genome::Genome;
use neat::organism::Organism;


#[derive(Debug, Clone)]
pub struct Specie{
    representative: Genome,
    pub organisms: Vec<Organism>
}

const MUTATION_PROBABILITY: f64 = 0.5f64;

impl Specie{
    pub fn new(genome: Genome) -> Specie{
        Specie{ organisms: vec![], representative: genome }
    }

    pub fn add(&mut self, organism: Organism){
        self.organisms.push(organism);
    }

    pub fn match_genome(&self, organism: &Organism) -> bool{
        self.representative.is_same_specie(&organism.genome)
    }

    pub fn average_fitness(&self) -> f64{
        let organisms_count = self.organisms.len().value_as::<f64>().unwrap();
        let total_fitness = self.organisms.iter().fold(0f64, |total, organism| total + organism.fitness);
        total_fitness / organisms_count 
    }

    pub fn generate_offspring(&self, num_of_organisms: usize, population_organisms: &Vec<Organism>) -> Vec<Organism>{
        let mut rng = rand::thread_rng();
        let selected_organisms = rand::sample(&mut rng, &self.organisms, num_of_organisms); 
        selected_organisms.iter().map(|organism| self.create_child(organism, population_organisms)).collect::<Vec<Organism>>()
    }

    fn create_child(&self, organism: &Organism, population_organisms: &Vec<Organism>) -> Organism {
        if rand::random::<f64>() < MUTATION_PROBABILITY || population_organisms.len() == 0 {
            let mut new_organism = organism.clone();
            new_organism.mutate();
            new_organism
        } else {
            self.create_child_by_mate(organism, population_organisms)
        }
    }

    fn create_child_by_mate(&self, organism: &Organism, population_organisms: &Vec<Organism>) -> Organism {
        unimplemented!();
    }
}

#[cfg(test)]
mod tests {
    use neat::*;

    #[test]
    fn specie_should_return_correct_average_fitness(){
        let mut specie = Specie::new(Genome::new());
        let mut organism1 = Organism::new(Genome::new());
        organism1.fitness = 10f64;

        let mut organism2 = Organism::new(Genome::new());
        organism2.fitness = 15f64;

        let mut organism3 = Organism::new(Genome::new());
        organism3.fitness = 20f64;

        specie.add(organism1);
        specie.add(organism2);
        specie.add(organism3);

        assert!(specie.average_fitness() == 15f64);
    }
}

