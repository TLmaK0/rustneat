extern crate conv;
extern crate rand;

use self::conv::prelude::*;
use neat::genome::Genome;
use neat::organism::Organism;
use self::rand::distributions::{IndependentSample, Range};

#[derive(Debug, Clone)]
pub struct Specie{
    representative: Genome,
    average_fitness: f64,
    champion_fitness: f64,
    age: usize,
    age_last_improvement: usize,
    pub organisms: Vec<Organism>
}

const MUTATION_PROBABILITY: f64 = 0.25f64;
const INTERSPECIE_MATE_PROBABILITY: f64 = 0.001f64;

impl Specie{
    pub fn new(genome: Genome) -> Specie{
        Specie{ organisms: vec![], representative: genome, average_fitness: 0f64, champion_fitness: 0f64, age: 0, age_last_improvement: 0 }
    }

    pub fn add(&mut self, organism: Organism){
        self.organisms.push(organism);
    }

    pub fn match_genome(&self, organism: &Organism) -> bool{
        self.representative.is_same_specie(&organism.genome)
    }

    pub fn calculate_champion_fitness(&self) -> f64{
        self.organisms.iter().fold(0f64, |max, organism| {
            if organism.fitness > max {
                organism.fitness
            } else {
                max
            }
        })
    }

    pub fn calculate_average_fitness(&mut self) -> f64{
        let organisms_count = self.organisms.len().value_as::<f64>().unwrap();
        if organisms_count == 0f64 {
            return 0f64;
        }

        let total_fitness = self.organisms.iter().fold(0f64, |total, organism| {
            total + organism.fitness
        });

        //TODO: check if total fitness should be an average or champion fitness
        let new_fitness = total_fitness / organisms_count;

        if new_fitness > self.average_fitness {
            self.age_last_improvement = self.age;
        }

        self.average_fitness = new_fitness;
        self.average_fitness
    }

    pub fn generate_offspring(&mut self, num_of_organisms: usize, population_organisms: &Vec<Organism>){
        self.age += 1;

        let copy_champion = if num_of_organisms > 5 { 1 } else { 0 };

        let mut rng = rand::thread_rng();
        let mut offspring: Vec<Organism> = {
            let mut selected_organisms = vec![];
            let range = Range::new(0, self.organisms.len());
            for _ in 0..num_of_organisms - copy_champion {
                selected_organisms.push(range.ind_sample(&mut rng));
            }
            selected_organisms.iter().map(|organism_pos| self.create_child(&self.organisms[*organism_pos], population_organisms)).collect::<Vec<Organism>>()
        };

        if copy_champion == 1 {
            let champion: Option<Organism> = self.organisms.iter().fold(None, |champion, organism| {
                if champion.is_none() || champion.as_ref().unwrap().fitness < organism.fitness {
                    Some(organism.clone())
                } else {
                    champion
                }
            });

            offspring.push(champion.unwrap());
        }
println!("Generated: #{:?}", offspring.len());
        self.organisms = offspring;
    }

    pub fn get_representative_genome(&self) -> Genome {
        self.representative.clone()
    }

    pub fn remove_organisms(&mut self) {
        self.adjust_fitness();
        self.organisms = vec![];
    }

    pub fn adjust_fitness(&mut self) {
        //TODO: adjust fitness
    }

    fn create_child(&self, organism: &Organism, population_organisms: &Vec<Organism>) -> Organism {
        if rand::random::<f64>() < MUTATION_PROBABILITY || population_organisms.len() < 2 {
            self.create_child_by_mutation(organism)
        } else {
            self.create_child_by_mate(organism, population_organisms)
        }
    }

    fn create_child_by_mutation(&self, organism: &Organism) -> Organism {
        organism.mutate()
    }

    fn create_child_by_mate(&self, organism: &Organism, population_organisms: &Vec<Organism>) -> Organism {
        let mut rng = rand::thread_rng();
        if rand::random::<f64>() > INTERSPECIE_MATE_PROBABILITY {
            let selected_mate = rand::sample(&mut rng, 0..self.organisms.len(), 1)[0];
            organism.mate(&self.organisms[selected_mate])
        }else{
            let selected_mate = rand::sample(&mut rng, 0..population_organisms.len(), 1)[0];
            organism.mate(&population_organisms[selected_mate])
        }
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

        assert!(specie.calculate_average_fitness() == 15f64);
    }
}

