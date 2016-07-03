extern crate conv;

use self::conv::prelude::*;
use neat::*;


#[derive(Debug)]
pub struct Population{
    pub species: Vec<Specie>,
    champion_fitness: f64,
    epochs_without_improvements: usize
}

const MAX_EPOCHS_WITHOUT_IMPROVEMENTS: usize = 5;

impl Population {
    pub fn create_population(population_size: usize) -> Population{
        let mut population = Population { 
            species: vec![], 
            champion_fitness: 0f64, 
            epochs_without_improvements: 0usize
        };

        population.create_organisms(population_size);
        population
    }

    pub fn size(&self) -> usize{
        self.species.iter().fold(0usize, |total, specie| total + specie.organisms.len())
    }

    pub fn evolve(&mut self){
        self.generate_offspring();
    }

    pub fn evaluate_in(&mut self, environment: &Environment){
        let mut improvement = false;
        
        for specie in &mut self.species {
            for organism in &mut specie.organisms {
                organism.fitness = environment.test(organism);
                if organism.fitness > self.champion_fitness {
                    self.champion_fitness = organism.fitness;
                    self.epochs_without_improvements = 0usize;
                    improvement = true;
                }
            }
        }

        if !improvement {
            self.epochs_without_improvements += 1;
        }
    }

    pub fn get_organisms(&self)-> Vec<Organism>{
        self.species.iter().flat_map(|specie| {
println!("Specie organisms: {:?}", specie.organisms.len());
                                     specie.organisms.clone()
        }).collect::<Vec<Organism>>()
    }

    pub fn epochs_without_improvements(&self) -> usize {
        self.epochs_without_improvements
    }

    fn generate_offspring(&mut self){
        self.speciate();

        let total_average_fitness = self.species.iter_mut()
            .fold(0f64, |total, specie| total + specie.calculate_average_fitness());

        let mut champion_fitness = 0f64;
        let mut second_best_fitness = 0f64;
            

        if self.epochs_without_improvements > MAX_EPOCHS_WITHOUT_IMPROVEMENTS {
            let fitness_sorted = self.get_species_fitness();
            champion_fitness = fitness_sorted[0];
            if fitness_sorted.len() > 1 {
                second_best_fitness = fitness_sorted[1];
            }
println!("Set champion fitness {:?}", champion_fitness);                    
println!("Set second fitness {:?}", second_best_fitness);                    
        }
println!("Champ fitness: {:?}, Second fitnes: {:?}", champion_fitness, second_best_fitness);
        let num_of_organisms = self.size();

        let organisms_by_average_fitness = num_of_organisms.value_as::<f64>().unwrap() / total_average_fitness;

        let organisms = self.get_organisms();
println!("Num of organisms {:?}", organisms.len());        

        let num_species = self.species.len();

        for specie in &mut self.species {
            let specie_fitness = specie.calculate_average_fitness();
            let mut offspring_size = (specie_fitness * organisms_by_average_fitness).round() as usize;

            if total_average_fitness == 0f64 {
                offspring_size = specie.organisms.len();
            }

            if self.epochs_without_improvements > MAX_EPOCHS_WITHOUT_IMPROVEMENTS && num_species > 1 {
                if specie.calculate_champion_fitness() >= second_best_fitness {
println!("Champion fitness: {:?}", specie.calculate_champion_fitness());                    
                    offspring_size = num_of_organisms.checked_div(2).unwrap();
                } else {
                    offspring_size = 0;
                }
            }
println!("Offspring {:?}, specie_fitness {:?}", offspring_size, specie_fitness);
            if offspring_size > 0 {
                //TODO: check if offspring is for organisms fitness also, not only by specie
                specie.generate_offspring(offspring_size, &organisms);
            } else {
                specie.remove_organisms();
            }
        }
        if self.epochs_without_improvements > MAX_EPOCHS_WITHOUT_IMPROVEMENTS {
            println!("--------------");
            self.epochs_without_improvements = 0;
        }
println!("Total generated {:?}", self.get_organisms().len());        
    }

    fn get_species_fitness(&self) -> Vec<f64>{
        let mut fitness: Vec<f64> = self.species.iter().map(|specie| {
            specie.calculate_champion_fitness()
        }).collect::<Vec<f64>>();
        fitness.sort_by(|a, b| b.partial_cmp(a).unwrap());
        fitness
    }

    fn speciate(&mut self){
        let organisms = &self.get_organisms();
        for specie in &mut self.species {
            specie.remove_organisms();
        }

        for organism in organisms{
            let mut new_specie: Option<Specie> = None;
            match self.species.iter_mut().find(|specie| specie.match_genome(&organism)) {
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
                self.species.push(new_specie.unwrap());
            }
        }
    }

    fn create_organisms(&mut self, population_size: usize){
        self.species = vec![];
        let mut organisms = vec![];

        while organisms.len() < population_size {
            organisms.push(Organism::new(Genome::new()));
        }

        let mut specie = Specie::new(organisms.first().unwrap().genome.clone());
        specie.organisms = organisms;
        self.species.push(specie);
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

        let mut population = Population::create_population(2);
        let organisms = vec![Organism::new(genome1), Organism::new(genome2)];
        let mut specie = Specie::new(organisms.first().unwrap().genome.clone());
        specie.organisms = organisms;
        population.species = vec![specie];
        population.speciate();
        assert_eq!(population.species.len(), 2usize);
    }

    #[test]
    fn after_population_evolve_population_should_be_the_same(){
        let mut population = Population::create_population(150);
        for _ in 0..150 {
            population.evolve();
        }
        assert!(population.size() == 150);
    }
}
