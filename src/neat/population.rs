use neat::*;

pub struct Population{
    genomes: Vec<Genome>
}

impl Population {
    pub fn create_population(input_nodes: usize, output_nodes: usize, population_size: usize) -> Population {
        let mut population = Population { genomes: vec![] };
        population.create_genomes(input_nodes, output_nodes, population_size);
        population
    }

    pub fn size(&self) -> usize{
        self.genomes.len()
    }

    pub fn total_weights(&self) -> f64{
        let mut total = 0f64;
        for genome in &self.genomes {
            total += genome.total_weights();
        }
        total
    }

    pub fn total_genes(&self) -> usize{
        let mut total = 0usize;
        for genome in &self.genomes {
            total += genome.total_genes();
        }
        total
    }

    pub fn evolve(&mut self){
        self.genomes = self.generate_offspring();
    }

    fn generate_offspring(&self) -> Vec<Genome>{
        self.speciate();
        unimplemented!();
    }

    fn speciate(&self) -> Vec<Specie>{
        let mut species: Vec<Specie> = vec![];
        for genome in &self.genomes{
            let mut species_search = species.clone(); 
            match species_search.iter_mut().find(|specie| specie.is_owner(&genome)) {
                Some(specie) => {
                    specie.add(genome.clone());
                },
                None => {
                    let mut specie = Specie::new(genome.clone());
                    specie.add(genome.clone());
                    species.push(specie);
                }
            };
        }

        species
    }

    fn create_genomes(&mut self, input_nodes: usize, output_nodes: usize, population_size: usize){
        let mut genomes = vec![];

        while genomes.len() < population_size {
            genomes.push(Genome::new(input_nodes, output_nodes));
        }

        self.genomes = genomes;
    }

}
