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

#[cfg(test)]
mod tests {
    use neat::*;

    #[test]
    fn population_should_be_able_to_speciate_genomes(){
        let mut genome1 = Genome::new(10, 10);
        genome1.create_gene(1, 1, 1f64);
        genome1.create_gene(1, 2, 1f64);
        let mut genome2 = Genome::new(10, 10);
        genome2.create_gene(1, 3, 1f64);
        genome2.create_gene(1, 4, 1f64);

        let mut population = Population::create_population(10, 10, 0);
        population.genomes = vec![genome1, genome2];
        let species = population.speciate();
        assert!(species.len() == 2usize);
    }
}
