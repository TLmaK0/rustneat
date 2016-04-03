use neat::genome::Genome;


#[derive(Debug, Clone)]
pub struct Specie{
    representative: Genome,
    pub genomes: Vec<Genome>
}

impl Specie{
    pub fn new(genome: Genome) -> Specie{
        Specie{ genomes: vec![], representative: genome }
    }

    pub fn add(&mut self, genome: Genome){
        self.genomes.push(genome);
    }

    pub fn is_owner(&self, genome: &Genome) -> bool{
        self.representative.is_same_specie(genome)
    }
}
