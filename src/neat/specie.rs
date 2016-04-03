use neat::genome::Genome;
use neat::organism::Organism;


#[derive(Debug, Clone)]
pub struct Specie{
    representative: Genome,
    pub organisms: Vec<Organism>
}

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
}
