use neat::*;

#[derive(Debug)]
pub struct Generation {
    genomes: Vec<Genome>,
}

impl Generation {
    pub fn new() -> Generation {
        Generation { genomes: vec![]}
    }

    pub fn create_genome(&mut self) -> Genome {
        Genome::new()
    }
}
