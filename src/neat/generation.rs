use neat::*;

#[derive(Debug)]
pub struct Generation<'a> {
    genomes: Vec<&'a Genome>,
}

impl<'a> Generation<'a> {
    pub fn new() -> Generation<'a> {
        Generation { genomes: vec![] }
    }
}
