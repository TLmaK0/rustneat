use neat::genome::Genome;

#[derive(Debug, Clone)]
pub struct Organism {
    pub genome: Genome,
    pub fitness: f64
}

impl Organism {
    pub fn new(genome: Genome) -> Organism {
        Organism { genome: genome, fitness: 0f64 }
    }

    pub fn mutate(&mut self) {
        self.genome.mutate();
    }

    pub fn mate(&self, other: &Organism) -> Organism {
       Organism::new(self.genome.mate(&other.genome))
    }
}
