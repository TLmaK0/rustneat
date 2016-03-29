struct Specie{
    representative: Genome,
}

impl Specie{
    fn new(genome: &Genome) -> Specie{
        Specie{ representative: genome.clone() }
    }
}
