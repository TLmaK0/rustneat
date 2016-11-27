use organism::Organism;

pub trait Environment: Sync {
    fn test(&self, organism: &mut Organism) -> f64;
}
