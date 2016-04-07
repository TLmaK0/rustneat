use neat::*;
pub trait Environment{
    fn test(&self, organism: &Organism) -> f64;
}
