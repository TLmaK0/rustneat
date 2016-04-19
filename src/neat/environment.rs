use neat::*;
pub trait Environment{
    fn test(&self, organism: &mut Organism) -> f64;
}
