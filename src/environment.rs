use organism::Organism;

/// A trait that is implemented by user to allow test of the Environment.
pub trait Environment: Sync {
    /// This test will return the value required by this enviroment to test
    /// against
    fn test(&self, organism: &mut Organism) -> f64;
}
