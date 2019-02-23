use crate::Genome;

/// A trait that is implemented by user to allow test of the Environment.
pub trait Environment<G: Genome>: Sync {
    /// This test will return the value required by this enviroment to test
    /// against
    fn test(&self, organism: &mut G) -> f64;
}
