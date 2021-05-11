use organism::Organism;

/// A trait that is implemented by user to allow test of the Environment.
pub trait Environment: Sync {
    /// This test will return the value required by this enviroment to test
    /// against
    fn test(&self, organism: &mut Organism) -> f64;

    /// Returns the number of threads to use on evaluation.
    /// Implement this method to use single thread environment returning 1.
    fn threads(&self) -> usize {
        return num_cpus::get();
    }
}
