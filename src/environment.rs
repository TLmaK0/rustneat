use crate::organism::Organism;

/// A trait that is implemented by user to allow test of the Environment.
pub trait Environment: Sync {
    /// This test will return the value required by this enviroment to test
    /// against
    fn test(&self, organism: &mut Organism) -> f64;

    /// Batch evaluation of multiple organisms. Default implementation calls test() sequentially.
    /// Override this method to implement efficient batch evaluation (e.g., using pool.starmap()).
    fn test_batch(&self, organisms: &mut [Organism]) {
        for organism in organisms.iter_mut() {
            organism.fitness = self.test(organism);
        }
    }

    /// Returns the number of threads to use on evaluation.
    /// Implement this method to use single thread environment returning 1.
    fn threads(&self) -> usize {
        return num_cpus::get();
    }
}
