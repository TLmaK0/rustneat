use crate::organism::Organism;

/// A trait that is implemented by user to allow test of the Environment.
///
/// # Usage
///
/// There are two ways to implement this trait:
///
/// ## Option 1: Individual evaluation (simple)
/// Override `test()` to evaluate one organism at a time:
/// ```ignore
/// impl Environment for MyEnv {
///     fn test(&self, organism: &mut Organism) -> f64 {
///         // evaluate and return fitness
///     }
/// }
/// ```
///
/// ## Option 2: Batch evaluation (efficient for vectorized environments)
/// Override `test_batch()` to evaluate all organisms together, useful for
/// GPU acceleration or vectorized simulators like gymnasium's VectorEnv:
/// ```ignore
/// impl Environment for MyVectorEnv {
///     fn test_batch(&self, organisms: &mut [Organism]) {
///         // Evaluate all organisms in parallel using VectorEnv
///         // Set organism.fitness for each
///     }
/// }
/// ```
pub trait Environment: Sync {
    /// Evaluate a single organism and return its fitness.
    ///
    /// Override this method for simple sequential evaluation.
    /// If you only use `test_batch()`, you don't need to override this.
    fn test(&self, _organism: &mut Organism) -> f64 {
        unimplemented!(
            "Override test() for individual evaluation or test_batch() for batch evaluation"
        )
    }

    /// Batch evaluation of multiple organisms.
    ///
    /// Default implementation calls `test()` sequentially.
    /// Override this method for efficient batch evaluation, e.g., using
    /// gymnasium's VectorEnv to run multiple environments in parallel.
    ///
    /// When overriding, you must set `organism.fitness` for each organism.
    fn test_batch(&self, organisms: &mut [Organism]) {
        for organism in organisms.iter_mut() {
            organism.fitness = self.test(organism);
        }
    }

    /// Returns the number of threads to use for evaluation.
    ///
    /// When threads > 1, organisms are divided into chunks and each chunk
    /// is passed to `test_batch()` in a separate thread.
    /// Return 1 for single-threaded evaluation (all organisms in one batch).
    fn threads(&self) -> usize {
        num_cpus::get()
    }
}
