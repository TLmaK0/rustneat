use crate::{Genome, NeuralNetwork};

/// A trait that is implemented by user to test the fitness of organisms.
pub trait Environment<G: Genome = NeuralNetwork>: Sync {
    /// This test will return the value required by this enviroment to test
    /// against
    fn test(&self, organism: &mut G) -> f64;
}
