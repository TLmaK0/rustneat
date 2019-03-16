use crate::{NeuralNetwork, Params};

/// Implementing `Genome` conceptually means that the implementor "has a genome", and the
/// implementor can be called an "organism".
pub trait Genome: Clone + Default + Send {
    /// Returns a new organism which is a clone of `&self` apart from possible mutations
    fn mutate(&mut self, innovation_id: &mut usize, p: &Params);

    /// `fittest` is true if `other` is more fit.
    fn mate(&self, other: &Self, fittest: bool, p: &Params) -> Self;

    /// TODO: how should it be implemented for e.g. a composed organism?
    fn distance(&self, other: &Self, p: &Params) -> f64;


    /// Compare another Genome for species equality
    // TODO This should be impl Eq
    fn is_same_specie(&self, other: &Self, p: &Params) -> bool {
        self.distance(other, p) < p.compatibility_threshold
    }
}

/// Used in algorithm just to group an organism (genome) with its fitness, and also in the
/// interface to get the fitness of organisms
#[derive(Default, Clone, Debug)]
pub struct Organism<G = NeuralNetwork> {
    /// The genome of this organism
    pub genome: G,
    /// The fitness calculated as part of the NEAT algorithm
    pub fitness: f64,
}
impl<G: Genome> Organism<G> {
    /// Create a new organism with fitness 0.0.
    pub fn new(organism: G) -> Organism<G> {
        Organism {
            genome: organism,
            fitness: 0.0,
        }
    }
    /// Returns a cloned `Organism` with a mutated genome
    pub fn mutate(&mut self, innovation_id: &mut usize, p: &Params) {
        self.genome.mutate(innovation_id, p)
    }
    /// Mate with another organism -- this mates the two genomes.
    pub fn mate(&self, other: &Self, p: &Params) -> Organism<G> {
        Organism::new(
            self.genome
                .mate(&other.genome, self.fitness > other.fitness, p))
    }
    /// 
    pub fn distance(&self, other: &Self, p: &Params) -> f64 {
        self.genome.distance(&other.genome,p )
    }
}
