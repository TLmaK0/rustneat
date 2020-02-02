const COMPATIBILITY_THRESHOLD: f64 = 3.0; // used to speciate organisms

/// Implementing `Genome` conceptually means that the implementor "has a
/// genome", and the implementor can be called an "organism".
// (TODO: remove Default?)
pub trait Genome: Clone + Default + Send {
    /// Returns a new organism which is a clone of `&self` apart from possible
    /// mutations
    fn mutate(&self) -> Self;

    /// `fittest` is true if `other` is more fit.
    fn mate(&self, other: &Self, fittest: bool) -> Self;

    /// TODO: how should it be implemented for e.g. a composed organism?
    fn distance(&self, other: &Self) -> f64;

    /// Compare another Genome for species equality
    // TODO This should be impl Eq
    fn is_same_specie(&self, other: &Self) -> bool {
        self.distance(other) < COMPATIBILITY_THRESHOLD
    }
}

/// Used in algorithm just to group an organism (genome) with its fitness, and
/// also in the interface to get the fitness of organisms
#[derive(Default, Clone, Debug)]
pub struct Organism<G> {
    /// The genome of this organism
    pub genome: G,
    /// The fitness calculated internally
    // TODO: Make fitness private with a getter?
    //       or Option<f64>
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
    pub fn mutate(&self) -> Organism<G> {
        Organism::new(self.genome.mutate())
    }
    /// Mate with another organism -- this mates the two genomes.
    pub fn mate(&self, other: &Self) -> Organism<G> {
        Organism::new(
            self.genome
                .mate(&other.genome, self.fitness < other.fitness),
        )
    }
    ///
    pub fn distance(&self, other: &Self) -> f64 {
        self.genome.distance(&other.genome)
    }
}
