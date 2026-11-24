use crate::genome::Genome;
use std::cmp;
use std::collections::HashMap;

/// Historical markings of a population (NEAT paper, section 3.2): the id of
/// the neuron created by splitting each connection.
///
/// Splitting the connection `in -> out` always gives the same neuron id, in
/// every genome and every generation, so the same structural innovation in
/// two lineages gets the same id and the genes `in -> id` and `id -> out`
/// line up in crossover and in the compatibility distance. Different splits
/// get different ids.
///
/// With neuron ids shared by the whole population, a connection gene is
/// identified by its `(in, out)` pair: it plays the role of the innovation
/// number of the paper (as in neat-python, whose connection key is the
/// pair), so genes need no number of their own.
///
/// A genome can split the same connection more than once (the connection
/// is disabled by the split, but it can be split again or come back through
/// crossover). Each split of `in -> out` has its own id: the first one the
/// genome does not have yet, so two genomes that split it twice still share
/// both ids.
#[derive(Debug, Clone, Default)]
pub struct InnovationRegistry {
    splits: HashMap<(usize, usize), Vec<usize>>,
    next_neuron_id: usize,
}

impl InnovationRegistry {
    /// An empty registry whose new neurons get ids from `first_neuron_id`
    /// on: the number of input plus output neurons, so they never take the
    /// id of one of them.
    pub fn new(first_neuron_id: usize) -> InnovationRegistry {
        InnovationRegistry {
            splits: HashMap::new(),
            next_neuron_id: first_neuron_id,
        }
    }

    /// Id of the neuron that `genome` gets by splitting the connection
    /// `in_neuron_id -> out_neuron_id`: the id this split already has in the
    /// population, or a new one. The id is never a neuron `genome` already has.
    pub fn split_neuron_id(
        &mut self,
        genome: &Genome,
        in_neuron_id: usize,
        out_neuron_id: usize,
    ) -> usize {
        let ids = self
            .splits
            .entry((in_neuron_id, out_neuron_id))
            .or_default();
        if let Some(&id) = ids.iter().find(|&&id| !genome.has_neuron(id)) {
            return id;
        }
        // Never an id the genome already uses, even one it got without
        // this registry.
        let id = cmp::max(self.next_neuron_id, genome.len());
        self.next_neuron_id = id + 1;
        ids.push(id);
        id
    }

    /// The connection `(in, out)` whose split created the neuron
    /// `neuron_id`, if this registry gave that id.
    pub fn split_of(&self, neuron_id: usize) -> Option<(usize, usize)> {
        self.splits
            .iter()
            .find(|(_, ids)| ids.contains(&neuron_id))
            .map(|(&split, _)| split)
    }

    /// Number of neuron ids given so far.
    pub fn len(&self) -> usize {
        self.splits.values().map(Vec::len).sum()
    }

    /// True if no neuron id has been given yet.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
