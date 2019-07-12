use serde_derive::{Deserialize, Serialize};
use std::hash::Hash;

// TODO make private - only needed internally I think
///
pub trait Gene: Copy {
    /// Innovation id, used to index a hashmap structure
    type Id: Hash + Eq + Clone + Copy;
    ///
    fn id(&self) -> Self::Id;
    /// Some way to get the (compatibility) distance between two genes, used in
    /// calculating the distance between genomes
    fn distance(&self, other: &Self) -> f64;
    // TODO maybe add `fn mutate(&mut self, p: &Params)` here :o
}

/// Innovation id of a neuron gene
pub type NeuronId = usize;

/// Gene for a neuron in the `NeuralNetwork`.
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct NeuronGene {
    /// Bias of the neuron.
    pub bias: f64,
    /// Innovation number of the neuron.
    pub innovation_id: usize,
}

impl NeuronGene {
    ///
    pub fn new(bias: f64, innovation_id: usize) -> NeuronGene {
        NeuronGene {
            bias,
            innovation_id,
        }
    }
}
impl Gene for NeuronGene {
    type Id = NeuronId;
    fn id(&self) -> NeuronId {
        self.innovation_id
    }
    fn distance(&self, other: &Self) -> f64 {
        (self.bias - other.bias).abs()
    }
}

/// Innovation id of a connection gene = (input neuron id, output neuron id)
pub type ConnectionId = (usize, usize);

/// Gene for a synapse/connection in the `NeuralNetwork`.
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct ConnectionGene {
    in_neuron_id: usize,
    out_neuron_id: usize,
    /// Weight of the connection
    pub weight: f64,
}

impl ConnectionGene {
    /// Create a new connection
    pub fn new(in_neuron_id: usize, out_neuron_id: usize, weight: f64) -> ConnectionGene {
        ConnectionGene {
            in_neuron_id: in_neuron_id,
            out_neuron_id: out_neuron_id,
            weight: weight,
        }
    }
    /// The neuron that acts as the input of this connection
    pub fn in_neuron_id(&self) -> usize {
        self.in_neuron_id
    }
    /// The neuron that acts as the output of this connection
    pub fn out_neuron_id(&self) -> usize {
        self.out_neuron_id
    }
}
impl Gene for ConnectionGene {
    type Id = ConnectionId;
    fn id(&self) -> ConnectionId {
        (self.in_neuron_id, self.out_neuron_id)
    }
    fn distance(&self, other: &Self) -> f64 {
        (self.weight - other.weight).abs()
    }
}

impl Default for ConnectionGene {
    // TODO remove?
    fn default() -> ConnectionGene {
        ConnectionGene {
            in_neuron_id: 1,
            out_neuron_id: 1,
            weight: 0.0,
        }
    }
}
