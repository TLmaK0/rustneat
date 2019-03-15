use rand;
use std::cmp::Ordering;

/// Gene for a neuron in the `NeuralNetwork`.
#[derive(Default, Debug, Copy, Clone)]
pub struct NeuronGene {
    /// Bias of the neuron.
    pub bias: f64,
}
impl NeuronGene {
    ///
    pub fn new(bias: f64) -> NeuronGene {
        NeuronGene {
            bias,
        }
    }
}

/// Gene for a synapse/connection in the `NeuralNetwork`.
#[derive(Debug, Copy, Clone)]
#[cfg_attr(feature = "telemetry", derive(Serialize))]
pub struct ConnectionGene {
    in_neuron_id: usize,
    out_neuron_id: usize,
    /// Weight of the connection
    pub weight: f64,
    /// Whether the expression of a gene is enabled.
    pub enabled: bool,
}

impl Eq for ConnectionGene {}

impl PartialEq for ConnectionGene {
    fn eq(&self, other: &ConnectionGene) -> bool {
        self.in_neuron_id == other.in_neuron_id && self.out_neuron_id == other.out_neuron_id
    }
}

impl Ord for ConnectionGene {
    fn cmp(&self, other: &ConnectionGene) -> Ordering {
        if self == other {
            Ordering::Equal
        } else if self.in_neuron_id == other.in_neuron_id {
            if self.out_neuron_id > other.out_neuron_id {
                Ordering::Greater
            } else {
                Ordering::Less
            }
        } else if self.in_neuron_id > other.in_neuron_id {
            Ordering::Greater
        } else {
            Ordering::Less
        }
    }
}

impl PartialOrd for ConnectionGene {
    fn partial_cmp(&self, other: &ConnectionGene) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl ConnectionGene {
    /// Create a new connection
    pub fn new(in_neuron_id: usize, out_neuron_id: usize, weight: f64, enabled: bool) -> ConnectionGene {
        ConnectionGene {
            in_neuron_id: in_neuron_id,
            out_neuron_id: out_neuron_id,
            weight: weight,
            enabled: enabled,
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

impl Default for ConnectionGene {
    fn default() -> ConnectionGene {
        ConnectionGene {
            in_neuron_id: 1,
            out_neuron_id: 1,
            weight: 0.0,
            enabled: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn g(n_in: usize, n_out: usize) -> ConnectionGene {
        ConnectionGene {
            in_neuron_id: n_in,
            out_neuron_id: n_out,
            ..ConnectionGene::default()
        }
    }

    #[test]
    fn should_be_able_to_binary_search_for_a_gene() {
        let mut genome = vec![g(0, 1), g(0, 2), g(3, 2), g(2, 3), g(1, 5)];
        genome.sort();
        genome.binary_search(&g(0, 1)).unwrap();
        genome.binary_search(&g(0, 2)).unwrap();
        genome.binary_search(&g(1, 5)).unwrap();
        genome.binary_search(&g(2, 3)).unwrap();
        genome.binary_search(&g(3, 2)).unwrap();
    }
}
