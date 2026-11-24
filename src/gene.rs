use rand::Rng;
use std::cmp::Ordering;

/// A connection Gene
#[derive(Debug, Copy, Clone)]
#[cfg_attr(feature = "telemetry", derive(serde::Serialize))]
pub struct Gene {
    in_neuron_id: usize,
    out_neuron_id: usize,
    weight: f64,
    enabled: bool,
    is_bias: bool,
}

impl Eq for Gene {}

impl PartialEq for Gene {
    fn eq(&self, other: &Gene) -> bool {
        self.in_neuron_id == other.in_neuron_id && self.out_neuron_id == other.out_neuron_id
    }
}

impl Ord for Gene {
    fn cmp(&self, other: &Gene) -> Ordering {
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

impl PartialOrd for Gene {
    fn partial_cmp(&self, other: &Gene) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Gene {
    /// Create a new gene
    pub fn new(
        in_neuron_id: usize,
        out_neuron_id: usize,
        weight: f64,
        enabled: bool,
        is_bias: bool,
    ) -> Gene {
        Gene {
            in_neuron_id: in_neuron_id,
            out_neuron_id: out_neuron_id,
            weight: weight,
            enabled: enabled,
            is_bias: is_bias,
        }
    }

    /// Create a new gene with a specific connection
    pub fn new_connection(in_neuron_id: usize, out_neuron_id: usize) -> Gene {
        Gene {
            in_neuron_id: in_neuron_id,
            out_neuron_id: out_neuron_id,
            ..Gene::default()
        }
    }

    /// Generate a weight in [-1, 1] using an entropy-seeded generator
    pub fn generate_weight() -> f64 {
        Gene::generate_weight_with_rng(&mut rand::thread_rng())
    }

    /// Generate a weight in [-1, 1] using the given generator
    pub fn generate_weight_with_rng<R: Rng + ?Sized>(rng: &mut R) -> f64 {
        Gene::generate_weight_in_range_with_rng(1.0, rng)
    }

    /// Generate a weight in [-range, range] using an entropy-seeded generator
    pub fn generate_weight_in_range(range: f64) -> f64 {
        Gene::generate_weight_in_range_with_rng(range, &mut rand::thread_rng())
    }

    /// Generate a weight in [-range, range] using the given generator
    pub fn generate_weight_in_range_with_rng<R: Rng + ?Sized>(range: f64, rng: &mut R) -> f64 {
        rng.gen::<f64>() * 2.0 * range - range
    }
    /// Connection in ->
    pub fn in_neuron_id(&self) -> usize {
        self.in_neuron_id
    }
    /// connection out <->
    pub fn out_neuron_id(&self) -> usize {
        self.out_neuron_id
    }
    /// getter for the wight of the gene
    pub fn weight(&self) -> f64 {
        self.weight
    }
    /// Setter
    pub fn set_weight(&mut self, weight: f64) {
        self.weight = weight;
    }
    /// Is gene enabled
    pub fn enabled(&self) -> bool {
        self.enabled
    }
    /// Set gene enabled
    pub fn set_enabled(&mut self) {
        self.enabled = true;
    }
    /// Set gene disabled
    pub fn set_disabled(&mut self) {
        self.enabled = false;
    }
    /// Get if gene is bias
    pub fn is_bias(&self) -> bool {
        self.is_bias
    }
    /// Set or disable bias
    pub fn set_bias(&mut self, is_bias: bool) {
        self.is_bias = is_bias
    }
}

impl Default for Gene {
    fn default() -> Gene {
        Gene {
            in_neuron_id: 1,
            out_neuron_id: 1,
            weight: Gene::generate_weight(),
            enabled: true,
            is_bias: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g(n_in: usize, n_out: usize) -> Gene {
        Gene {
            in_neuron_id: n_in,
            out_neuron_id: n_out,
            ..Gene::default()
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
