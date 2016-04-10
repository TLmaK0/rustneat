extern crate rand;

use std::cmp::Ordering;

#[derive(Debug, Clone)]
pub struct Gene{
    pub in_neuron_id: usize,
    pub out_neuron_id: usize,
    pub weight: f64,
    pub enabled: bool,
}

impl Eq for Gene {
}

impl PartialEq for Gene{
    fn eq(&self, other: &Gene) -> bool {
        self.in_neuron_id == other.in_neuron_id && self.out_neuron_id == other.out_neuron_id
    }
}

impl Ord for Gene{
    fn cmp(&self, other: &Gene) -> Ordering {
        if self == other {
            Ordering::Equal
        }else if self.in_neuron_id == other.in_neuron_id {
            if self.out_neuron_id > other.out_neuron_id {
                Ordering::Greater
            }else {
                Ordering::Less
            }
        }else if self.in_neuron_id > other.in_neuron_id {
            Ordering::Greater
        } else {
            Ordering::Less
        }
    }
}

impl PartialOrd for Gene{
    fn partial_cmp(&self, other: &Gene) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Gene {
    pub fn generate_weight () -> f64 {
        rand::random::<f64>()
    }
}

impl Default for Gene{
    fn default () -> Gene {
        Gene { in_neuron_id: 1, out_neuron_id: 1, weight: Gene::generate_weight(), enabled: true }
    }
}
