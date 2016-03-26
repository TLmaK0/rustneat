extern crate rand;

use std::cmp::Ordering;

#[derive(Debug, Clone, Copy)]
pub struct ConnectionGene{
    pub in_node_id: u32,
    pub out_node_id: u32,
    pub weight: f64,
    pub enabled: bool,
}

impl Eq for ConnectionGene {
}

impl PartialEq for ConnectionGene{
    fn eq(&self, other: &ConnectionGene) -> bool {
        self.in_node_id == other.in_node_id && self.out_node_id == other.out_node_id
    }
}

impl Ord for ConnectionGene{
    fn cmp(&self, other: &ConnectionGene) -> Ordering {
        if self == other {
            Ordering::Equal
        }else if self.in_node_id == other.in_node_id {
            if self.out_node_id > other.out_node_id {
                Ordering::Greater
            }else {
                Ordering::Less
            }
        }else if self.in_node_id > other.in_node_id {
            Ordering::Greater
        } else {
            Ordering::Less
        }
    }
}

impl PartialOrd for ConnectionGene{
    fn partial_cmp(&self, other: &ConnectionGene) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl ConnectionGene {
    pub fn generate_weight () -> f64 {
        rand::random::<f64>()
    }
}

impl Default for ConnectionGene{
    fn default () -> ConnectionGene {
        ConnectionGene { in_node_id: 1, out_node_id: 1, weight: ConnectionGene::generate_weight(), enabled: true }
    }
}
