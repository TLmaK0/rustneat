extern crate rand;

#[derive(Debug, Clone, Copy)]
pub struct ConnectionGene{
    pub in_node_id: u32,
    pub out_node_id: u32,
    pub weight: f64,
    pub enabled: bool,
    pub innovation: u32,
}

impl ConnectionGene {
    pub fn generate_weight () -> f64 {
        rand::random::<f64>()
    }
}

impl Default for ConnectionGene{
    fn default () -> ConnectionGene {
        ConnectionGene { in_node_id: 1, out_node_id: 1, weight: ConnectionGene::generate_weight(), enabled: true, innovation: 0 }
    }
}
