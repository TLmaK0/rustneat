use neat::*;

#[derive(Debug, Clone)]
pub struct Neuron{
    input: f64,
    pub connections: Vec<Connection>
}

impl Neuron{
    pub fn new() -> Neuron {
        Neuron { input: 0f64, connections: vec![] }
    }
}
