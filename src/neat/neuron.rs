use neat::*;

#[derive(Debug, Clone)]
pub struct Neuron{
    input: f64,
    potential: f64,
    pub connections: Vec<Connection>
}

impl Neuron{
    pub fn new() -> Neuron {
        Neuron { input: 0f64, potential: 0f64, connections: vec![]}
    }

    pub fn stimulate(&mut self, stimul: f64){
        self.input += stimul;
    }

    pub fn potential(&self) -> f64 {
        self.potential
    }

    pub fn shot(&mut self) {
       self.potential = 1f64 / ( 1f64 +  self.input.exp() );
       self.input = 0f64;
    }
}
