use neat::*;

#[derive(Debug, Clone)]
pub struct Organism {
    pub genome: Genome,
    pub fitness: f64,
    pub neurons: Vec<Option<Neuron>>
}

impl Organism {
    pub fn new(genome: Genome) -> Organism {
        Organism { genome: genome, fitness: 0f64, neurons: vec![] }
    }

    pub fn mutate(&mut self) {
        self.genome.mutate();
    }

    pub fn mate(&self, other: &Organism) -> Organism {
       Organism::new(self.genome.mate(&other.genome))
    }

    pub fn activate(&mut self, sensors: Vec<f64>){
       if self.neurons.len() == 0 {
           self.generate_phenome();
       };
    }

    fn generate_phenome(&mut self){
        let neurons_to_generate = self.genome.len();

        self.neurons = vec![None; neurons_to_generate as usize];

        for gene in self.genome.get_genes() {
            if self.neurons[gene.in_neuron_id].is_none(){
                self.neurons[gene.in_neuron_id] = Some(Neuron::new())
            }

            let mut neuron = self.neurons[gene.in_neuron_id].as_mut().unwrap();
            neuron.connections.push(Connection(gene.out_neuron_id, gene.weight));
        }
    }
}

#[cfg(test)]
mod tests {
    use neat::*;

    #[test]
    fn should_be_able_to_generate_his_phenome(){
        let mut organism = Organism::new(Genome::new());
        organism.genome.inject_gene(0, 0, 0.1f64);
        organism.genome.inject_gene(0, 1, 0.2f64);
        organism.genome.inject_gene(1, 1, 0.3f64);
        organism.genome.inject_gene(1, 0, 0.4f64);
        organism.generate_phenome();
        assert!(organism.neurons.len() == 2);
        let neuron0 = &organism.neurons[0].as_ref().unwrap();
        let connection00 = &neuron0.connections[0];
        assert!(connection00.0 == 0usize);
        assert!(connection00.1 == 0.1f64);
        let connection01 = &neuron0.connections[1];
        assert!(connection01.0 == 1usize);
        assert!(connection01.1 == 0.2f64);
        let neuron1 = &organism.neurons[1].as_ref().unwrap();
        let connection10 = &neuron1.connections[0];
        assert!(connection10.0 == 1usize);
        assert!(connection10.1 == 0.3f64);
        let connection11 = &neuron1.connections[1];
        assert!(connection11.0 == 0usize);
        assert!(connection11.1 == 0.4f64);
    }
}
