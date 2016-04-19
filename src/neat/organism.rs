use neat::*;

#[derive(Debug, Clone)]
pub struct Organism {
    pub genome: Genome,
    pub fitness: f64,
    pub neurons: Vec<Option<Neuron>>
}

const ACTIVATION_CYCLES: usize = 5;

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

    pub fn activate(&mut self, sensors: &Vec<f64>, outputs: &mut Vec<f64>){
       if self.neurons.len() == 0 {
           self.generate_phenome();
       };

       for neuron_id in 0..sensors.len() {
           if neuron_id < self.neurons.len() {
               self.neurons[neuron_id].as_mut().unwrap().stimulate(sensors[neuron_id]);
           }
       }

       for _n in 0..ACTIVATION_CYCLES {
           self.activate_neurons(sensors.len());
           self.propagate_signals();
       }

       //Take outputs from next neurons after sensors
       for neuron_id in sensors.len()..(sensors.len() + outputs.len()){
           if neuron_id < self.neurons.len() {
               outputs[neuron_id - sensors.len()] = self.neurons[neuron_id].as_ref().map_or(0f64, |neuron| neuron.stimulation());
           } else {
               outputs[neuron_id - sensors.len()] = 0f64;
           }
       }
    }

    fn activate_neurons(&mut self, sensor_neurons_size: usize){
        //Don't reset sensor neurons during activation
        for neuron_id in 0..sensor_neurons_size {
            if neuron_id < self.neurons.len() {
                self.neurons[neuron_id].as_mut().unwrap().shot(true);
            }
        }

        for neuron_id in sensor_neurons_size..self.neurons.len() {
            self.neurons[neuron_id].as_mut().unwrap().shot(false);
        }
    }

    fn propagate_signals(&mut self){
        for neuron_option in self.neurons.clone() {
            let neuron = neuron_option.unwrap();
            for connection in &neuron.connections {
               self.neurons[connection.0].as_mut().unwrap().stimulate(neuron.potential() * connection.1);
            }
        }
    }

    fn generate_phenome(&mut self){
        let neurons_to_generate = self.genome.len();

        self.neurons = vec![Some(Neuron::new()); neurons_to_generate as usize];

        for gene in self.genome.get_genes() {
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

    #[test]
    fn should_propagate_signal(){
        let mut organism = Organism::new(Genome::new());
        organism.genome.inject_gene(0, 1, 1f64);
        let sensors = vec![0f64];
        let mut output = vec![0f64];
        organism.activate(&sensors, &mut output);        
        assert!(output[0] == 0.5f64, format!("{:?} is not 0.5", output[0]));
    }


    fn sigmoid(x: f64) -> f64 {
       1f64 / ( 1f64 +  (-x).exp() )
    }

    #[test]
    fn should_propagate_signal_over_hidden_layers(){
        let mut organism = Organism::new(Genome::new());
        organism.genome.inject_gene(0, 1, 0f64); //disable connection
        organism.genome.inject_gene(0, 2, 1f64);
        organism.genome.inject_gene(2, 1, 1f64);
        let sensors = vec![0f64];
        let mut output = vec![0f64];
        organism.activate(&sensors, &mut output);
        assert!(output[0] == sigmoid(sigmoid(sensors[0])), format!("{:?} is not 0.5", output[0]));
    }

    #[test]
    fn should_work_with_cyclic_networks(){
        let mut organism = Organism::new(Genome::new());
        organism.genome.inject_gene(0, 1, 1f64); 
        organism.genome.inject_gene(1, 2, 1f64);
        organism.genome.inject_gene(2, 1, 1f64);
        let sensors = vec![0f64];
        let mut output = vec![0f64];
        organism.activate(&sensors, &mut output);
        let mut n1 = 0f64;
        let mut n2 = 0f64;
        for _n in 0..5 {
            let temp = n1;
            n1 = sigmoid(0f64) + sigmoid(n2);
            n2 = sigmoid(temp); 
        }
        assert!(output[0] == n1, format!("{:?} is not {:?}", output[0], n1));
    }

    #[test]
    fn activate_organims_sensor_without_enough_neurons_should_ignore_it(){
        let mut organism = Organism::new(Genome::new());
        organism.genome.inject_gene(0, 1, 1f64); 
        let sensors = vec![0f64,0f64,0f64];
        let mut output = vec![0f64];
        organism.activate(&sensors, &mut output);
    }
}
