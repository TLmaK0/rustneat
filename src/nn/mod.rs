use rand::{self, seq::IteratorRandom, distributions::{Distribution, Uniform, Normal}};
use std::cmp;
use crate::Genome;

mod ctrnn;
mod gene;
pub use self::ctrnn::*;
pub use self::gene::*;

/// Vector of Genes
/// Holds a count of last neuron added, similar to Innovation number
// NOTE: With regard to the "competing conventions" problem in the orginal paper:
// Connections are identified by their (in_neuron_id, out_neuron_id) pair. Neurons are identified just
// by their position in the Vec (implicit id).
#[derive(Default, Debug, Clone)]
pub struct NeuralNetwork {
    /// Sorted at all times
    pub connections: Vec<ConnectionGene>,
    pub neurons: Vec<NeuronGene>,
}

const MUTATE_CONNECTION_WEIGHT: f64 = 0.90;
const MUTATE_ADD_CONNECTION: f64 = 0.005;
const MUTATE_ADD_NEURON: f64 = 0.004;
const MUTATE_TOGGLE_EXPRESSION: f64 = 0.001;
const MUTATE_CONNECTION_WEIGHT_PERTURBED_PROBABILITY: f64 = 0.9;
const MUTATE_BIAS: f64 = 0.01;

impl Genome for NeuralNetwork {
    // http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf - Pag. 110
    // Doesn't distinguish between disjoint and excess genes.
    // Only cares about connection genes (topology).
    fn distance(&self, other: &NeuralNetwork) -> f64 {
        // TODO: optimize this method
        let c2 = 1.0;
        let c3 = 0.4;

        // Number of excess
        let n1 = self.connections.len();
        let n2 = other.connections.len();
        let n = cmp::max(n1, n2);

        if n == 0 {
            return 0.0; // no genes in any genome, the genomes are equal
        }

        let z = if n < 20 { 1.0 } else { n as f64 };

        let matching_genes = self.connections
            .iter()
            .filter(|i1_gene| other.connections.contains(i1_gene))
            .collect::<Vec<_>>();
        let n3 = matching_genes.len();

        // Disjoint / excess genes
        let d = n1 + n2 - (2 * n3);

        // average weight differences of matching genes
        let w1 = matching_genes.iter().fold(0.0, |acc, &m_gene| {
            acc + (m_gene.weight
                - &other.connections[other.connections.binary_search(m_gene).unwrap()].weight)
                .abs()
        });

        let w = if n3 == 0 { 0.0 } else { w1 / n3 as f64 };

        // compatibility distance
        (c2 * d as f64 / z) + c3 * w
    }
    /// May add a connection &| neuron &| mutat connection weight &|
    /// enable/disable connection
    fn mutate(&self) -> Self {
        let mut new = self.clone();
        if rand::random::<f64>() < MUTATE_ADD_CONNECTION || new.connections.is_empty() {
            new.mutate_add_connection();
        };

        if rand::random::<f64>() < MUTATE_ADD_NEURON {
            new.mutate_add_neuron();
        };

        if rand::random::<f64>() < MUTATE_CONNECTION_WEIGHT {
            new.mutate_connection_weight();
        };

        if rand::random::<f64>() < MUTATE_TOGGLE_EXPRESSION {
            new.mutate_toggle_expression();
        };

        if rand::random::<f64>() < MUTATE_BIAS {
            new.mutate_bias();
        };
        new
    }

    /// Mate two genes
    fn mate(&self, other: &NeuralNetwork, fittest: bool) -> NeuralNetwork {
        let mut genome = NeuralNetwork::default();
        genome.neurons = 
            if self.neurons.len() > other.neurons.len() {
                self.mate_neurons(other, fittest)
            } else {
                other.mate_neurons(self, !fittest)
            };
        genome.connections = 
            if self.connections.len() > other.connections.len() {
                self.mate_connections(other, fittest)
            } else {
                other.mate_connections(self, !fittest)
            };
        genome
    }

}

impl NeuralNetwork {
    /// Creates a network that with no connections, but enough neurons to cover all inputs and
    /// outputs.
    pub fn with_input_and_output(inputs: usize, outputs: usize) -> NeuralNetwork {
        let mut nn = NeuralNetwork::default();
        for _ in 0..(inputs+outputs) {
            nn.neurons.push(NeuronGene::new(0.0))
        }
        nn
    }

    /// Activate the neural network by sending input `sensors` into its first `sensors.len()`
    /// neurons
    pub fn activate(&mut self, sensors: Vec<f64>, outputs: &mut Vec<f64>) {
        let neurons_len = self.n_neurons();
        let sensors_len = sensors.len();

        let tau = vec![1.0; neurons_len];
        let theta = self.get_bias(); 

        let mut i = sensors.clone();

        if neurons_len < sensors_len {
            i.truncate(neurons_len);
        } else {
            i = [i, vec![0.0; neurons_len - sensors_len]].concat();
        }

        let wij = self.get_weights();

        let activations =
            Ctrnn {
                y: &i,  //initial state is the sensors
                delta_t: 1.0,
                tau: &tau,
                wij: &wij,
                theta: &theta,
                i: &i
            }.activate_nn(10);

        if sensors_len < neurons_len {
            let outputs_activations = activations.split_at(sensors_len).1.to_vec();

            for n in 0..cmp::min(outputs_activations.len(), outputs.len()) {
                outputs[n] = outputs_activations[n];
            }
        }
    }

    /// Helper function for `activate()`. Get weights of connections (as a matrix represented
    /// linearly)
    fn get_weights(&self) -> Vec<f64> {
        let n_neurons = self.neurons.len();
        let mut matrix = vec![0.0; n_neurons * n_neurons];
        for gene in &self.connections {
            if gene.enabled {
                matrix[(gene.out_neuron_id() * n_neurons) + gene.in_neuron_id()] = gene.weight;
            }
        }
        matrix
    }

    /// Helper function for `activate()`. Get bias of neurons.
    fn get_bias(&self) -> Vec<f64> {
        self.neurons.iter().map(|x| x.bias).collect()
    }


    /// Get number of neurons
    pub fn n_neurons(&self) -> usize {
        self.neurons.len()
    }
    /// Get number of connections (this equals the number of genes)
    pub fn n_connections(&self) -> usize {
        self.connections.len()
    }
    /// is genome empty
    pub fn is_empty(&self) -> bool {
        self.n_neurons() == 0
    }

    fn mutate_add_connection(&mut self) {
        let mut rng = rand::thread_rng();
        assert!(self.neurons.len() != 0);
        let neuron_ids_to_connect = {
            match self.neurons.len() {
                0 => panic!("Must have at least one neuron"),
                1 => vec![0, 0],
                _ => (0..self.neurons.len() + 1).choose_multiple(&mut rng, 2),
            }
        };
        self.add_connection(neuron_ids_to_connect[0], neuron_ids_to_connect[1]);
    }

    fn mutate_connection_weight(&mut self) {
        // TODO mutate all or just one?
        for gene in &mut self.connections {
            let perturbation = rand::random::<f64>() < MUTATE_CONNECTION_WEIGHT_PERTURBED_PROBABILITY;

            let mut new_weight = ConnectionGene::generate_weight();
            if perturbation {
                new_weight += gene.weight;
            }
            gene.weight = new_weight;
        }
    }

    /// Toggles the expression of a random connection
    fn mutate_toggle_expression(&mut self) {
        let mut rng = rand::thread_rng();
        let selected_gene = Uniform::from(0..self.connections.len()).sample(&mut rng);
        self.connections[selected_gene].enabled = !self.connections[selected_gene].enabled;
    }

    /// Sets the bias of a random neuron to a sample from the normal distribution
    fn mutate_bias(&mut self) {
        let mut rng = rand::thread_rng();
        let selected_gene = Uniform::from(0..self.neurons.len()).sample(&mut rng);
        self.neurons[selected_gene].bias = Normal::new(0.0, 1.0).sample(&mut rng);
    }

    fn mutate_add_neuron(&mut self) {
        // Select a random connections along which to add neuron.. and disable it 
        let mut rng = rand::thread_rng();
        let gene = Uniform::from(0..self.connections.len()).sample(&mut rng);
        self.connections[gene].enabled = false;
        // Create new neuron
        let new_neuron = NeuronGene::new(0.0);
        self.neurons.push(new_neuron);
        // ... and make two new connections that go through the new neuron
        self.add_connection(
            ConnectionGene::new(self.connections[gene].in_neuron_id(), self.neurons.len(), 1.0, true));
        self.add_connection(
            ConnectionGene::new(self.neurons.len(),
                self.connections[gene].out_neuron_id(),
                self.connections[gene].weight, true));
    }


    fn mate_neurons(&self, other: &NeuralNetwork, fittest: bool) -> Vec<NeuronGene> {
        // Guarantee: self.neurons.len() is greater than other.neurons.len()
        // TODO: change `0.5` according to which organism is the fittest
        let mut genes = Vec::new();
        for i in 0..self.neurons.len() {
            genes.push({
                if rand::random::<f64>() > 0.5 {
                    self.neurons[i]
                } else {
                    if let Some(gene) = other.neurons.get(i) {
                        *gene
                    } else {
                        self.neurons[i]
                    }
                }
            });
        }
        genes
    }
    fn mate_connections(&self, other: &NeuralNetwork, fittest: bool) -> Vec<ConnectionGene> {
        // Guarantee: self.connections.len() is greater than other.connections.len()
        let mut genes = Vec::new();
        for gene in &self.connections {
            genes.push({
                if rand::random::<f64>() > 0.5 {
                    *gene
                } else {
                    match other.connections.binary_search(gene) {
                        Ok(position) => other.connections[position],
                        Err(_) => *gene,
                    }
                }
            });
        }
        genes.sort();
        genes
    }


    /// Add a new connection.
    pub fn add_connection(&mut self, in_neuron: usize, out_neuron: usize, weight: f64) {
        let new_gene = ConnectionGene::new(in_neuron, out_neuron, weight, true);
        let max_neuron_id = self.neurons.len()-1;

        if gene.in_neuron_id() > max_neuron_id
            || gene.out_neuron_id() > max_neuron_id
            || gene.in_neuron_id() == gene.out_neuron_id() {
            panic!("Invalid connection ({} -> {}). (max neuron id = {})",
                   gene.in_neuron_id(), gene.out_neuron_id(), max_neuron_id);
        }

        match self.connections.binary_search(&gene) {
            Ok(pos) => self.connections[pos].enabled = true,
            Err(_) => self.connections.push(gene),
        }
        self.connections.sort();
    }


    /// Total weigths of all genes
    pub fn total_weights(&self) -> f64 {
        let mut total = 0.0;
        for gene in &self.genes {
            total += gene.weight;
        }
        total
    }

    /// Total num genes
    // TODO len() is enough
    pub fn total_genes(&self) -> usize {
        self.genes.len()
    }

}

#[cfg(test)]
mod tests {
    use std::f64::EPSILON;
    use crate::{nn::NeuralNetwork, nn::Gene, Genome, Organism};

    #[test]
    fn mutation_connection_weight() {
        let mut genome = NeuralNetwork::default();
        genome.add_gene(Gene::new(0, 0, 1.0, true, false));
        let orig_gene = genome.genes[0];
        genome.mutate_connection_weight();
        // These should not be same size
        assert!((genome.genes[0].weight - orig_gene.weight).abs() > EPSILON);
    }

    #[test]
    fn mutation_add_connection() {
        let mut genome = NeuralNetwork::default();
        genome.add_connection(1, 2);

        assert!(genome.genes[0].in_neuron_id() == 1);
        assert!(genome.genes[0].out_neuron_id() == 2);
    }

    #[test]
    fn mutation_add_neuron() {
        let mut genome = NeuralNetwork::default();
        genome.mutate_add_connection();
        genome.mutate_add_neuron();
        assert!(!genome.genes[0].enabled);
        assert!(genome.genes[1].in_neuron_id() == genome.genes[0].in_neuron_id());
        assert!(genome.genes[1].out_neuron_id() == 1);
        assert!(genome.genes[2].in_neuron_id() == 1);
        assert!(genome.genes[2].out_neuron_id() == genome.genes[0].out_neuron_id());
    }

    #[test]
    #[should_panic(expected = "Try to create a gene neuron unconnected, max neuron id 1, 2 -> 2")]
    fn try_to_inject_a_unconnected_neuron_gene_should_panic() {
        let mut genome1 = NeuralNetwork::default();
        genome1.add_gene(Gene::new(2, 2, 0.5, true, false));
    }

    #[test]
    fn two_genomes_without_differences_should_be_in_same_specie() {
        let mut genome1 = NeuralNetwork::default();
        genome1.add_gene(Gene::new(0, 0, 1.0, true, false));
        genome1.add_gene(Gene::new(0, 1, 1.0, true, false));
        let mut genome2 = NeuralNetwork::default();
        genome2.add_gene(Gene::new(0, 0, 0.0, true, false));
        genome2.add_gene(Gene::new(0, 1, 0.0, true, false));
        genome2.add_gene(Gene::new(0, 2, 0.0, true, false));
        assert!(genome1.is_same_specie(&genome2));
    }

    #[test]
    fn two_genomes_with_enought_difference_should_be_in_different_species() {
        let mut genome1 = NeuralNetwork::default();
        genome1.add_gene(Gene::new(0, 0, 1.0, true, false));
        genome1.add_gene(Gene::new(0, 1, 1.0, true, false));
        let mut genome2 = NeuralNetwork::default();
        genome2.add_gene(Gene::new(0, 0, 5.0, true, false));
        genome2.add_gene(Gene::new(0, 1, 5.0, true, false));
        genome2.add_gene(Gene::new(0, 2, 1.0, true, false));
        genome2.add_gene(Gene::new(0, 3, 1.0, true, false));
        assert!(!genome1.is_same_specie(&genome2));
    }

    #[test]
    fn already_existing_gene_should_be_not_duplicated() {
        let mut genome1 = NeuralNetwork::default();
        genome1.add_gene(Gene::new(0, 0, 1.0, true, false));
        genome1.add_connection(0, 0);
        assert_eq!(genome1.genes.len(), 1);
        assert!((genome1.get_genes()[0].weight - 1.0).abs() < EPSILON);
    }

    #[test]
    fn adding_an_existing_gene_disabled_should_enable_original() {
        let mut genome1 = NeuralNetwork::default();
        genome1.add_gene(Gene::new(0, 1, 0.0, true, false));
        genome1.mutate_add_neuron();
        assert!(!genome1.genes[0].enabled);
        assert!(genome1.genes.len() == 3);
        genome1.add_connection(0, 1);
        assert!(genome1.genes[0].enabled);
        assert!((genome1.genes[0].weight - 0.0).abs() < EPSILON);
        assert_eq!(genome1.genes.len(), 3);
    }

    #[test]
    fn genomes_with_same_genes_with_little_differences_on_weight_should_be_in_same_specie() {
        let mut genome1 = NeuralNetwork::default();
        genome1.add_gene(Gene::new(0, 0, 16.0, true, false));
        let mut genome2 = NeuralNetwork::default();
        genome2.add_gene(Gene::new(0, 0, 16.1, true, false));
        assert!(genome1.is_same_specie(&genome2));
    }

    #[test]
    fn genomes_with_same_genes_with_big_differences_on_weight_should_be_in_other_specie() {
        let mut genome1 = NeuralNetwork::default();
        genome1.add_gene(Gene::new(0, 0, 5.0, true, false));
        let mut genome2 = NeuralNetwork::default();
        genome2.add_gene(Gene::new(0, 0, 15.0, true, false));
        assert!(!genome1.is_same_specie(&genome2));
    }


    // From former genome.rs:

    #[test]
    fn should_propagate_signal_without_hidden_layers() {
        let mut organism = NeuralNetwork::default();
        organism.add_gene(Gene::new(0, 1, 5.0, true, false));
        let sensors = vec![7.5];
        let mut output = vec![0.0];
        organism.activate(sensors, &mut output);
        assert!(
            output[0] > 0.9,
            format!("{:?} is not bigger than 0.9", output[0])
        );

        let mut organism = NeuralNetwork::default();
        organism.add_gene(Gene::new(0, 1, -2.0, true, false));
        let sensors = vec![1.0];
        let mut output = vec![0.0];
        organism.activate(sensors, &mut output);
        assert!(
            output[0] < 0.1,
            format!("{:?} is not smaller than 0.1", output[0])
        );
    }

    #[test]
    fn should_propagate_signal_over_hidden_layers() {
        let mut organism = NeuralNetwork::default();
        organism.add_gene(Gene::new(0, 1, 0.0, true, false));
        organism.add_gene(Gene::new(0, 2, 5.0, true, false));
        organism.add_gene(Gene::new(2, 1, 5.0, true, false));
        let sensors = vec![0.0];
        let mut output = vec![0.0];
        organism.activate(sensors, &mut output);
        assert!(
            output[0] > 0.9,
            format!("{:?} is not bigger than 0.9", output[0])
        );
    }

    #[test]
    fn should_work_with_cyclic_networks() {
        let mut organism = NeuralNetwork::default();
        organism.add_gene(Gene::new(0, 1, 2.0, true, false));
        organism.add_gene(Gene::new(1, 2, 2.0, true, false));
        organism.add_gene(Gene::new(2, 1, 2.0, true, false));
        let mut output = vec![0.0];
        organism.activate(vec![1.0], &mut output);
        assert!(
            output[0] > 0.9,
            format!("{:?} is not bigger than 0.9", output[0])
        ); // <- TODO this fails... -7.14... not bigger than 0.9

        let mut organism = NeuralNetwork::default();
        organism.add_gene(Gene::new(0, 1, -2.0, true, false));
        organism.add_gene(Gene::new(1, 2, -2.0, true, false));
        organism.add_gene(Gene::new(2, 1, -2.0, true, false));
        let mut output = vec![0.0];
        organism.activate(vec![1.0], &mut output);
        assert!(
            output[0] < 0.1,
            format!("{:?} is not smaller than 0.1", output[0])
        );
    }

    #[test]
    fn activate_organims_sensor_without_enough_neurons_should_ignore_it() {
        let mut organism = NeuralNetwork::default();
        organism.add_gene(Gene::new(0, 1, 1.0, true, false));
        let sensors = vec![0.0, 0.0, 0.0];
        let mut output = vec![0.0];
        organism.activate(sensors, &mut output);
    }

    #[test]
    fn should_allow_multiple_output() {
        let mut organism = NeuralNetwork::default();
        organism.add_gene(Gene::new(0, 1, 1.0, true, false));
        let sensors = vec![0.0];
        let mut output = vec![0.0, 0.0];
        organism.activate(sensors, &mut output);
    }

    #[test]
    fn should_be_able_to_get_matrix_representation_of_the_neuron_connections() {
        let mut organism = NeuralNetwork::default();
        organism.add_gene(Gene::new(0, 1, 1.0, true, false));
        organism.add_gene(Gene::new(1, 2, 0.5, true, false));
        organism.add_gene(Gene::new(2, 1, 0.5, true, false));
        organism.add_gene(Gene::new(2, 2, 0.75, true, false));
        organism.add_gene(Gene::new(1, 0, 1.0, true, false));
        assert_eq!(
            organism.get_weights(),
            vec![0.0, 1.0, 0.0, 1.0, 0.0, 0.5, 0.0, 0.5, 0.75]
        );
    }

    #[test]
    fn should_not_raise_exception_if_less_neurons_than_required() {
        let mut organism = NeuralNetwork::default();
        organism.add_gene(Gene::new(0, 1, 1.0, true, false));
        let sensors = vec![0.0, 0.0, 0.0];
        let mut output = vec![0.0, 0.0, 0.0];
        organism.activate(sensors, &mut output);
    }
}

