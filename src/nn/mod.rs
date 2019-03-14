use crate::{Genome, Params};
use rand::{self, distributions::{Distribution, Uniform}};
use std::cmp;

mod ctrnn;
mod gene;
pub use self::ctrnn::*;
pub use self::gene::*;

/// Vector of Genes
/// Holds a count of last neuron added, similar to Innovation number
// NOTE: With regard to the "competing conventions" problem in the orginal paper:
// Connections are identified by their (in_neuron_id, out_neuron_id) pair, which serves as their 'innovration number'.
// Neurons are identified just by their position in the Vec (implicit id).
#[derive(Debug, Clone)]
pub struct NeuralNetwork {
    /// Connections between neurons. Sorted at all times. Use `add_connection()` to add a
    /// connection!
    // TODO :should it be private with a getter?
    pub connections: Vec<ConnectionGene>,
    /// Neurons with bias. Can simple be pushed to.
    pub neurons: Vec<NeuronGene>,
}
impl Default for NeuralNetwork {
    fn default() -> NeuralNetwork {
        NeuralNetwork {
            connections: Vec::new(),
            neurons: vec![NeuronGene::new(0.0)],
        }
    }
}


impl Genome for NeuralNetwork {
    // http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf - Pag. 110
    // Doesn't distinguish between disjoint and excess genes.
    // Only cares about connection genes (topology).
    fn distance(&self, other: &NeuralNetwork, p: &Params) -> f64 {
        // TODO: optimize this method
        let c2 = p.c2;
        let c3 = p.c3;

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
    fn mutate(&mut self, p: &Params) {
        if rand::random::<f64>() < p.mutate_add_conn_pr || self.connections.is_empty() {
            self.mutate_add_connection();
        };

        if rand::random::<f64>() < p.mutate_add_neuron_pr {
            self.mutate_add_neuron();
        };

        if rand::random::<f64>() < p.mutate_conn_weight_pr {
            self.mutate_connection_weight(p);
        };

        if rand::random::<f64>() < p.mutate_toggle_expr_pr {
            self.mutate_toggle_expression();
        };

        if rand::random::<f64>() < p.mutate_bias_pr {
            self.mutate_bias(p);
        };
    }

    /// Mate two genes. `fittest` is true if `self` is the fittest one
    fn mate(&self, other: &NeuralNetwork, fittest: bool, p: &Params) -> NeuralNetwork {
        let mut genome = NeuralNetwork::default();
        genome.neurons = 
            if self.neurons.len() > other.neurons.len() {
                self.mate_neurons(other, fittest)
            } else {
                other.mate_neurons(self, !fittest)
            };
        genome.connections = 
            if self.connections.len() > other.connections.len() {
                self.mate_connections(other, fittest, p)
            } else {
                other.mate_connections(self, !fittest, p)
            };
        genome
    }

}

impl NeuralNetwork {
    /// Creates a network that with no connections, but enough neurons to cover all inputs and
    /// outputs.
    pub fn with_neurons(n: usize) -> NeuralNetwork {
        let mut neurons = Vec::new();
        for _ in 0..n {
            neurons.push(NeuronGene::new(0.0))
        }
        NeuralNetwork {
            neurons,
            connections: Vec::new(),
        }
    }

    /// Activate the neural network by sending input `inputs` into its first `inputs.len()`
    /// neurons
    pub fn activate(&self, mut inputs: Vec<f64>, outputs: &mut Vec<f64>) {
        let n_neurons = self.n_neurons();
        let n_inputs = inputs.len();

        let tau = vec![1.0; n_neurons];
        let theta = self.get_bias(); 


        if n_neurons < n_inputs {
            inputs.truncate(n_neurons);
        } else {
            inputs = [inputs, vec![0.0; n_neurons - n_inputs]].concat();
        }

        let wij = self.get_weights();

        let activations =
            Ctrnn {
                y: &inputs,  //initial state is the sensors
                delta_t: 1.0,
                tau: &tau,
                wij: &wij,
                theta: &theta,
                i: &inputs
            }.activate_nn(10);

        if n_inputs < n_neurons {
            let outputs_activations = activations.split_at(n_inputs).1.to_vec();

            for n in 0..cmp::min(outputs_activations.len(), outputs.len()) {
                outputs[n] = outputs_activations[n];
            }
        }
    }

    /// Helper function for `activate()`. Get weights of connections (as a matrix represented
    /// linearly)
    pub fn get_weights(&self) -> Vec<f64> {
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
    pub fn get_bias(&self) -> Vec<f64> {
        self.neurons.iter().map(|x| x.bias).collect()
    }
    /// Returns a list of the 'enabled' field along connections
    pub fn get_enabled(&self) -> Vec<bool> { // TODO remove
        self.connections.iter().map(|x| x.enabled).collect()
    }

    /// Get number of neurons
    pub fn n_neurons(&self) -> usize {
        self.neurons.len()
    }
    /// Get number of connections
    pub fn n_connections(&self) -> usize {
        self.connections.len()
    }

    fn mutate_add_connection(&mut self) {
        if self.neurons.len() == 0 {
            return
        }
        let in_neuron_id = rand::random::<usize>() % self.neurons.len();
        let out_neuron_id = rand::random::<usize>() % self.neurons.len();
        // TODO: function to pick multiple random unique values from a range?
        self.add_connection(in_neuron_id, out_neuron_id, ConnectionGene::generate_weight());
    }

    fn mutate_connection_weight(&mut self, p: &Params) {
        // NOTE: the SharpNeat implementation seems to mutate 1 to 3 connections.
        // However, this didn't seem to be any good. Anyway, the code to pick N random connections
        // is still here.
        let mut rng = rand::thread_rng();

        // Random :
        // use rand::seq::index::sample;
        // let n_connections_to_mutate =
            // std::cmp::min(self.connections.len(), rand::random::<usize>() % 3);
        // let selected =
            // sample(&mut rng, self.connections.len(), n_connections_to_mutate).iter();
        let selected = 0..self.connections.len();
        selected.for_each(|i| {
            let perturbation = rand::random::<f64>() < p.mutate_conn_weight_perturbed_pr;

            let mut new_weight = ConnectionGene::generate_weight();
            if perturbation {
                new_weight += self.connections[i].weight;
            }
            self.connections[i].weight = new_weight;
        });
    }

    /// Toggles the expression of a random connection
    fn mutate_toggle_expression(&mut self) {
        if self.connections.len() == 0 {
            return;
        }
        let mut rng = rand::thread_rng();
        let selected_gene = Uniform::from(0..self.connections.len()).sample(&mut rng);
        self.connections[selected_gene].enabled = !self.connections[selected_gene].enabled;
    }

    /// Sets the bias of a random neuron to a sample from the normal distribution
    fn mutate_bias(&mut self, p: &Params) {
        let mut rng = rand::thread_rng();

        let gene = Uniform::from(0..self.neurons.len()).sample(&mut rng);

        let perturbation = rand::random::<f64>() < p.mutate_conn_weight_perturbed_pr;
        let mut new_bias = NeuronGene::generate_bias();
        if perturbation {
            new_bias += self.neurons[gene].bias;
        }
        self.neurons[gene].bias = new_bias;
    }

    fn mutate_add_neuron(&mut self) {
        if self.connections.len() == 0 {
            self.neurons.push(NeuronGene::new(0.0));
        } else {
            // Select a random connections along which to add neuron.. and disable it 
            let mut rng = rand::thread_rng();
            let gene = Uniform::from(0..self.connections.len()).sample(&mut rng);
            self.connections[gene].enabled = false;
            // Create new neuron
            self.neurons.push(NeuronGene::new(0.0));
            let new_neuron_id = self.neurons.len()-1;
            // ... and make two new connections that go through the new neuron
            self.add_connection(self.connections[gene].in_neuron_id(), new_neuron_id, 1.0);
            self.add_connection(new_neuron_id, self.connections[gene].out_neuron_id(),
                                self.connections[gene].weight);
        }
    }


    /// `fittest` is true if `self` is the fittest one
    fn mate_neurons(&self, other: &NeuralNetwork, fittest: bool) -> Vec<NeuronGene> {
        // Guarantee: self.neurons.len() is greater than other.neurons.len()
        let mut genes = Vec::new();
        for i in 0..self.neurons.len() {
            genes.push({
                if fittest {
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
    /// `fittest` is true if `self` is the fittest oneo
    /// This logic is inspired from the implementation of SharpNEAT:
    /// - matching genes always at random (0.5)
    /// - disjoint genes are inherited for sure from the fittest organism,
    /// - disjoint genes are inherited from the not fittest organism with probability `INCLUDE_WEAK_DISJOINT_GENE`
    fn mate_connections(&self, other: &NeuralNetwork, fittest: bool, p: &Params) -> Vec<ConnectionGene> {
        let mut genes = Vec::new();

        // Add all shared genes, as well as the genes unique to `self`
        for gene in &self.connections {
            match other.connections.binary_search(gene) {
                Ok(pos) => {
                    // *Shared* genes are copied from a random parent
                    if rand::random::<f64>() < 0.5 {
                        genes.push(*gene);
                    } else {
                        genes.push(other.connections[pos])
                    }
                }
                Err(_) => {
                    // *Disjoint/excess* gene in `self`
                    if fittest || (!fittest && rand::random::<f64>() < p.include_weak_disjoint_gene) {
                        genes.push(*gene);
                    }
                }
            }
        }

        // Add genes unique to `other`
        for gene in &other.connections {
            if let Err(_) = self.connections.binary_search(gene) {
                if fittest || (!fittest && rand::random::<f64>() < p.include_weak_disjoint_gene) {
                    genes.push(*gene);
                }
            }
        }
        genes.sort();
        genes
    }

    /// Add a new connection. Panics if in_neuron or out_neuron are invalid neuron IDs.
    pub fn add_connection(&mut self, in_neuron: usize, out_neuron: usize, weight: f64) {
        assert!(self.neurons.len() > 0, "add_connection: Tried to add a connection to network with no neurons");
        let new_gene = ConnectionGene::new(in_neuron, out_neuron, weight, true);
        let max_neuron_id = self.neurons.len()-1;

        if in_neuron > max_neuron_id || out_neuron > max_neuron_id {
            panic!("Invalid connection ({} -> {}). (max neuron id = {})",
                   new_gene.in_neuron_id(), new_gene.out_neuron_id(), max_neuron_id);
        }

        match self.connections.binary_search(&new_gene) {
            Ok(pos) => self.connections[pos].enabled = true,
            Err(_) => self.connections.push(new_gene),
        }
        self.connections.sort();
    }


    /// Total weigths of all genes
    pub fn total_weights(&self) -> f64 {
        let mut total = 0.0;
        for gene in &self.connections {
            total += gene.weight;
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use std::f64::EPSILON;
    use crate::{nn::NeuralNetwork, nn::ConnectionGene, Genome, Params};

    #[test]
    fn mutation_connection_weight() {
        let mut genome = NeuralNetwork::with_neurons(1);
        genome.add_connection(0, 0, 1.0);
        let orig_gene = genome.connections[0];
        genome.mutate_connection_weight(&Params::default());
        // These should not be same size
        assert!((genome.connections[0].weight - orig_gene.weight).abs() > EPSILON);
    }

    #[test]
    fn mutation_add_connection() {
        let mut genome = NeuralNetwork::with_neurons(3);
        genome.add_connection(1, 2, ConnectionGene::generate_weight());

        assert!(genome.connections[0].in_neuron_id() == 1);
        assert!(genome.connections[0].out_neuron_id() == 2);
    }

    #[test]
    fn mutation_add_neuron() {
        let mut genome = NeuralNetwork::with_neurons(1);
        genome.mutate_add_connection();
        genome.mutate_add_neuron();
        assert!(!genome.connections[0].enabled);
        assert!(genome.connections[1].in_neuron_id() == genome.connections[0].in_neuron_id());
        assert!(genome.connections[1].out_neuron_id() == 1);
        assert!(genome.connections[2].in_neuron_id() == 1);
        assert!(genome.connections[2].out_neuron_id() == genome.connections[0].out_neuron_id());
    }

    #[test]
    #[should_panic(expected = "Invalid connection (2 -> 2). (max neuron id = 0)")]
    fn try_to_inject_a_unconnected_neuron_gene_should_panic() {
        let mut genome1 = NeuralNetwork::with_neurons(1);
        genome1.add_connection(2, 2, 0.5);
    }

    #[test]
    fn two_genomes_with_little_differences_should_be_in_same_specie() {
        let mut genome1 = NeuralNetwork::with_neurons(2);
        genome1.add_connection(0, 0, 1.0);
        genome1.add_connection(0, 1, 1.0);
        let mut genome2 = NeuralNetwork::with_neurons(3);
        genome2.add_connection(0, 0, 0.0);
        genome2.add_connection(0, 1, 0.0);
        genome2.add_connection(0, 2, 0.0);
        assert!(genome1.is_same_specie(&genome2, &Params::default()));
    }

    #[test]
    fn two_genomes_with_enough_difference_should_be_in_different_species() {
        let p = Params {
            c2: 1.0,
            c3: 0.4,
            ..Default::default()
        };
        let mut genome1 = NeuralNetwork::with_neurons(2);
        genome1.add_connection(0, 0, 1.0);
        genome1.add_connection(0, 1, 1.0);
        let mut genome2 = NeuralNetwork::with_neurons(4);
        genome2.add_connection(0, 0, 5.0);
        genome2.add_connection(0, 1, 5.0);
        genome2.add_connection(0, 2, 1.0);
        genome2.add_connection(0, 3, 1.0);
        assert!(!genome1.is_same_specie(&genome2, &p));
    }

    #[test]
    fn already_existing_gene_should_be_not_duplicated() {
        let mut genome1 = NeuralNetwork::with_neurons(2);
        genome1.add_connection(0, 0, 1.0);
        genome1.add_connection(0, 0, ConnectionGene::generate_weight());
        assert_eq!(genome1.connections.len(), 1);
        assert!((genome1.connections[0].weight - 1.0).abs() < EPSILON);
    }

    #[test]
    fn adding_an_existing_gene_disabled_should_enable_original() {
        let mut genome1 = NeuralNetwork::with_neurons(2);
        genome1.add_connection(0, 1, 0.0);
        genome1.mutate_add_neuron();
        assert!(!genome1.connections[0].enabled);
        assert!(genome1.connections.len() == 3);
        genome1.add_connection(0, 1, ConnectionGene::generate_weight());
        assert!(genome1.connections[0].enabled);
        assert!((genome1.connections[0].weight - 0.0).abs() < EPSILON);
        assert_eq!(genome1.connections.len(), 3);
    }

    #[test]
    fn genomes_with_same_genes_with_little_differences_on_weight_should_be_in_same_specie() {
        let mut genome1 = NeuralNetwork::with_neurons(1);
        genome1.add_connection(0, 0, 16.0);
        let mut genome2 = NeuralNetwork::with_neurons(1);
        genome2.add_connection(0, 0, 16.1);
        assert!(genome1.is_same_specie(&genome2, &Params::default()));
    }

    #[test]
    fn genomes_with_same_genes_with_big_differences_on_weight_should_be_in_other_specie() {
        let p = Params {
            c2: 1.0,
            c3: 0.4,
            ..Default::default()
        };
        let mut genome1 = NeuralNetwork::with_neurons(1);
        genome1.add_connection(0, 0, 5.0);
        let mut genome2 = NeuralNetwork::with_neurons(1);
        genome2.add_connection(0, 0, 15.0);
        assert!(!genome1.is_same_specie(&genome2, &p));
    }


    // From former genome.rs:

    #[test]
    fn should_propagate_signal_without_hidden_layers() {
        let mut organism = NeuralNetwork::with_neurons(2);
        organism.add_connection(0, 1, 5.0);
        let sensors = vec![7.5];
        let mut output = vec![0.0];
        organism.activate(sensors, &mut output);
        assert!(
            output[0] > 0.9,
            format!("{:?} is not bigger than 0.9", output[0])
        );

        let mut organism = NeuralNetwork::with_neurons(2);
        organism.add_connection(0, 1, -2.0);
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
        let mut organism = NeuralNetwork::with_neurons(3);
        organism.add_connection(0, 1, 0.0);
        organism.add_connection(0, 2, 5.0);
        organism.add_connection(2, 1, 5.0);
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
        let mut organism = NeuralNetwork::with_neurons(3);
        organism.add_connection(0, 1, 2.0);
        organism.add_connection(1, 2, 2.0);
        organism.add_connection(2, 1, 2.0);
        let mut output = vec![0.0];
        organism.activate(vec![1.0], &mut output);
        assert!(
            output[0] > 0.9,
            format!("{:?} is not bigger than 0.9", output[0])
        ); // <- TODO this fails... -7.14... not bigger than 0.9

        let mut organism = NeuralNetwork::with_neurons(3);
        organism.add_connection(0, 1, -2.0);
        organism.add_connection(1, 2, -2.0);
        organism.add_connection(2, 1, -2.0);
        let mut output = vec![0.0];
        organism.activate(vec![1.0], &mut output);
        assert!(
            output[0] < 0.1,
            format!("{:?} is not smaller than 0.1", output[0])
        );
    }

    #[test]
    fn activate_organims_sensor_without_enough_neurons_should_ignore_it() {
        let mut organism = NeuralNetwork::with_neurons(2);
        organism.add_connection(0, 1, 1.0);
        let sensors = vec![0.0, 0.0, 0.0];
        let mut output = vec![0.0];
        organism.activate(sensors, &mut output);
    }

    #[test]
    fn should_allow_multiple_output() {
        let mut organism = NeuralNetwork::with_neurons(2);
        organism.add_connection(0, 1, 1.0);
        let sensors = vec![0.0];
        let mut output = vec![0.0, 0.0];
        organism.activate(sensors, &mut output);
    }

    #[test]
    fn should_be_able_to_get_correct_matrix_representation_of_connections() {
        let mut organism = NeuralNetwork::with_neurons(3);
        organism.add_connection(0, 1, 1.0);
        organism.add_connection(1, 2, 0.5);
        organism.add_connection(2, 1, 0.5);
        organism.add_connection(2, 2, 0.75);
        organism.add_connection(1, 0, 1.0);
        assert_eq!(
            organism.get_weights(),
            vec![0.0, 1.0, 0.0, 1.0, 0.0, 0.5, 0.0, 0.5, 0.75]
        );
    }

    #[test]
    fn should_not_raise_exception_if_less_neurons_than_required() {
        let mut organism = NeuralNetwork::with_neurons(2);
        organism.add_connection(0, 1, 1.0);
        let sensors = vec![0.0, 0.0, 0.0];
        let mut output = vec![0.0, 0.0, 0.0];
        organism.activate(sensors, &mut output);
    }

    #[test]
    fn xor_solution() {
        let mut nn = NeuralNetwork::with_neurons(3);
        nn.add_connection(0, 1, -7.782108477795758);
        nn.add_connection(0, 2, 1.3884584755783556);
        nn.add_connection(1, 0, -5.530080797669007);
        nn.add_connection(1, 2, 1.1255631958464876);
        nn.add_connection(2, 1, 0.8066131214269232);

        let mut output = vec![0.0];
        nn.activate(vec![0.0, 0.0], &mut output);
        println!("(0.0, 0.0) -> {}", output[0]);
        assert!(output[0] < 0.15);
        nn.activate(vec![0.0, 1.0], &mut output);
        println!("(0.0, 1.0) -> {}", output[0]);
        assert!(output[0] > 0.85);
        nn.activate(vec![1.0, 0.0], &mut output);
        println!("(1.0, 0.0) -> {}", output[0]);
        assert!(output[0] > 0.85);
        nn.activate(vec![1.0, 1.0], &mut output);
        println!("(1.0, 1.0) -> {}", output[0]);
        assert!(output[0] < 0.15);

    }
}

