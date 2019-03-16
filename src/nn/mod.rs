use crate::{Genome, Params};
use std::cmp;
use indexmap::map::IndexMap;

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
    pub connections: IndexMap<ConnectionId, ConnectionGene>,
    /// Neurons with bias. Can simple be pushed to.
    pub neurons: IndexMap<NeuronId, NeuronGene>,
}
impl Default for NeuralNetwork {
    fn default() -> NeuralNetwork {
        let mut neurons = IndexMap::new();
        neurons.insert(0, NeuronGene::new(0.0, 0));
        NeuralNetwork {
            connections: IndexMap::new(),
            neurons,
        }
    }
}

fn distance<T: Gene>(genome1: &IndexMap<T::Id, T>, genome2: &IndexMap<T::Id, T>, p: &Params) -> f64 {
    let common_genes = genome1.values()
        .filter_map(|gene| {
            genome2.get(&gene.id())
                .map(|other_gene| (*gene, *other_gene))
        })
        .collect::<Vec<(T,T)>>();

    // Disjoint / excess genes
    let disjoint_genes = (genome1.len() + genome2.len()
        - (2 * common_genes.len())) as f64;

    // Get the distance between common genes and neurons
    let genes_distance = common_genes.iter()
        .map(|(gene1, gene2)| gene1.distance(gene2))
        .sum::<f64>();

    let max_genes = std::cmp::max(genome1.len(), genome2.len()) as f64;

    if max_genes == 0.0 {
        0.0
    } else {
        (disjoint_genes * p.distance_disjoint_coef
            + genes_distance * p.distance_weight_coef) / max_genes
    }
}

impl Genome for NeuralNetwork {
    // Inspired by python-neat
    fn distance(&self, other: &NeuralNetwork, p: &Params) -> f64 {
        distance(&self.connections, &other.connections, p)
            + distance(&self.neurons, &other.neurons, p)

    }
    /// May add a connection &| neuron &| mutat connection weight &|
    /// enable/disable connection
    fn mutate(&mut self, innovation_id: &mut usize, p: &Params) {
        use rand::distributions::{Normal, Distribution};
        let mut rng = rand::thread_rng();

        // Topological mutations
        if rand::random::<f64>() < p.mutate_add_conn_pr || self.connections.is_empty() {
            self.mutate_add_connection(p);
        };

        if rand::random::<f64>() < p.mutate_add_neuron_pr {
            self.mutate_add_neuron(*innovation_id);
            *innovation_id += 1;
        };


        if rand::random::<f64>() < p.mutate_toggle_expr_pr {
            self.mutate_toggle_expression();
        };

        // For each connection and neuron, there is some probability to mutate it

        let bias_distr = Normal::new(0.0, p.bias_mutate_var);
        let weight_distr = Normal::new(0.0, p.weight_mutate_var);
        for gene in self.neurons.values_mut() {
            if rand::random::<f64>() < p.bias_mutate_pr {
                gene.bias += bias_distr.sample(&mut rng);
            } else if rand::random::<f64>() < p.bias_replace_pr {
                gene.bias = bias_distr.sample(&mut rng);
            }
        }
        for gene in self.connections.values_mut() {
            if rand::random::<f64>() < p.weight_mutate_pr {
                gene.weight += weight_distr.sample(&mut rng);
            } else if rand::random::<f64>() < p.weight_replace_pr {
                gene.weight = weight_distr.sample(&mut rng);
            }
        }
    }

    /// Mate two genes. `fittest` is true if `self` is the fittest one
    fn mate(&self, other: &NeuralNetwork, fittest: bool, _: &Params) -> NeuralNetwork {
        let (best, worst) = if fittest {(self, other)}
                                else   {(other, self)};
        let mut genome = NeuralNetwork::default();
        genome.neurons = NeuralNetwork::reproduce(&best.neurons, &worst.neurons);
        genome.connections = NeuralNetwork::reproduce(&best.connections, &worst.connections);
        genome
    }

}

impl NeuralNetwork {
    /// Creates a network that with no connections, but enough neurons to cover all inputs and
    /// outputs.
    pub fn with_neurons(n: usize) -> NeuralNetwork {
        let mut neurons = IndexMap::new();
        for i in 0..n {
            neurons.insert(i, NeuronGene::new(0.0, i));
        }
        NeuralNetwork {
            neurons,
            connections: IndexMap::new(),
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
        for gene in self.connections.values() {
            if gene.enabled {
                let (out_neuron_idx,_,_) = self.neurons.get_full(&gene.out_neuron_id()).unwrap();
                let (in_neuron_idx,_,_) = self.neurons.get_full(&gene.in_neuron_id()).unwrap();
                matrix[(out_neuron_idx * n_neurons) + in_neuron_idx] = gene.weight;
            }
        }
        matrix
    }
    /// Helper function for `activate()`. Get bias of neurons.
    pub fn get_bias(&self) -> Vec<f64> {
        self.neurons.values().map(|x| x.bias).collect()
    }

    /// Get number of neurons
    pub fn n_neurons(&self) -> usize {
        self.neurons.len()
    }
    /// Get number of connections
    pub fn n_connections(&self) -> usize {
        self.connections.len()
    }

    fn mutate_add_connection(&mut self, p: &Params) {
        use rand::distributions::{Normal, Distribution};
        let mut rng = rand::thread_rng();
        if self.neurons.len() == 0 {
            return
        }
        // TODO: function to pick multiple random unique values from a range?
        let in_neuron_id = get_random_key(&self.neurons);
        let out_neuron_id = get_random_key(&self.neurons);

        let weight = Normal::new(0.0, p.weight_init_var).sample(&mut rng);
        self.add_connection(in_neuron_id, out_neuron_id, weight);
    }


    /// Toggles the expression of a random connection
    fn mutate_toggle_expression(&mut self) {
        if self.connections.len() == 0 {
            return;
        }
        let selected_gene = get_random_mut(&mut self.connections);
        selected_gene.enabled = !selected_gene.enabled;
    }


    fn mutate_add_neuron(&mut self, innovation_id: usize) {
        if self.connections.len() == 0 {
            let gene = NeuronGene::new(0.0, innovation_id);
            self.neurons.insert(gene.id(), gene);
        } else {
            // Select a random connections along which to add neuron.. and disable it 
            let old_connection = {
                let selected_gene = get_random_mut(&mut self.connections);
                selected_gene.enabled = false;
                *selected_gene
            };
            // Create new neuron
            let new_neuron = NeuronGene::new(0.0, innovation_id);
            self.neurons.insert(new_neuron.id(), new_neuron);
            // ... and make two new connections that go through the new neuron
            self.add_connection(old_connection.in_neuron_id(), new_neuron.id(), 1.0);
            self.add_connection(new_neuron.id(), old_connection.out_neuron_id(),
                                old_connection.weight);
        }
    }

    fn reproduce<T: Gene + Copy>(best: &IndexMap<T::Id, T>, worst: &IndexMap<T::Id, T>) -> IndexMap<T::Id, T> {
        // Copy all disjoint/excess genes from the `best` parent, and randomly
        // cross-over the homologous genes
        let mut genes = IndexMap::new();
        for (id, best) in best.iter() {
            genes.insert(*id,
                if let Some(worst) = worst.get(id) {
                    if rand::random::<f64>() < 0.5 {
                        *best
                    } else {
                        *worst
                    }
                } else {
                    *best
                });
        }
        genes
    }

    /// Add a new connection. Panics if in_neuron or out_neuron are invalid neuron IDs.
    pub fn add_connection(&mut self, in_neuron: NeuronId, out_neuron: NeuronId, weight: f64) {
        assert!(self.neurons.len() > 0, "add_connection: Tried to add a connection to network with no neurons");
        let new_gene = ConnectionGene::new(in_neuron, out_neuron, weight, true);

        assert!(self.neurons.contains_key(&in_neuron));
        assert!(self.neurons.contains_key(&out_neuron));

        if let Some(gene) = self.connections.get_mut(&new_gene.id()) {
            gene.enabled = true;
            gene.weight = weight;
        } else {
            self.connections.insert(new_gene.id(), new_gene);
        }
    }

    /// Total weigths of all genes
    pub fn total_weights(&self) -> f64 {
        let mut total = 0.0;
        for gene in self.connections.values() {
            total += gene.weight;
        }
        total
    }
}

/// Convenience function for selecting a random element in an IndexMap
fn get_random_mut<K, V>(map: &mut IndexMap<K, V>) -> &mut V {
    let idx = rand::random::<usize>() % map.len();
    map.get_index_mut(idx).unwrap().1
}
fn get_random_key<K: Clone, V>(map: &IndexMap<K, V>) -> K {
    let idx = rand::random::<usize>() % map.len();
    map.get_index(idx).unwrap().0.clone()
}

#[cfg(test)]
mod tests {
    use std::f64::EPSILON;
    use crate::{nn::NeuralNetwork, nn::ConnectionGene, Genome, Params};

    #[test]
    fn mutation_connection_weight() {
        let p = Params {
            weight_mutate_pr: 1.0,
            ..Default::default()
        };
        let mut genome = NeuralNetwork::with_neurons(1);
        genome.add_connection(0, 0, 0.0);
        genome.mutate(&mut 0, &p);
        let gene = genome.connections[&(0,0)];
        // These should not be same size
        assert!(gene.weight.abs() > EPSILON);
    }

    #[test]
    fn mutation_add_connection() {
        let mut genome = NeuralNetwork::with_neurons(3);
        genome.add_connection(1, 2, 0.0);

        assert!(genome.connections[&(1,2)].in_neuron_id() == 1);
        assert!(genome.connections[&(1,2)].out_neuron_id() == 2);
    }

    #[test]
    fn mutation_add_neuron() {
        let p = Params::default();
        let mut genome = NeuralNetwork::with_neurons(1);
        genome.mutate_add_connection(&p);
        genome.mutate_add_neuron(1);
        let connections = genome.connections.values().collect::<Vec<_>>();
        assert!(!connections[0].enabled);
        assert!(connections[1].in_neuron_id() == connections[0].in_neuron_id());
        assert!(connections[1].out_neuron_id() == 1);
        assert!(connections[2].in_neuron_id() == 1);
        assert!(connections[2].out_neuron_id() == connections[0].out_neuron_id());
    }

    #[test]
    #[should_panic]
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
    fn two_genomes_with_big_difference_should_be_in_different_species() {
        let p = Params {
            compatibility_threshold: 3.0,
            distance_weight_coef: 1.0,
            distance_disjoint_coef: 1.0,
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
    fn adding_an_existing_gene_disabled_should_enable_original() {
        let mut genome = NeuralNetwork::with_neurons(2);
        genome.add_connection(0, 1, 0.0);
        genome.mutate_add_neuron(2);
        let orig_conn = genome.connections[&(0,1)];
        assert!(!orig_conn.enabled);
        assert!(genome.connections.len() == 3);
        genome.add_connection(0, 1, 1.0);
        let orig_conn = *genome.connections.get_index(0).unwrap().1;
        assert!(orig_conn.enabled);
        assert_eq!(orig_conn.weight, 1.0);
        assert_eq!(genome.connections.len(), 3);
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
    fn genomes_with_big_weight_difference_should_be_in_other_specie() {
        let p = Params {
            ..Default::default()
        };
        let mut genome1 = NeuralNetwork::with_neurons(1);
        genome1.add_connection(0, 0, 0.0);
        let mut genome2 = NeuralNetwork::with_neurons(1);
        genome2.add_connection(0, 0, 30.0);
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
}

