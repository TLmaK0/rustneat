use crate::ctrnn::{Ctrnn, CtrnnNeuralNetwork};
use crate::genome::Genome;
use std::cmp;
use std::cmp::Ordering;

/// An organism is a Genome with fitness.
/// Also maitain a fitenss measure of the organism
#[allow(missing_docs)]
#[derive(Debug, Clone)]
pub struct Organism {
    pub genome: Genome,
    pub fitness: f64,
    /// Fitness adjusted by species size (fitness sharing)
    pub adjusted_fitness: f64,
    /// If true, skip evaluation and preserve current fitness (used for elitism)
    pub preserve_fitness: bool,
    /// Persistent CTRNN state across activate() calls within an episode
    ctrnn_state: Vec<f64>,
    /// CTRNN neuron time constant τ (default 0.01).
    /// Small τ = feedforward (instant response), large τ = temporal memory (slow response).
    pub tau: f64,
    /// Simulated time per activate() call in seconds (default 0.1).
    /// Number of Euler steps = step_time / dt where dt=0.01.
    pub step_time: f64,
}

impl Ord for Organism {
    fn cmp(&self, other: &Self) -> Ordering {
        other.fitness.partial_cmp(&self.fitness).unwrap()
    }
}

impl Eq for Organism {}

impl PartialEq for Organism {
    fn eq(&self, other: &Self) -> bool {
        self.fitness == other.fitness
    }
}

impl PartialOrd for Organism {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Organism {
    fn cmp(&self, other: &Self) -> Ordering {
        other.fitness.partial_cmp(&self.fitness).unwrap()
    }
}

impl Eq for Organism {}

impl PartialEq for Organism {
    fn eq(&self, other: &Self) -> bool {
        self.fitness == other.fitness
    }
}

impl PartialOrd for Organism {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Organism {
    /// Create a new organmism form a single genome.
    pub fn new(genome: Genome) -> Organism {
        Organism {
            genome: genome,
            fitness: 0f64,
            adjusted_fitness: 0f64,
            preserve_fitness: false,
            ctrnn_state: vec![],
            tau: 0.01,
            step_time: 0.1,
        }
    }
    /// Return a new Orgnaism by mutating this Genome and fitness of zero
    pub fn mutate(&self) -> Organism {
        let mut new_genome = self.genome.clone();
        new_genome.mutate();
        let mut child = Organism::new(new_genome);
        child.tau = self.tau;
        child.step_time = self.step_time;
        child
    }
    /// Return a new Organism by mutating with specific config
    pub fn mutate_with_config(&self, config: &crate::mutation_config::MutationConfig) -> Organism {
        let mut new_genome = self.genome.clone();
        new_genome.mutate_with_config(config);
        let mut child = Organism::new(new_genome);
        child.tau = self.tau;
        child.step_time = self.step_time;
        child
    }
    /// Mate this organism with another
    pub fn mate(&self, other: &Organism) -> Organism {
        let mut child = Organism::new(
            self.genome
                .mate(&other.genome, self.fitness < other.fitness),
        );
        child.tau = self.tau;
        child.step_time = self.step_time;
        child
    }
    /// Reset the internal CTRNN state (call at the start of each episode)
    pub fn reset_state(&mut self) {
        self.ctrnn_state = vec![];
    }

    /// Activate this organism in the NN
    pub fn activate(&mut self, sensors: Vec<f64>, outputs: &mut Vec<f64>) {
        let neurons_len = self.genome.len();
        let sensors_len = sensors.len();

        let tau = vec![self.tau; neurons_len];
        let theta = self.get_bias();

        let mut i = sensors.clone();

        if neurons_len < sensors_len {
            i.truncate(neurons_len);
        } else {
            i = [i, vec![0.0; neurons_len - sensors_len]].concat();
        }

        let wji = self.get_weights();

        // Initialize state if needed (first call or after reset_state())
        if self.ctrnn_state.len() != neurons_len {
            self.ctrnn_state = vec![0.0; neurons_len];
        }

        let activations = Ctrnn::default().activate_nn(
            self.step_time,
            0.01,
            &CtrnnNeuralNetwork {
                y: &self.ctrnn_state.clone(),
                tau: &tau,
                wji: &wji,
                theta: &theta,
                i: &i,
            },
        );

        // Persist state for next activate() call
        self.ctrnn_state = activations.clone();

        if sensors_len < neurons_len {
            let outputs_activations = activations.split_at(sensors_len).1.to_vec();

            for n in 0..cmp::min(outputs_activations.len(), outputs.len()) {
                outputs[n] = outputs_activations[n];
            }
        }
    }

    fn get_weights(&self) -> Vec<f64> {
        let neurons_len = self.genome.len();
        let mut matrix = vec![0.0; neurons_len * neurons_len];
        for gene in self.genome.get_genes() {
            if gene.enabled() {
                matrix[(gene.out_neuron_id() * neurons_len) + gene.in_neuron_id()] = gene.weight()
            }
        }
        matrix
    }

    fn get_bias(&self) -> Vec<f64> {
        let neurons_len = self.genome.len();
        let mut matrix = vec![0.0; neurons_len];
        for gene in self.genome.get_genes() {
            if gene.is_bias() {
                matrix[gene.in_neuron_id()] += 1f64;
            }
        }
        matrix
    }
}

#[cfg(test)]
use crate::gene::Gene;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::Genome;

    #[test]
    fn should_propagate_signal_without_hidden_layers() {
        let mut organism = Organism::new(Genome::default());
        organism.genome.add_gene(Gene::new(0, 1, 1f64, true, false));
        let sensors = vec![1.0];
        let mut output = vec![0f64];
        organism.activate(sensors, &mut output);
        assert!(output[0] > 0.5f64, "{:?} is not bigger than 0.9", output[0]);

        let mut organism = Organism::new(Genome::default());
        organism
            .genome
            .add_gene(Gene::new(0, 1, -2f64, true, false));
        let sensors = vec![1f64];
        let mut output = vec![0f64];
        organism.activate(sensors, &mut output);
        assert!(
            output[0] < 0.1f64,
            "{:?} is not smaller than 0.1",
            output[0]
        );
    }

    #[test]
    fn should_propagate_signal_over_hidden_layers() {
        let mut organism = Organism::new(Genome::default());
        organism.genome.add_gene(Gene::new(0, 1, 0f64, true, false));
        organism.genome.add_gene(Gene::new(0, 2, 5f64, true, false));
        organism.genome.add_gene(Gene::new(2, 1, 5f64, true, false));
        let sensors = vec![0f64];
        let mut output = vec![0f64];
        organism.activate(sensors, &mut output);
        assert!(output[0] > 0.9f64, "{:?} is not bigger than 0.9", output[0]);
    }

    #[test]
    fn should_work_with_cyclic_networks() {
        let mut organism = Organism::new(Genome::default());
        organism.genome.add_gene(Gene::new(0, 1, 2f64, true, false));
        organism.genome.add_gene(Gene::new(1, 2, 2f64, true, false));
        organism.genome.add_gene(Gene::new(2, 1, 2f64, true, false));
        let mut output = vec![0f64];
        organism.activate(vec![10f64], &mut output);
        assert!(output[0] > 0.9, "{:#?} is not bigger than 0.9", output[0]);

        let mut organism = Organism::new(Genome::default());
        organism
            .genome
            .add_gene(Gene::new(0, 1, -2f64, true, false));
        organism
            .genome
            .add_gene(Gene::new(1, 2, -2f64, true, false));
        organism
            .genome
            .add_gene(Gene::new(2, 1, -2f64, true, false));
        let mut output = vec![0f64];
        organism.activate(vec![1f64], &mut output);
        assert!(output[0] < 0.1, "{:?} is not smaller than 0.1", output[0]);
    }

    #[test]
    fn activate_organims_sensor_without_enough_neurons_should_ignore_it() {
        let mut organism = Organism::new(Genome::default());
        organism.genome.add_gene(Gene::new(0, 1, 1f64, true, false));
        let sensors = vec![0f64, 0f64, 0f64];
        let mut output = vec![0f64];
        organism.activate(sensors, &mut output);
    }

    #[test]
    fn should_allow_multiple_output() {
        let mut organism = Organism::new(Genome::default());
        organism.genome.add_gene(Gene::new(0, 1, 1f64, true, false));
        let sensors = vec![0f64];
        let mut output = vec![0f64, 0f64];
        organism.activate(sensors, &mut output);
    }

    #[test]
    fn should_be_able_to_get_matrix_representation_of_the_neuron_connections() {
        let mut organism = Organism::new(Genome::default());
        organism.genome.add_gene(Gene::new(0, 1, 1f64, true, false));
        organism
            .genome
            .add_gene(Gene::new(1, 2, 0.5f64, true, false));
        organism
            .genome
            .add_gene(Gene::new(2, 1, 0.5f64, true, false));
        organism
            .genome
            .add_gene(Gene::new(2, 2, 0.75f64, true, false));
        organism.genome.add_gene(Gene::new(1, 0, 1f64, true, false));
        assert_eq!(
            organism.get_weights(),
            vec![0.0, 1.0, 0.0, 1.0, 0.0, 0.5, 0.0, 0.5, 0.75]
        );
    }

    #[test]
    fn should_not_raise_exception_if_less_neurons_than_required() {
        let mut organism = Organism::new(Genome::default());
        organism.genome.add_gene(Gene::new(0, 1, 1f64, true, false));
        let sensors = vec![0f64, 0f64, 0f64];
        let mut output = vec![0f64, 0f64, 0f64];
        organism.activate(sensors, &mut output);
    }
}
