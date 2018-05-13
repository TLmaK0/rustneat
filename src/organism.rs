use ctrnn::{Ctrnn, CtrnnNeuralNetwork};
use genome::Genome;
use std::cmp;

/// An organism is a Genome with fitness.
/// Also maitain a fitenss measure of the organism
#[allow(missing_docs)]
#[derive(Debug, Clone)]
pub struct Organism {
    pub genome: Genome,
    pub fitness: f64,
}

impl Organism {
    /// Create a new organmism form a single genome.
    pub fn new(genome: Genome) -> Organism {
        Organism {
            genome: genome,
            fitness: 0f64,
        }
    }
    /// Return a new Orgnaism by mutating this Genome and fitness of zero
    pub fn mutate(&self) -> Organism {
        let mut new_genome = self.genome.clone();
        new_genome.mutate();
        Organism::new(new_genome)
    }
    /// Mate this organism with another
    pub fn mate(&self, other: &Organism) -> Organism {
        Organism::new(
            self.genome
                .mate(&other.genome, self.fitness < other.fitness),
        )
    }
    /// Activate this organism in the NN
    pub fn activate(&mut self, sensors: &[f64], outputs: &mut Vec<f64>) {
        let neurons_len = self.genome.len();
        let gamma = vec![0.0; neurons_len];
        let tau = vec![10.0; neurons_len];
        let theta = vec![0.0; neurons_len];
        let wik = vec![1.0; sensors.len() * neurons_len];
        let i = sensors;
        let wij = self.get_weights_matrix();

        let activations = Ctrnn::default().activate_nn(
            30,
            &CtrnnNeuralNetwork {
                gamma: gamma.as_slice(),
                delta_t: 10.0,
                tau: tau.as_slice(),
                wij: &(wij.0, wij.1, wij.2.as_slice()),
                theta: theta.as_slice(),
                wik: &(neurons_len, sensors.len(), wik.as_slice()),
                i: i,
            },
        );

        if sensors.len() < neurons_len {
            let outputs_activations = activations.split_at(sensors.len()).1.to_vec();

            for n in 0..cmp::min(outputs_activations.len(), outputs.len()) {
                outputs[n] = outputs_activations[n];
            }
        }
    }

    fn get_weights_matrix(&self) -> (usize, usize, Vec<f64>) {
        let neurons_len = self.genome.len();
        let mut matrix = vec![0.0; neurons_len * neurons_len];
        for gene in self.genome.get_genes() {
            if gene.enabled() {
                matrix[(gene.out_neuron_id() * neurons_len) + gene.in_neuron_id()] = gene.weight()
            }
        }
        (neurons_len, neurons_len, matrix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use genome::Genome;

    #[test]
    fn should_propagate_signal_without_hidden_layers() {
        let mut organism = Organism::new(Genome::default());
        organism.genome.inject_gene(0, 1, 5f64);
        let sensors = vec![1f64];
        let mut output = vec![0f64];
        organism.activate(&sensors, &mut output);
        assert!(
            output[0] > 0.9f64,
            format!("{:?} is not bigger than 0.9", output[0])
        );

        let mut organism = Organism::new(Genome::default());
        organism.genome.inject_gene(0, 1, -2f64);
        let sensors = vec![1f64];
        let mut output = vec![0f64];
        organism.activate(&sensors, &mut output);
        assert!(
            output[0] < 0.1f64,
            format!("{:?} is not smaller than 0.1", output[0])
        );
    }

    #[test]
    fn should_propagate_signal_over_hidden_layers() {
        let mut organism = Organism::new(Genome::default());
        organism.genome.inject_gene(0, 1, 0f64);
        organism.genome.inject_gene(0, 2, 5f64);
        organism.genome.inject_gene(2, 1, 5f64);
        let sensors = vec![0f64];
        let mut output = vec![0f64];
        organism.activate(&sensors, &mut output);
        assert!(
            output[0] > 0.9f64,
            format!("{:?} is not bigger than 0.9", output[0])
        );
    }

    #[test]
    fn should_work_with_cyclic_networks() {
        let mut organism = Organism::new(Genome::default());
        organism.genome.inject_gene(0, 1, 2f64);
        organism.genome.inject_gene(1, 2, 2f64);
        organism.genome.inject_gene(2, 1, 2f64);
        let mut output = vec![0f64];
        organism.activate(&[1f64], &mut output);
        assert!(
            output[0] > 0.9,
            format!("{:?} is not bigger than 0.9", output[0])
        );

        let mut organism = Organism::new(Genome::default());
        organism.genome.inject_gene(0, 1, -2f64);
        organism.genome.inject_gene(1, 2, -2f64);
        organism.genome.inject_gene(2, 1, -2f64);
        let mut output = vec![0f64];
        organism.activate(&[1f64], &mut output);
        assert!(
            output[0] < 0.1,
            format!("{:?} is not smaller than 0.1", output[0])
        );
    }

    #[test]
    fn activate_organims_sensor_without_enough_neurons_should_ignore_it() {
        let mut organism = Organism::new(Genome::default());
        organism.genome.inject_gene(0, 1, 1f64);
        let sensors = vec![0f64, 0f64, 0f64];
        let mut output = vec![0f64];
        organism.activate(&sensors, &mut output);
    }

    #[test]
    fn should_allow_multiple_output() {
        let mut organism = Organism::new(Genome::default());
        organism.genome.inject_gene(0, 1, 1f64);
        let sensors = vec![0f64];
        let mut output = vec![0f64, 0f64];
        organism.activate(&sensors, &mut output);
    }

    #[test]
    fn should_be_able_to_get_matrix_representation_of_the_neuron_connections() {
        let mut organism = Organism::new(Genome::default());
        organism.genome.inject_gene(0, 1, 1f64);
        organism.genome.inject_gene(1, 2, 0.5f64);
        organism.genome.inject_gene(2, 1, 0.5f64);
        organism.genome.inject_gene(2, 2, 0.75f64);
        organism.genome.inject_gene(1, 0, 1f64);
        assert_eq!(
            organism.get_weights_matrix(),
            (3, 3, vec![0.0, 1.0, 0.0, 1.0, 0.0, 0.5, 0.0, 0.5, 0.75])
        );
    }

    #[test]
    fn should_not_raise_exception_if_less_neurons_than_required() {
        let mut organism = Organism::new(Genome::default());
        organism.genome.inject_gene(0, 1, 1f64);
        let sensors = vec![0f64, 0f64, 0f64];
        let mut output = vec![0f64, 0f64, 0f64];
        organism.activate(&sensors, &mut output);
    }
}
