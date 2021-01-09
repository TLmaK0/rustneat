use ctrnn::{Ctrnn, CtrnnNeuralNetwork};
use genome::Genome;
use std::cmp;
use std::cmp::Ordering;

/// An organism is a Genome with fitness.
/// Also maitain a fitenss measure of the organism
#[allow(missing_docs)]
#[derive(Debug, Clone)]
pub struct Organism {
    pub genome: Genome,
    pub fitness: f64,
}

impl Ord for Organism {
    fn cmp(&self, other: &Self) -> Ordering {
        other.fitness.partial_cmp(&self.fitness).unwrap()
    }
}

impl Eq for Organism { }

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
    pub fn activate(&mut self, sensors: Vec<f64>, outputs: &mut Vec<f64>) {
        let neurons_len = self.genome.len();
        let sensors_len = sensors.len();

        let tau = vec![0.01; neurons_len];
        let theta = self.get_bias();

        let mut i = sensors.clone();

        if neurons_len < sensors_len {
            i.truncate(neurons_len);
        } else {
            i = [i, vec![0.0; neurons_len - sensors_len]].concat();
        }

        let wji = self.get_weights();

        let activations = Ctrnn::default().activate_nn(
            0.1,
            0.01,
            &CtrnnNeuralNetwork {
                y: &vec![0.0; neurons_len],
                tau: &tau,
                wji: &wji,
                theta: &theta,
                i: &i,
            },
        );

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
use gene::Gene;

#[cfg(test)]
mod tests {
    use super::*;
    use genome::Genome;

    #[test]
    fn should_propagate_signal_without_hidden_layers() {
        let mut organism = Organism::new(Genome::default());
        organism.genome.add_gene(Gene::new(0, 1, 1f64, true, false));
        let sensors = vec![1.0];
        let mut output = vec![0f64];
        organism.activate(sensors, &mut output);
        assert!(
            output[0] > 0.5f64,
            format!("{:?} is not bigger than 0.9", output[0])
        );

        let mut organism = Organism::new(Genome::default());
        organism
            .genome
            .add_gene(Gene::new(0, 1, -2f64, true, false));
        let sensors = vec![1f64];
        let mut output = vec![0f64];
        organism.activate(sensors, &mut output);
        assert!(
            output[0] < 0.1f64,
            format!("{:?} is not smaller than 0.1", output[0])
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
        assert!(
            output[0] > 0.9f64,
            format!("{:?} is not bigger than 0.9", output[0])
        );
    }

    #[test]
    fn should_work_with_cyclic_networks() {
        let mut organism = Organism::new(Genome::default());
        organism.genome.add_gene(Gene::new(0, 1, 2f64, true, false));
        organism.genome.add_gene(Gene::new(1, 2, 2f64, true, false));
        organism.genome.add_gene(Gene::new(2, 1, 2f64, true, false));
        let mut output = vec![0f64];
        organism.activate(vec![10f64], &mut output);
        assert!(
            output[0] > 0.9,
            format!("{:?} is not bigger than 0.9", output[0])
        );

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
        assert!(
            output[0] < 0.1,
            format!("{:?} is not smaller than 0.1", output[0])
        );
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
