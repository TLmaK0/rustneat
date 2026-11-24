use crate::ctrnn::{Ctrnn, CtrnnNeuralNetwork};
use crate::genome::Genome;
use crate::mutation_config::MutationConfig;
use rand::Rng;
use std::cmp;
use std::cmp::Ordering;

/// How the value of an output neuron is returned by `Organism::activate`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputActivation {
    /// `σ(y + θ)`, always in (0, 1). Default.
    #[default]
    Sigmoid,
    /// `y + θ`, unbounded (for regression).
    Linear,
}

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
    /// How output neurons are read (default `OutputActivation::Sigmoid`).
    pub output_activation: OutputActivation,
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
            output_activation: OutputActivation::default(),
        }
    }
    /// Return a new Orgnaism by mutating this Genome and fitness of zero
    pub fn mutate(&self) -> Organism {
        let mut new_genome = self.genome.clone();
        new_genome.mutate();
        let mut child = Organism::new(new_genome);
        child.tau = self.tau;
        child.step_time = self.step_time;
        child.output_activation = self.output_activation;
        child
    }
    /// Return a new Organism by mutating with specific config
    pub fn mutate_with_config(&self, config: &MutationConfig) -> Organism {
        self.mutate_with_config_and_rng(config, &mut rand::thread_rng())
    }
    /// Return a new Organism by mutating with specific config, drawing every
    /// random decision from `rng`
    pub fn mutate_with_config_and_rng<R: Rng + ?Sized>(
        &self,
        config: &MutationConfig,
        rng: &mut R,
    ) -> Organism {
        let mut new_genome = self.genome.clone();
        new_genome.mutate_with_config_and_rng(config, rng);
        let mut child = Organism::new(new_genome);
        child.tau = self.tau;
        child.step_time = self.step_time;
        child.output_activation = self.output_activation;
        child
    }
    /// Mate this organism with another
    pub fn mate(&self, other: &Organism) -> Organism {
        self.mate_with_rng(other, &mut rand::thread_rng())
    }
    /// Mate this organism with another, drawing every random decision from `rng`
    pub fn mate_with_rng<R: Rng + ?Sized>(&self, other: &Organism, rng: &mut R) -> Organism {
        // NEAT paper, section 3.2: the child inherits the structure of the
        // fitter parent; with equal fitness, disjoint and excess genes come
        // from either parent at random.
        let genome = match self.fitness.partial_cmp(&other.fitness) {
            Some(Ordering::Greater) => self.genome.mate_with_rng(&other.genome, true, rng),
            Some(Ordering::Less) => self.genome.mate_with_rng(&other.genome, false, rng),
            _ => self.genome.mate_equal_with_rng(&other.genome, rng),
        };
        let mut child = Organism::new(genome);
        child.tau = self.tau;
        child.step_time = self.step_time;
        child.output_activation = self.output_activation;
        child
    }
    /// Reset the internal CTRNN state (call at the start of each episode)
    pub fn reset_state(&mut self) {
        self.ctrnn_state = vec![];
    }

    /// Activate this organism in the NN.
    ///
    /// Neurons `0..sensors.len()` are the inputs and the next `outputs.len()`
    /// neurons are the outputs; both always exist, connected or not. Any
    /// further neuron of the genome is hidden.
    ///
    /// Input neurons are sources: they are not integrated and their value
    /// enters as is, `w * x`, the neurons they are connected to (connections
    /// into an input neuron have no effect). Output and hidden neurons are
    /// CTRNN neurons, which see each other through `σ(y + θ)`. Each output is
    /// returned as `σ(y + θ)`, or as `y + θ` with `OutputActivation::Linear`.
    pub fn activate(&mut self, sensors: Vec<f64>, outputs: &mut Vec<f64>) {
        let inputs_len = sensors.len();
        let neurons_len = cmp::max(self.genome.len(), inputs_len + outputs.len());
        // Integrated neurons: outputs and hidden, in neuron id order.
        let integrated_len = neurons_len - inputs_len;

        let weights = self.get_weights(neurons_len);
        let mut wji = vec![0.0; integrated_len * integrated_len];
        let mut i = vec![0.0; integrated_len];
        for row in 0..integrated_len {
            let to = (inputs_len + row) * neurons_len;
            for (from, sensor) in sensors.iter().enumerate() {
                i[row] += weights[to + from] * sensor;
            }
            wji[row * integrated_len..(row + 1) * integrated_len]
                .copy_from_slice(&weights[to + inputs_len..to + neurons_len]);
        }
        let theta = self.get_bias(neurons_len).split_off(inputs_len);
        let tau = vec![self.tau; integrated_len];

        // Initialize state if needed (first call or after reset_state())
        if self.ctrnn_state.len() != integrated_len {
            self.ctrnn_state = vec![0.0; integrated_len];
        }
        if integrated_len == 0 {
            return;
        }

        let activations = Ctrnn::default().activate_nn(
            self.step_time,
            0.01,
            &CtrnnNeuralNetwork {
                y: &self.ctrnn_state,
                tau: &tau,
                wji: &wji,
                theta: &theta,
                i: &i,
            },
        );

        for (n, output) in outputs.iter_mut().enumerate() {
            let value = activations[n] + theta[n];
            *output = match self.output_activation {
                OutputActivation::Sigmoid => Ctrnn::sigmoid(value),
                OutputActivation::Linear => value,
            };
        }

        // Persist state for next activate() call
        self.ctrnn_state = activations;
    }

    /// Connection weights as a `neurons_len` x `neurons_len` matrix, row =
    /// destination neuron, column = source neuron.
    fn get_weights(&self, neurons_len: usize) -> Vec<f64> {
        let mut matrix = vec![0.0; neurons_len * neurons_len];
        for gene in self.genome.get_genes() {
            if gene.enabled() {
                matrix[(gene.out_neuron_id() * neurons_len) + gene.in_neuron_id()] = gene.weight()
            }
        }
        matrix
    }

    fn get_bias(&self, neurons_len: usize) -> Vec<f64> {
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
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn topology(genome: &Genome) -> Vec<(usize, usize)> {
        genome
            .get_genes()
            .iter()
            .map(|g| (g.in_neuron_id(), g.out_neuron_id()))
            .collect()
    }

    /// A: 1 gene, B: 3 genes sharing A's gene, so B has two excess genes.
    fn crossover_parents(fitness_a: f64, fitness_b: f64) -> (Organism, Organism) {
        let mut a = Organism::new(Genome::default());
        a.genome.add_gene(Gene::new(0, 1, 1f64, true, false));
        a.fitness = fitness_a;

        let mut b = Organism::new(Genome::default());
        b.genome.add_gene(Gene::new(0, 1, 2f64, true, false));
        b.genome.add_gene(Gene::new(0, 2, 2f64, true, false));
        b.genome.add_gene(Gene::new(2, 1, 2f64, true, false));
        b.fitness = fitness_b;
        (a, b)
    }

    #[test]
    fn child_inherits_topology_of_fitter_parent_in_both_call_orders() {
        for (fitness_a, fitness_b) in [(10f64, 1f64), (1f64, 10f64)] {
            let (a, b) = crossover_parents(fitness_a, fitness_b);
            let fitter = if a.fitness > b.fitness { &a } else { &b };
            let expected = topology(&fitter.genome);
            let mut rng = ChaCha8Rng::seed_from_u64(1);
            for (first, second) in [(&a, &b), (&b, &a)] {
                let matching = (0..1000)
                    .filter(|_| topology(&first.mate_with_rng(second, &mut rng).genome) == expected)
                    .count();
                assert_eq!(
                    matching, 1000,
                    "fitness {} vs {}: only {}/1000 children have the fitter topology",
                    first.fitness, second.fitness, matching
                );
            }
        }
    }

    #[test]
    fn equal_fitness_inherits_disjoint_genes_from_both_parents_at_random() {
        let (a, b) = crossover_parents(5f64, 5f64);
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        for (first, second) in [(&a, &b), (&b, &a)] {
            let mut with_excess = [0usize; 2];
            for _ in 0..1000 {
                let child = first.mate_with_rng(second, &mut rng);
                let genes = topology(&child.genome);
                assert!(
                    genes.contains(&(0, 1)),
                    "matching gene must always be inherited"
                );
                if genes.contains(&(0, 2)) {
                    with_excess[0] += 1;
                }
                if genes.contains(&(2, 1)) {
                    with_excess[1] += 1;
                }
            }
            for count in with_excess {
                assert!(
                    (400..=600).contains(&count),
                    "excess gene inherited {}/1000 times, expected ~50%",
                    count
                );
            }
        }
    }

    #[test]
    fn equal_fitness_crossover_is_reproducible_with_same_seed() {
        let (a, b) = crossover_parents(5f64, 5f64);
        let run = |seed| {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            (0..50)
                .map(|_| topology(&a.mate_with_rng(&b, &mut rng).genome))
                .collect::<Vec<_>>()
        };
        assert_eq!(run(7), run(7));
    }

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
            .add_gene(Gene::new(0, 1, -3f64, true, false));
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
            organism.get_weights(organism.genome.len()),
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

    fn single_connection(weight: f64, activation: OutputActivation) -> Organism {
        let mut organism = Organism::new(Genome::default());
        organism
            .genome
            .add_gene(Gene::new(0, 1, weight, true, false));
        organism.output_activation = activation;
        organism
    }

    fn activate_once(organism: &mut Organism, sensors: Vec<f64>, outputs: usize) -> Vec<f64> {
        let mut output = vec![-1f64; outputs];
        organism.reset_state();
        organism.activate(sensors, &mut output);
        output
    }

    #[test]
    fn input_zero_contributes_nothing_to_target_neuron() {
        for weight in [-30f64, -1.0, 0.3, 1.0, 7.0, 30.0] {
            let mut linear = single_connection(weight, OutputActivation::Linear);
            assert_eq!(activate_once(&mut linear, vec![0f64], 1), vec![0f64]);
            let mut sigmoid = single_connection(weight, OutputActivation::Sigmoid);
            assert_eq!(activate_once(&mut sigmoid, vec![0f64], 1), vec![0.5f64]);
        }
    }

    #[test]
    fn input_value_enters_target_neuron_as_is() {
        // tau = 0.01 and dt = 0.01: the target neuron settles on w * x.
        for input in [-100f64, -1.0, 0.5, 1.0, 10.0, 100.0] {
            let mut organism = single_connection(2f64, OutputActivation::Linear);
            let output = activate_once(&mut organism, vec![input], 1);
            assert!(
                (output[0] - 2f64 * input).abs() < 1e-9,
                "input {} gave {}, expected {}",
                input,
                output[0],
                2f64 * input
            );
        }
    }

    #[test]
    fn sigmoid_output_is_in_open_unit_interval_with_any_weight() {
        let mut organism = Organism::new(Genome::default());
        organism.genome.add_gene(Gene::new(0, 2, 1f64, true, false));
        organism.genome.add_gene(Gene::new(1, 2, 1f64, true, false));
        organism.genome.add_gene(Gene::new(0, 3, 1f64, true, false));
        organism.genome.add_gene(Gene::new(3, 2, 1f64, true, false));
        organism.genome.add_gene(Gene::new(2, 2, 1f64, true, false));
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        for _ in 0..2000 {
            for gene in organism.genome.get_genes().clone() {
                let mut gene = gene;
                // |y| stays below ~36, where σ is still distinguishable from 0 and 1 in f64
                gene.set_weight(rng.gen_range(-8f64..8f64));
                organism.genome.add_gene(gene);
            }
            let sensors = vec![rng.gen_range(-1f64..1f64), rng.gen_range(-1f64..1f64)];
            let output = activate_once(&mut organism, sensors, 1);
            assert!(
                output[0] > 0f64 && output[0] < 1f64,
                "output {} out of (0, 1)",
                output[0]
            );
        }
        for weight in [-30f64, -7.0, -1.0, 1.0, 7.0, 30.0] {
            let mut organism = single_connection(weight, OutputActivation::Sigmoid);
            let output = activate_once(&mut organism, vec![1f64], 1);
            assert!(output[0] > 0f64 && output[0] < 1f64, "{}", output[0]);
        }
        // Far beyond that range f64 saturates to the closed bounds, never past them.
        for weight in [-1e6f64, 1e6] {
            let mut organism = single_connection(weight, OutputActivation::Sigmoid);
            let output = activate_once(&mut organism, vec![1f64], 1);
            assert!((0f64..=1f64).contains(&output[0]), "{}", output[0]);
        }
    }

    #[test]
    fn input_and_output_neurons_exist_without_connections() {
        let mut organism = Organism::new(Genome::default());
        assert_eq!(
            activate_once(&mut organism, vec![1f64, 1f64], 2),
            vec![0.5f64, 0.5]
        );

        let mut organism = Organism::new(Genome::default());
        organism.genome.add_gene(Gene::new(0, 2, 3f64, true, false));
        organism.output_activation = OutputActivation::Linear;
        assert_eq!(
            activate_once(&mut organism, vec![1f64, 1f64], 2),
            vec![3f64, 0f64]
        );
    }

    #[test]
    fn connections_into_input_neurons_do_not_change_inputs() {
        let mut organism = single_connection(1f64, OutputActivation::Linear);
        organism
            .genome
            .add_gene(Gene::new(1, 0, 100f64, true, false));
        let output = activate_once(&mut organism, vec![1f64], 1);
        assert!((output[0] - 1f64).abs() < 1e-9, "{}", output[0]);
    }

    #[test]
    fn hidden_neuron_receives_input_as_is() {
        // 0 -> 2 (hidden) -> 1 (output): hidden settles on x, output on σ(x).
        let mut organism = Organism::new(Genome::default());
        organism.genome.add_gene(Gene::new(0, 2, 1f64, true, false));
        organism.genome.add_gene(Gene::new(2, 1, 1f64, true, false));
        organism.output_activation = OutputActivation::Linear;
        organism.step_time = 1.0;
        let output = activate_once(&mut organism, vec![-3f64], 1);
        assert!(
            (output[0] - Ctrnn::sigmoid(-3f64)).abs() < 1e-9,
            "{}",
            output[0]
        );
    }

    #[test]
    fn output_activation_is_inherited() {
        let parent = single_connection(1f64, OutputActivation::Linear);
        let config = MutationConfig::default();
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        assert_eq!(
            parent
                .mutate_with_config_and_rng(&config, &mut rng)
                .output_activation,
            OutputActivation::Linear
        );
        assert_eq!(
            parent.mate_with_rng(&parent, &mut rng).output_activation,
            OutputActivation::Linear
        );
    }
}
