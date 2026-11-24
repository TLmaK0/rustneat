use crate::ctrnn::{Ctrnn, CtrnnNeuralNetwork};
use crate::genome::Genome;
use crate::innovation::InnovationRegistry;
use crate::mutation_config::MutationConfig;
use rand::Rng;
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
    /// True once `SpeciesEvaluator` has evaluated this organism.
    evaluated: bool,
    /// Persistent CTRNN state across activate() calls within an episode
    ctrnn_state: Vec<f64>,
    /// Default CTRNN time constant τ (default 0.01): that of the neurons
    /// without τ gene (`Genome::tau`). See `MutationConfig::tau`.
    pub tau: f64,
    /// Integration step dt in seconds (default 0.01).
    pub dt: f64,
    /// Simulated time per activate() call in seconds (default 0.1), in
    /// `round(step_time / dt)` integration steps.
    pub step_time: f64,
    /// How output neurons are read (default `OutputActivation::Sigmoid`).
    pub output_activation: OutputActivation,
}

/// Fittest first; a NaN fitness goes after every number, so sorting never
/// panics.
impl Ord for Organism {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.fitness.is_nan(), other.fitness.is_nan()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            (false, false) => other.fitness.partial_cmp(&self.fitness).unwrap(),
        }
    }
}

impl Eq for Organism {}

impl PartialEq for Organism {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
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
            evaluated: false,
            ctrnn_state: vec![],
            tau: 0.01,
            dt: 0.01,
            step_time: 0.1,
            output_activation: OutputActivation::default(),
        }
    }
    /// Return a new Orgnaism by mutating this Genome and fitness of zero
    pub fn mutate(&self) -> Organism {
        let mut new_genome = self.genome.clone();
        new_genome.mutate();
        self.child(new_genome)
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
        self.child(new_genome)
    }
    /// Like `mutate_with_config_and_rng`, taking the ids of new neurons from
    /// `innovations` (see `Genome::mutate_with_innovations`).
    pub fn mutate_with_innovations<R: Rng + ?Sized>(
        &self,
        config: &MutationConfig,
        innovations: &mut InnovationRegistry,
        rng: &mut R,
    ) -> Organism {
        let mut new_genome = self.genome.clone();
        new_genome.mutate_with_innovations(config, innovations, rng);
        self.child(new_genome)
    }
    /// A new organism with `genome` and the CTRNN settings of `self`
    /// (default τ, dt, step time and output activation).
    fn child(&self, genome: Genome) -> Organism {
        let mut child = Organism::new(genome);
        child.tau = self.tau;
        child.dt = self.dt;
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
        self.mate_with_config_and_rng(other, &MutationConfig::default(), rng)
    }
    /// Like `mate_with_rng`, with the crossover settings of `config`
    /// (`MutationConfig::disabled_gene_inheritance`).
    pub fn mate_with_config_and_rng<R: Rng + ?Sized>(
        &self,
        other: &Organism,
        config: &MutationConfig,
        rng: &mut R,
    ) -> Organism {
        // NEAT paper, section 3.2: the child inherits the structure of the
        // fitter parent; with equal fitness, disjoint and excess genes come
        // from either parent at random. The CTRNN settings come from the
        // fitter parent (`self` on a tie); the τ of each neuron is a gene
        // and is inherited with it.
        match self.fitness.partial_cmp(&other.fitness) {
            Some(Ordering::Greater) => {
                self.child(
                    self.genome
                        .mate_with_config_and_rng(&other.genome, true, config, rng),
                )
            }
            Some(Ordering::Less) => {
                other.child(
                    self.genome
                        .mate_with_config_and_rng(&other.genome, false, config, rng),
                )
            }
            _ => self.child(
                self.genome
                    .mate_equal_with_config_and_rng(&other.genome, config, rng),
            ),
        }
    }
    /// True once the population has evaluated this organism (its `fitness`
    /// comes from the environment).
    pub fn is_evaluated(&self) -> bool {
        self.evaluated
    }
    pub(crate) fn mark_evaluated(&mut self) {
        self.evaluated = true;
    }
    /// The organism with the highest fitness, ignoring NaN; ties go to the
    /// first one. None if there is no organism with a fitness that is not NaN.
    pub(crate) fn fittest<'a, I>(organisms: I) -> Option<&'a Organism>
    where
        I: IntoIterator<Item = &'a Organism>,
    {
        organisms
            .into_iter()
            .filter(|organism| !organism.fitness.is_nan())
            .fold(None, |best: Option<&Organism>, organism| match best {
                Some(best) if best.fitness >= organism.fitness => Some(best),
                _ => Some(organism),
            })
    }
    /// Reset the internal CTRNN state, so the next `activate` starts from
    /// rest. Call it at the start of each episode, and before each case when
    /// the cases are independent (see `activate`).
    pub fn reset_state(&mut self) {
        self.ctrnn_state = vec![];
    }

    /// Activate this organism in the NN.
    ///
    /// Neurons `0..sensors.len()` are the inputs and the next `outputs.len()`
    /// neurons are the outputs; both always exist, connected or not. Any
    /// further neuron of the genome (`Genome::neurons`) is hidden; ids that
    /// are not neurons of the genome (with an `InnovationRegistry` they
    /// belong to other genomes) take no place in the network.
    ///
    /// Input neurons are sources: they are not integrated and their value
    /// enters as is, `w * x`, the neurons they are connected to (connections
    /// into an input neuron have no effect). Output and hidden neurons are
    /// CTRNN neurons, which see each other through `σ(y + θ)`, where θ is the
    /// neuron's own bias (`Genome::bias`). Each output is returned as
    /// `σ(y + θ)`, or as `y + θ` with `OutputActivation::Linear`.
    ///
    /// Each call integrates `step_time` seconds in steps of `dt`
    /// (exponential Euler, stable for any τ > 0; each neuron uses its τ gene
    /// or the default `tau`). A change travels one connection per step, so
    /// with the defaults (10 steps) the output of one call only sees inputs
    /// at most 10 connections upstream; longer paths act on later calls.
    ///
    /// **The state persists between calls** until `reset_state`: the output
    /// of a call depends on the inputs of the calls before it. That is what
    /// a control task wants (one episode = many calls), but when the cases
    /// are independent (a classification such as XOR) call `reset_state`
    /// before each one; otherwise evolution may find networks that only
    /// answer right in the order they were evaluated in.
    ///
    /// ```
    /// use rustneat::Population;
    /// let mut organism = Population::create_population_initialized(1, 2, 1)
    ///     .with_seed(1)
    ///     .get_organisms()
    ///     .remove(0);
    /// let mut output = vec![0f64];
    /// for inputs in [[0f64, 0f64], [0f64, 1f64], [1f64, 0f64], [1f64, 1f64]] {
    ///     organism.reset_state(); // stateless: each case starts from rest
    ///     organism.activate(inputs.to_vec(), &mut output);
    ///     assert!(output[0] > 0f64 && output[0] < 1f64);
    /// }
    /// ```
    pub fn activate(&mut self, sensors: Vec<f64>, outputs: &mut Vec<f64>) {
        let inputs_len = sensors.len();
        let fixed_len = inputs_len + outputs.len();
        // Neuron ids of the network, in id order: inputs, outputs, hidden.
        let ids: Vec<usize> = (0..fixed_len)
            .chain(
                self.genome
                    .neurons()
                    .into_iter()
                    .filter(|&id| id >= fixed_len),
            )
            .collect();
        let neurons_len = ids.len();
        // Integrated neurons: outputs and hidden, in neuron id order.
        let integrated_len = neurons_len - inputs_len;

        let weights = self.get_weights(&ids);
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
        let theta: Vec<f64> = ids[inputs_len..]
            .iter()
            .map(|&id| self.genome.bias(id))
            .collect();
        let tau: Vec<f64> = ids[inputs_len..]
            .iter()
            .map(|&id| self.genome.tau(id).unwrap_or(self.tau))
            .collect();

        // Initialize state if needed (first call or after reset_state())
        if self.ctrnn_state.len() != integrated_len {
            self.ctrnn_state = vec![0.0; integrated_len];
        }
        if integrated_len == 0 {
            return;
        }

        let activations = Ctrnn::default().activate_nn(
            self.step_time,
            self.dt,
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

    /// Connection weights as a square matrix over the neurons `ids` (sorted),
    /// row = destination neuron, column = source neuron, both by position in
    /// `ids`.
    fn get_weights(&self, ids: &[usize]) -> Vec<f64> {
        let neurons_len = ids.len();
        let position = |id: usize| ids.binary_search(&id).ok();
        let mut matrix = vec![0.0; neurons_len * neurons_len];
        for gene in self.genome.get_genes() {
            if gene.enabled() {
                if let (Some(from), Some(to)) = (
                    position(gene.in_neuron_id()),
                    position(gene.out_neuron_id()),
                ) {
                    matrix[to * neurons_len + from] = gene.weight()
                }
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
        a.genome.add_gene(Gene::new(0, 1, 1f64, true));
        a.fitness = fitness_a;

        let mut b = Organism::new(Genome::default());
        b.genome.add_gene(Gene::new(0, 1, 2f64, true));
        b.genome.add_gene(Gene::new(0, 2, 2f64, true));
        b.genome.add_gene(Gene::new(2, 1, 2f64, true));
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
        organism.genome.add_gene(Gene::new(0, 1, 1f64, true));
        let sensors = vec![1.0];
        let mut output = vec![0f64];
        organism.activate(sensors, &mut output);
        assert!(output[0] > 0.5f64, "{:?} is not bigger than 0.9", output[0]);

        let mut organism = Organism::new(Genome::default());
        organism.genome.add_gene(Gene::new(0, 1, -3f64, true));
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
        organism.genome.add_gene(Gene::new(0, 1, 0f64, true));
        organism.genome.add_gene(Gene::new(0, 2, 5f64, true));
        organism.genome.add_gene(Gene::new(2, 1, 5f64, true));
        let sensors = vec![0f64];
        let mut output = vec![0f64];
        organism.activate(sensors, &mut output);
        assert!(output[0] > 0.9f64, "{:?} is not bigger than 0.9", output[0]);
    }

    #[test]
    fn should_work_with_cyclic_networks() {
        let mut organism = Organism::new(Genome::default());
        organism.genome.add_gene(Gene::new(0, 1, 2f64, true));
        organism.genome.add_gene(Gene::new(1, 2, 2f64, true));
        organism.genome.add_gene(Gene::new(2, 1, 2f64, true));
        let mut output = vec![0f64];
        organism.activate(vec![10f64], &mut output);
        assert!(output[0] > 0.9, "{:#?} is not bigger than 0.9", output[0]);

        let mut organism = Organism::new(Genome::default());
        organism.genome.add_gene(Gene::new(0, 1, -2f64, true));
        organism.genome.add_gene(Gene::new(1, 2, -2f64, true));
        organism.genome.add_gene(Gene::new(2, 1, -2f64, true));
        let mut output = vec![0f64];
        organism.activate(vec![1f64], &mut output);
        assert!(output[0] < 0.1, "{:?} is not smaller than 0.1", output[0]);
    }

    #[test]
    fn activate_organims_sensor_without_enough_neurons_should_ignore_it() {
        let mut organism = Organism::new(Genome::default());
        organism.genome.add_gene(Gene::new(0, 1, 1f64, true));
        let sensors = vec![0f64, 0f64, 0f64];
        let mut output = vec![0f64];
        organism.activate(sensors, &mut output);
    }

    #[test]
    fn should_allow_multiple_output() {
        let mut organism = Organism::new(Genome::default());
        organism.genome.add_gene(Gene::new(0, 1, 1f64, true));
        let sensors = vec![0f64];
        let mut output = vec![0f64, 0f64];
        organism.activate(sensors, &mut output);
    }

    #[test]
    fn should_be_able_to_get_matrix_representation_of_the_neuron_connections() {
        let mut organism = Organism::new(Genome::default());
        organism.genome.add_gene(Gene::new(0, 1, 1f64, true));
        organism.genome.add_gene(Gene::new(1, 2, 0.5f64, true));
        organism.genome.add_gene(Gene::new(2, 1, 0.5f64, true));
        organism.genome.add_gene(Gene::new(2, 2, 0.75f64, true));
        organism.genome.add_gene(Gene::new(1, 0, 1f64, true));
        assert_eq!(
            organism.get_weights(&organism.genome.neurons()),
            vec![0.0, 1.0, 0.0, 1.0, 0.0, 0.5, 0.0, 0.5, 0.75]
        );
    }

    /// 2 inputs, 1 output and the hidden neurons `hidden`, wired the same way.
    fn network_with_hidden(hidden: [usize; 2]) -> Organism {
        let [h1, h2] = hidden;
        let mut genome = Genome::new_unconnected(2, 1);
        for (from, to, weight) in [
            (0, h1, 3f64),
            (1, h1, -2f64),
            (1, h2, 1.5f64),
            (h1, h2, 2f64),
            (h2, 2, -4f64),
            (h1, 2, 1f64),
            (h2, h2, 0.5f64),
        ] {
            genome.add_gene(Gene::new(from, to, weight, true));
        }
        genome.set_bias(h1, 0.3);
        genome.set_bias(h2, -0.7);
        genome.set_bias(2, 0.1);
        Organism::new(genome)
    }

    #[test]
    fn hidden_neuron_ids_need_not_be_consecutive() {
        // With ids from an innovation registry a genome can have hidden
        // neurons 40 and 900 and nothing in between: the network is the same
        // as with ids 3 and 4, and it does not grow with the ids.
        let mut dense = network_with_hidden([3, 4]);
        let mut sparse = network_with_hidden([40, 900]);
        assert_eq!(sparse.genome.neurons(), vec![0, 1, 2, 40, 900]);
        for sensors in [[0f64, 0f64], [1f64, 0f64], [0f64, 1f64], [1f64, 1f64]] {
            let (mut a, mut b) = (vec![0f64], vec![0f64]);
            dense.activate(sensors.to_vec(), &mut a);
            sparse.activate(sensors.to_vec(), &mut b);
            assert_eq!(a, b);
        }
        assert_eq!(sparse.ctrnn_state.len(), 3);
    }

    #[test]
    fn should_not_raise_exception_if_less_neurons_than_required() {
        let mut organism = Organism::new(Genome::default());
        organism.genome.add_gene(Gene::new(0, 1, 1f64, true));
        let sensors = vec![0f64, 0f64, 0f64];
        let mut output = vec![0f64, 0f64, 0f64];
        organism.activate(sensors, &mut output);
    }

    fn single_connection(weight: f64, activation: OutputActivation) -> Organism {
        let mut organism = Organism::new(Genome::default());
        organism.genome.add_gene(Gene::new(0, 1, weight, true));
        organism.output_activation = activation;
        organism
    }

    /// One activation long enough (1 s, 100 steps of τ) for the network to
    /// settle: these tests are about where it settles, not how fast.
    fn activate_once(organism: &mut Organism, sensors: Vec<f64>, outputs: usize) -> Vec<f64> {
        let mut output = vec![-1f64; outputs];
        organism.step_time = 1.0;
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
        organism.genome.add_gene(Gene::new(0, 2, 1f64, true));
        organism.genome.add_gene(Gene::new(1, 2, 1f64, true));
        organism.genome.add_gene(Gene::new(0, 3, 1f64, true));
        organism.genome.add_gene(Gene::new(3, 2, 1f64, true));
        organism.genome.add_gene(Gene::new(2, 2, 1f64, true));
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
        organism.genome.add_gene(Gene::new(0, 2, 3f64, true));
        organism.output_activation = OutputActivation::Linear;
        assert_eq!(
            activate_once(&mut organism, vec![1f64, 1f64], 2),
            vec![3f64, 0f64]
        );
    }

    #[test]
    fn connections_into_input_neurons_do_not_change_inputs() {
        let mut organism = single_connection(1f64, OutputActivation::Linear);
        organism.genome.add_gene(Gene::new(1, 0, 100f64, true));
        let output = activate_once(&mut organism, vec![1f64], 1);
        assert!((output[0] - 1f64).abs() < 1e-9, "{}", output[0]);
    }

    #[test]
    fn hidden_neuron_receives_input_as_is() {
        // 0 -> 2 (hidden) -> 1 (output): hidden settles on x, output on σ(x).
        let mut organism = Organism::new(Genome::default());
        organism.genome.add_gene(Gene::new(0, 2, 1f64, true));
        organism.genome.add_gene(Gene::new(2, 1, 1f64, true));
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
    fn negative_bias_applies_to_the_output_neuron_itself() {
        for bias in [-3f64, -0.5, 0.0, 2.0] {
            let mut organism = single_connection(1f64, OutputActivation::Sigmoid);
            organism.genome.set_bias(1, bias);
            let output = activate_once(&mut organism, vec![0f64], 1);
            assert!(
                (output[0] - Ctrnn::sigmoid(bias)).abs() < 1e-12,
                "bias {} gave {}",
                bias,
                output[0]
            );
            organism.output_activation = OutputActivation::Linear;
            let output = activate_once(&mut organism, vec![1f64], 1);
            assert!((output[0] - (1f64 + bias)).abs() < 1e-9, "{}", output[0]);
        }
    }

    #[test]
    fn bias_of_a_neuron_does_not_leak_to_the_neurons_it_feeds() {
        // 0 -> 2 (hidden) -> 1 (output). The hidden bias shifts what the
        // hidden neuron emits, σ(x + θ2); the output only adds its own θ1.
        let mut organism = Organism::new(Genome::default());
        organism.genome.add_gene(Gene::new(0, 2, 1f64, true));
        organism.genome.add_gene(Gene::new(2, 1, 1f64, true));
        organism.genome.set_bias(2, -2f64);
        organism.genome.set_bias(1, 0.25);
        organism.output_activation = OutputActivation::Linear;
        organism.step_time = 1.0;
        let output = activate_once(&mut organism, vec![1f64], 1);
        let expected = Ctrnn::sigmoid(1f64 - 2f64) + 0.25;
        assert!((output[0] - expected).abs() < 1e-9, "{}", output[0]);
    }

    #[test]
    fn bias_of_an_input_neuron_has_no_effect() {
        let mut organism = single_connection(2f64, OutputActivation::Linear);
        organism.genome.set_bias(0, 10f64);
        let output = activate_once(&mut organism, vec![1f64], 1);
        assert!((output[0] - 2f64).abs() < 1e-9, "{}", output[0]);
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

    /// Network of the audit of issue #69: 0 -> 1 (w = 1) and 1 -> 1
    /// (w = 0.5), linear output; its output after `activations` calls with
    /// input 1.
    fn audit_output(tau: f64, activations: usize) -> f64 {
        let mut organism = single_connection(1f64, OutputActivation::Linear);
        organism.genome.add_gene(Gene::new(1, 1, 0.5, true));
        organism.tau = tau;
        let mut output = vec![0f64];
        for _ in 0..activations {
            organism.activate(vec![1f64], &mut output);
        }
        output[0]
    }

    #[test]
    fn small_tau_stays_finite_and_converges_to_the_same_fixed_point() {
        // Fixed point: y = 1 + 0.5·σ(y).
        let mut fixed_point = 1f64;
        for _ in 0..200 {
            fixed_point = 1f64 + 0.5 * Ctrnn::sigmoid(fixed_point);
        }
        let reference = audit_output(0.02, 40);
        assert!((reference - fixed_point).abs() < 1e-3, "{}", reference);
        for tau in [0.004, 0.001] {
            for activations in [1, 5, 40] {
                assert!(audit_output(tau, activations).is_finite());
            }
            let output = audit_output(tau, 40);
            assert!(
                (output - reference).abs() < 1e-3,
                "tau {}: {} vs {}",
                tau,
                output,
                reference
            );
        }
    }

    #[test]
    fn each_neuron_uses_its_own_tau() {
        // Same network, the output neuron is slow only through its τ gene.
        let mut fast = single_connection(1f64, OutputActivation::Linear);
        let mut slow = fast.clone();
        slow.genome.set_tau(1, 1.0);
        let (mut a, mut b) = (vec![0f64], vec![0f64]);
        fast.activate(vec![1f64], &mut a);
        slow.activate(vec![1f64], &mut b);
        assert!((a[0] - 1f64).abs() < 1e-3, "{}", a[0]);
        // τ = 1: 1 - e^(-0.1) of the way in 0.1 s.
        assert!((b[0] - (1f64 - (-0.1f64).exp())).abs() < 1e-9, "{}", b[0]);
    }

    /// A chain of `connections` connections from the input to the output
    /// (through `connections - 1` hidden neurons), weight 4.
    fn chain(connections: usize) -> Organism {
        let mut organism = Organism::new(Genome::new_unconnected(1, 1));
        let mut from = 0;
        for k in 0..connections - 1 {
            organism.genome.add_gene(Gene::new(from, 2 + k, 4f64, true));
            from = 2 + k;
        }
        organism.genome.add_gene(Gene::new(from, 1, 4f64, true));
        organism.output_activation = OutputActivation::Linear;
        organism
    }

    /// True if one activation of `organism` (fresh state) depends on the input.
    fn sees_input(organism: &Organism) -> bool {
        let run = |input| {
            let mut organism = organism.clone();
            let mut output = vec![0f64];
            organism.activate(vec![input], &mut output);
            output[0]
        };
        run(5f64) != run(-5f64)
    }

    #[test]
    fn one_activation_propagates_one_connection_per_step() {
        // Default: dt = 0.01, step_time = 0.1, 10 steps per activation.
        assert!(sees_input(&chain(10)));
        assert!(!sees_input(&chain(11)));
        // dt is configurable: 0.05 gives 2 steps.
        let mut two_steps = chain(2);
        two_steps.dt = 0.05;
        assert!(sees_input(&two_steps));
        let mut two_steps = chain(3);
        two_steps.dt = 0.05;
        assert!(!sees_input(&two_steps));
    }

    #[test]
    fn child_takes_ctrnn_settings_of_the_fitter_parent() {
        let mut a = single_connection(1f64, OutputActivation::Linear);
        a.tau = 0.5;
        a.dt = 0.02;
        a.step_time = 0.4;
        a.fitness = 1f64;
        let mut b = single_connection(1f64, OutputActivation::Sigmoid);
        b.fitness = 10f64;
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        for (first, second) in [(&a, &b), (&b, &a)] {
            let child = first.mate_with_rng(second, &mut rng);
            assert_eq!(child.tau, b.tau);
            assert_eq!(child.dt, b.dt);
            assert_eq!(child.step_time, b.step_time);
            assert_eq!(child.output_activation, b.output_activation);
        }
        let child = a.mutate_with_config_and_rng(&MutationConfig::default(), &mut rng);
        assert_eq!((child.tau, child.dt, child.step_time), (0.5, 0.02, 0.4));
    }
}
