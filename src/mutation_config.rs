use crate::genome::{
    BIAS_GENE_MUTATION_RATE, BIAS_MAX, BIAS_MUTATE_POWER, BIAS_PERTURBED_PROBABILITY,
    WEIGHT_GENE_MUTATION_RATE, WEIGHT_MAX, WEIGHT_MUTATE_POWER,
};
use crate::genome::{
    COMPATIBILITY_BIAS_COEFFICIENT, COMPATIBILITY_DISJOINT_COEFFICIENT,
    COMPATIBILITY_EXCESS_COEFFICIENT, COMPATIBILITY_SMALL_GENOME_GENES, COMPATIBILITY_THRESHOLD,
    COMPATIBILITY_THRESHOLD_MIN, COMPATIBILITY_THRESHOLD_STEP, COMPATIBILITY_WEIGHT_COEFFICIENT,
    MUTATE_ADD_CONNECTION, MUTATE_ADD_NEURON, MUTATE_BIAS, MUTATE_CONNECTION_WEIGHT,
    MUTATE_CONNECTION_WEIGHT_PERTURBED_PROBABILITY, MUTATE_TOGGLE_EXPRESSION, TARGET_SPECIES,
};
use crate::organism::OutputActivation;

/// Configuration for mutation rates in NEAT
///
/// Allows customizing mutation probabilities per problem.
/// Use `MutationConfig::default()` for standard NEAT values defined in `genome.rs`.
#[derive(Debug, Clone, Copy)]
pub struct MutationConfig {
    /// Probability that a mutation touches the connection weights at all.
    /// When it does, each gene is mutated independently with probability
    /// `weight_gene_mutation_rate`.
    pub weight_mutation_rate: f64,
    /// Probability that each connection gene is mutated when the weights
    /// are mutated (so a mutation changes on average
    /// `weight_mutation_rate · weight_gene_mutation_rate` of the weights).
    pub weight_gene_mutation_rate: f64,
    /// Probability of adding a new connection
    pub add_connection_rate: f64,
    /// Probability of adding a new neuron
    pub add_neuron_rate: f64,
    /// Probability of toggling a connection's enabled state
    pub toggle_expression_rate: f64,
    /// Probability that a mutated weight is perturbed rather than replaced:
    /// the replacement probability is `1 - weight_perturbation_rate`. A
    /// replaced weight is drawn uniformly from
    /// `[-weight_init_range, weight_init_range]`.
    pub weight_perturbation_rate: f64,
    /// Weights are clipped to `[-weight_max, weight_max]` after every
    /// mutation (and when a connection is created).
    pub weight_max: f64,
    /// Probability that a mutation touches the neuron biases at all. When it
    /// does, each bias (inputs excluded) is mutated independently with
    /// probability `bias_gene_mutation_rate`.
    pub bias_mutation_rate: f64,
    /// Probability that each neuron bias is mutated when the biases are
    /// mutated.
    pub bias_gene_mutation_rate: f64,
    /// Probability that a mutated bias is perturbed rather than replaced
    /// (replacement probability `1 - bias_perturbation_rate`); a replaced
    /// bias is drawn uniformly from `[-weight_init_range, weight_init_range]`.
    pub bias_perturbation_rate: f64,
    /// Standard deviation σ of the gaussian bias perturbation `N(0, σ²)`.
    pub bias_mutate_power: f64,
    /// Biases are clipped to `[-bias_max, bias_max]` after every mutation.
    pub bias_max: f64,
    /// Compatibility threshold for speciation: two genomes are of the same
    /// species when their compatibility distance is below it. With
    /// `target_species` set this is only the initial value: the population
    /// moves it every generation (see `target_species`).
    pub compatibility_threshold: f64,
    /// `c1` of the compatibility distance (NEAT paper, eq. 1):
    /// `δ = c1·E/N + c2·D/N + c3·W̄ + c_bias·B̄`. Weight of the excess genes.
    pub compatibility_excess_coefficient: f64,
    /// `c2`: weight of the disjoint genes in the compatibility distance.
    pub compatibility_disjoint_coefficient: f64,
    /// `c3`: weight of `W̄`, the average weight difference of the matching
    /// genes (0 when there are none).
    pub compatibility_weight_coefficient: f64,
    /// Weight of `B̄`, the average bias difference of the neurons both
    /// genomes have (inputs excluded; 0 when there are none).
    pub compatibility_bias_coefficient: f64,
    /// `N` criterion: if both genomes have fewer genes than this, `N = 1`;
    /// otherwise `N` is the number of genes of the larger genome. The paper
    /// uses 20; 0 always normalizes. Default `usize::MAX`: never normalize
    /// (`N = 1`), because the jump at 20 genes breaks speciation when the
    /// genomes grow across it (see `genome.rs`).
    pub compatibility_small_genome_genes: usize,
    /// Number of species the dynamic threshold aims at. `Some(n)`: after
    /// every speciation the threshold rises by `compatibility_threshold_step`
    /// if there are more than `n` species and drops by it (down to
    /// `compatibility_threshold_min`) if there are fewer. `None`: fixed
    /// threshold.
    pub target_species: Option<usize>,
    /// Amount the dynamic threshold moves per generation.
    pub compatibility_threshold_step: f64,
    /// The dynamic threshold never goes below this value.
    pub compatibility_threshold_min: f64,
    /// Probability of mutating vs crossing over (0.0 = all crossover, 1.0 = all mutation)
    pub mutation_probability: f64,
    /// Range for initial weight generation: [-weight_init_range, weight_init_range]
    pub weight_init_range: f64,
    /// Standard deviation σ of the gaussian weight perturbation `N(0, σ²)`.
    pub weight_mutate_power: f64,
    /// CTRNN neuron time constant τ (default 0.01).
    /// Represents how fast neurons respond — like biological membrane resistance time.
    /// Small τ (e.g. 0.01): neurons react instantly, network behaves as feedforward.
    /// Large τ (e.g. 0.1-1.0): neurons have inertia, network retains temporal memory.
    /// What matters is the ratio dt/τ where dt=0.01 is the simulation step.
    pub tau: f64,
    /// Simulated time per activate() call in seconds (default 0.1).
    /// With dt=0.01, this means 10 Euler integration steps per activation.
    /// More time = more steps = more accurate convergence but slower execution.
    pub step_time: f64,
    /// How output neurons are read: `σ(y + θ)` (default) or linear `y + θ`
    /// for regression.
    pub output_activation: OutputActivation,
}

impl Default for MutationConfig {
    fn default() -> Self {
        MutationConfig {
            weight_mutation_rate: MUTATE_CONNECTION_WEIGHT,
            weight_gene_mutation_rate: WEIGHT_GENE_MUTATION_RATE,
            add_connection_rate: MUTATE_ADD_CONNECTION,
            add_neuron_rate: MUTATE_ADD_NEURON,
            toggle_expression_rate: MUTATE_TOGGLE_EXPRESSION,
            weight_perturbation_rate: MUTATE_CONNECTION_WEIGHT_PERTURBED_PROBABILITY,
            weight_max: WEIGHT_MAX,
            bias_mutation_rate: MUTATE_BIAS,
            bias_gene_mutation_rate: BIAS_GENE_MUTATION_RATE,
            bias_perturbation_rate: BIAS_PERTURBED_PROBABILITY,
            bias_mutate_power: BIAS_MUTATE_POWER,
            bias_max: BIAS_MAX,
            compatibility_threshold: COMPATIBILITY_THRESHOLD,
            compatibility_excess_coefficient: COMPATIBILITY_EXCESS_COEFFICIENT,
            compatibility_disjoint_coefficient: COMPATIBILITY_DISJOINT_COEFFICIENT,
            compatibility_weight_coefficient: COMPATIBILITY_WEIGHT_COEFFICIENT,
            compatibility_bias_coefficient: COMPATIBILITY_BIAS_COEFFICIENT,
            compatibility_small_genome_genes: COMPATIBILITY_SMALL_GENOME_GENES,
            target_species: Some(TARGET_SPECIES),
            compatibility_threshold_step: COMPATIBILITY_THRESHOLD_STEP,
            compatibility_threshold_min: COMPATIBILITY_THRESHOLD_MIN,
            mutation_probability: 0.4,
            weight_init_range: 1.0,
            weight_mutate_power: WEIGHT_MUTATE_POWER,
            tau: 0.01,
            step_time: 0.1,
            output_activation: OutputActivation::Sigmoid,
        }
    }
}

impl MutationConfig {
    /// Create a new configuration with custom values
    pub fn new() -> MutationConfigBuilder {
        MutationConfigBuilder::default()
    }
}

/// Builder for MutationConfig
#[derive(Debug, Clone, Copy)]
pub struct MutationConfigBuilder {
    config: MutationConfig,
}

impl Default for MutationConfigBuilder {
    fn default() -> Self {
        MutationConfigBuilder {
            config: MutationConfig::default(),
        }
    }
}

impl MutationConfigBuilder {
    /// Set weight mutation rate
    pub fn weight_mutation_rate(mut self, rate: f64) -> Self {
        self.config.weight_mutation_rate = rate;
        self
    }

    /// Set the probability that each connection gene is mutated when the
    /// weights are mutated
    pub fn weight_gene_mutation_rate(mut self, rate: f64) -> Self {
        self.config.weight_gene_mutation_rate = rate;
        self
    }

    /// Set the weight limit: weights stay in `[-weight_max, weight_max]`
    pub fn weight_max(mut self, weight_max: f64) -> Self {
        self.config.weight_max = weight_max;
        self
    }

    /// Set add connection rate
    pub fn add_connection_rate(mut self, rate: f64) -> Self {
        self.config.add_connection_rate = rate;
        self
    }

    /// Set add neuron rate
    pub fn add_neuron_rate(mut self, rate: f64) -> Self {
        self.config.add_neuron_rate = rate;
        self
    }

    /// Set toggle expression rate
    pub fn toggle_expression_rate(mut self, rate: f64) -> Self {
        self.config.toggle_expression_rate = rate;
        self
    }

    /// Set weight perturbation rate
    pub fn weight_perturbation_rate(mut self, rate: f64) -> Self {
        self.config.weight_perturbation_rate = rate;
        self
    }

    /// Set the probability of mutating the neuron biases
    pub fn bias_mutation_rate(mut self, rate: f64) -> Self {
        self.config.bias_mutation_rate = rate;
        self
    }

    /// Set the probability that each neuron bias is mutated when the
    /// biases are mutated
    pub fn bias_gene_mutation_rate(mut self, rate: f64) -> Self {
        self.config.bias_gene_mutation_rate = rate;
        self
    }

    /// Set the probability that a mutated bias is perturbed rather than
    /// replaced
    pub fn bias_perturbation_rate(mut self, rate: f64) -> Self {
        self.config.bias_perturbation_rate = rate;
        self
    }

    /// Set the standard deviation of the gaussian bias perturbation
    pub fn bias_mutate_power(mut self, power: f64) -> Self {
        self.config.bias_mutate_power = power;
        self
    }

    /// Set the bias limit: biases stay in `[-bias_max, bias_max]`
    pub fn bias_max(mut self, bias_max: f64) -> Self {
        self.config.bias_max = bias_max;
        self
    }

    /// Set compatibility threshold
    pub fn compatibility_threshold(mut self, threshold: f64) -> Self {
        self.config.compatibility_threshold = threshold;
        self
    }

    /// Set the compatibility coefficients `c1` (excess), `c2` (disjoint),
    /// `c3` (weights) and the bias coefficient
    pub fn compatibility_coefficients(
        mut self,
        excess: f64,
        disjoint: f64,
        weight: f64,
        bias: f64,
    ) -> Self {
        self.config.compatibility_excess_coefficient = excess;
        self.config.compatibility_disjoint_coefficient = disjoint;
        self.config.compatibility_weight_coefficient = weight;
        self.config.compatibility_bias_coefficient = bias;
        self
    }

    /// Set the `N` criterion: genomes with fewer genes than `genes` are not
    /// normalized (`N = 1`); 0 always normalizes, 20 is the paper value
    pub fn compatibility_small_genome_genes(mut self, genes: usize) -> Self {
        self.config.compatibility_small_genome_genes = genes;
        self
    }

    /// Move the compatibility threshold every generation to aim at
    /// `species` species
    pub fn target_species(mut self, species: usize) -> Self {
        self.config.target_species = Some(species);
        self
    }

    /// Keep the compatibility threshold fixed (no target number of species)
    pub fn fixed_compatibility_threshold(mut self) -> Self {
        self.config.target_species = None;
        self
    }

    /// Set how much the dynamic threshold moves per generation and its
    /// minimum
    pub fn compatibility_threshold_adjustment(mut self, step: f64, min: f64) -> Self {
        self.config.compatibility_threshold_step = step;
        self.config.compatibility_threshold_min = min;
        self
    }

    /// Set mutation vs crossover probability (0.4 = 40% mutation, 60% crossover)
    pub fn mutation_probability(mut self, prob: f64) -> Self {
        self.config.mutation_probability = prob;
        self
    }

    /// Set weight initialization range [-range, range]
    pub fn weight_init_range(mut self, range: f64) -> Self {
        self.config.weight_init_range = range;
        self
    }

    /// Set the standard deviation of the gaussian weight perturbation
    pub fn weight_mutate_power(mut self, power: f64) -> Self {
        self.config.weight_mutate_power = power;
        self
    }

    /// Set CTRNN time constant
    pub fn tau(mut self, tau: f64) -> Self {
        self.config.tau = tau;
        self
    }

    /// Set CTRNN integration time per activate() call
    pub fn step_time(mut self, step_time: f64) -> Self {
        self.config.step_time = step_time;
        self
    }

    /// Set how output neurons are read (sigmoid or linear)
    pub fn output_activation(mut self, output_activation: OutputActivation) -> Self {
        self.config.output_activation = output_activation;
        self
    }

    /// Build the configuration
    pub fn build(self) -> MutationConfig {
        self.config
    }
}
