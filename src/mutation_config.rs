use crate::ctrnn::Integrator;
use crate::genome::{
    BIAS_GENE_MUTATION_RATE, BIAS_MAX, BIAS_MUTATE_POWER, BIAS_PERTURBED_PROBABILITY, MUTATE_TAU,
    TAU_GENE_MUTATION_RATE, TAU_MAX, TAU_MIN, TAU_MUTATE_POWER, WEIGHT_GENE_MUTATION_RATE,
    WEIGHT_MAX, WEIGHT_MUTATE_POWER,
};
use crate::genome::{
    COMPATIBILITY_BIAS_COEFFICIENT, COMPATIBILITY_DISJOINT_COEFFICIENT,
    COMPATIBILITY_EXCESS_COEFFICIENT, COMPATIBILITY_SMALL_GENOME_GENES,
    COMPATIBILITY_TAU_COEFFICIENT, COMPATIBILITY_THRESHOLD, COMPATIBILITY_THRESHOLD_MIN,
    COMPATIBILITY_THRESHOLD_STEP, COMPATIBILITY_WEIGHT_COEFFICIENT, MUTATE_ADD_CONNECTION,
    MUTATE_ADD_NEURON, MUTATE_BIAS, MUTATE_CONNECTION_WEIGHT,
    MUTATE_CONNECTION_WEIGHT_PERTURBED_PROBABILITY, MUTATE_TOGGLE_EXPRESSION, SPLIT_BIAS,
    SPLIT_INPUT_WEIGHT, TARGET_SPECIES,
};
use crate::organism::OutputActivation;

/// How `speciate` chooses the species of an organism among the species
/// whose representative is under the compatibility threshold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeciesAssignment {
    /// The first compatible species, oldest first (NEAT paper, Stanley &
    /// Miikkulainen 2002: "placed in the first species in which g is
    /// compatible with the representative"). The default: when the dynamic
    /// threshold rises, neighbouring species merge into the older ones, so
    /// `target_species` controls the number of species well.
    FirstCompatible,
    /// The compatible species with the smallest distance to its
    /// representative; ties go to the oldest species (lowest id). This is
    /// what neat-python does: an organism is never pulled into an older
    /// species while a closer one exists.
    ///
    /// Limitation: an organism nearly always stays with its own species'
    /// representative, so raising the threshold does not merge species: it
    /// only stops new ones from appearing, and species disappear only by
    /// stagnation. With `target_species` the number of species overshoots
    /// the target: on the XOR bench (`xor_bench 50 500 init reset=1
    /// assign=1`, seeds 1/101/201/301) about 17 species at the end of a run
    /// for a target of 10 (about 10 with `FirstCompatible`), and medians
    /// 57/58/58/57 generations against 52/53/53/51. While there are too many
    /// species the threshold keeps climbing (about 23 after 80 generations),
    /// which can stop new species from forming for a long time; set
    /// `compatibility_threshold_max` to bound it (6 bounds it without
    /// changing the XOR results; 4 gives about 24 species and slower runs).
    /// Not raising the threshold when no new species was founded was also
    /// tried: about 21 species and medians up to 62. No variant brought the
    /// number of species close to the target.
    Closest,
}

/// How a crossover chooses the second parent (the mate) among the
/// organisms of the species allowed to reproduce. The first parent is
/// always drawn by fitness (roulette wheel on the shifted adjusted fitness).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MateSelection {
    /// Any parent with the same probability (the default).
    Uniform,
    /// By fitness, like the first parent.
    FitnessProportional,
}

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
    /// Weight of `T̄`, the average `|ln τa - ln τb|` of the neurons both
    /// genomes have (inputs excluded). Default 0: τ does not separate
    /// species (see `genome.rs` for the measurements).
    pub compatibility_tau_coefficient: f64,
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
    /// The dynamic threshold never goes above this value (default: no
    /// limit). Mostly useful with `SpeciesAssignment::Closest`.
    pub compatibility_threshold_max: f64,
    /// How an organism compatible with several species chooses one
    /// (default `SpeciesAssignment::FirstCompatible`, as in the paper).
    pub species_assignment: SpeciesAssignment,
    /// Probability of mutating vs crossing over (0.0 = all crossover, 1.0 = all mutation).
    ///
    /// Default 0.4; the NEAT paper uses 0.25. Bench A (`xor_bench 50 500
    /// init reset=1`, seeds 1/101/201/301, 50/50 solved in every case),
    /// median generations: 0.4 gives 52/53/53/51 (mean 52.25), 0.25 gives
    /// 53/55/53/55 (54.0), 0.3 gives 55/53/53/59 (55.0) and 0.5 gives
    /// 54/53/55/51 (53.25), so 0.4 is kept. With the #108 defaults
    /// (survival 0.2, σ_w 2.0): 0.4 gives 30/31/31/30 and 0.25 30/30/30/30
    /// (seeds 401/501/601/701: 30/29/28/31 and 29/28/27/30), within the
    /// noise, so 0.4 is still kept.
    pub mutation_probability: f64,
    /// Probability that a child of crossover is not mutated (NEAT's
    /// `mate_only_prob`, 0.2 in Stanley's code). A crossover of an organism
    /// with itself is always mutated. Default 0: every crossover child is
    /// mutated. Bench A medians, seeds 1/101/201/301, with the defaults
    /// before #108 (survival 0.5, σ_w 1.0): 0 gives 52/53/53/51,
    /// 0.2 56/57/59/54, 0.5 60/59/60/60; always crossover then mutate with
    /// probability 0.25 (`mutation_probability` 0, this 0.75; #38) gives
    /// 93/92/92/94.
    pub mate_only_probability: f64,
    /// How a crossover chooses the second parent (default
    /// `MateSelection::Uniform`; by fitness was not faster on bench A with
    /// the defaults before #108: 55/54/51/55 against 52/53/53/51).
    pub mate_selection: MateSelection,
    /// Probability that a gene disabled in either parent is disabled in the
    /// child (NEAT paper, section 4.1: 0.75). Otherwise it is enabled.
    pub disabled_gene_inheritance: f64,
    /// Probability that a crossover takes the second parent from another
    /// species (NEAT paper, section 4.1: 0.001). That parent is drawn from
    /// the organisms allowed to reproduce in the other species (see
    /// `survival_threshold`); with a single species the mate is always
    /// from the same species.
    pub interspecies_mate_rate: f64,
    /// Probability that a crossover gives every matching gene (and every
    /// neuron both parents have) the average of the two parents' weights
    /// (biases, τ) instead of the value of one parent chosen at random
    /// (default 0.4).
    pub crossover_weight_averaging_rate: f64,
    /// Fraction of each species, the fittest first, allowed to reproduce
    /// (at least one organism always reproduces). The same organisms are
    /// the candidates for an interspecies crossover.
    ///
    /// Default 0.2, as `survival_thresh` in Stanley's NEAT code (0.5, the
    /// fittest half, until #108). Bench A (`xor_bench 50 500 init reset=1`)
    /// medians for seeds 1/101/201/301: 0.1 33/33/32/34, 0.2 37/38/38/39,
    /// 0.3 44/42/41/41, 0.5 52/53/53/51, 0.7 70/69/68/69.
    pub survival_threshold: f64,
    /// A species that gets at least this many offspring keeps its champion:
    /// one of the offspring is an unchanged copy of it (default 2). The NEAT
    /// paper copies the champion of species with more than five members
    /// (6); `usize::MAX` turns this elitism off. 0 behaves like 1.
    pub champion_copy_min_offspring: usize,
    /// Splitting a connection `i -> o` of weight w (add neuron mutation)
    /// adds `i -> new` with this weight, a new neuron with bias
    /// `split_bias` and `new -> o` with weight w. The defaults (5, -2.5)
    /// make the new neuron relay its input in [0, 1] almost unchanged (see
    /// `genome.rs`); the paper uses weight 1 and no bias.
    pub split_input_weight: f64,
    /// Bias of the neuron added by a split (see `split_input_weight`).
    pub split_bias: f64,
    /// Range `(lo, hi)` of the input values (default `(0, 1)`). A split of a
    /// connection from an input rescales the split rule to it: with
    /// `r = hi - lo`, `i -> new` gets weight `split_input_weight / r`, the
    /// new neuron bias `split_bias - split_input_weight·lo / r`, `new -> o`
    /// weight `r·w` and the bias of `o` is shifted by `lo·w`, so the new
    /// neuron relays signed inputs (e.g. `(-1, 1)`) almost unchanged. CTRNN
    /// neurons emit `σ(y + θ)` in (0, 1), so splits of their connections
    /// always use the plain rule. A more neutral split is not always a
    /// faster search (slower on XOR with inputs in [-1, 1], faster in
    /// [-10, 10]; measurements in `genome.rs`): set it when it helps.
    pub split_input_range: (f64, f64),
    /// Range for initial weight generation: [-weight_init_range, weight_init_range]
    pub weight_init_range: f64,
    /// Standard deviation σ of the gaussian weight perturbation `N(0, σ²)`.
    pub weight_mutate_power: f64,
    /// Default CTRNN time constant τ (default 0.01), that of every neuron
    /// without τ gene (`Genome::tau`): new neurons and, while τ has not
    /// mutated, all of them. It is how fast a neuron follows its input:
    /// each integration step it moves `1 - e^(-dt/τ)` of the way to its
    /// target (63 % with dt = τ; see `Integrator::ExponentialEuler`).
    /// Small τ (τ ≤ dt/10): the neuron jumps to its target every step and
    /// the network behaves as a discrete map. Large τ (e.g. 0.1-1.0 with
    /// dt = 0.01): neurons have inertia and the network keeps temporal
    /// memory. Any τ > 0 is stable.
    pub tau: f64,
    /// Integration step dt of the CTRNN in seconds (default 0.01).
    pub dt: f64,
    /// Simulated time per activate() call in seconds (default 0.1).
    /// With dt = 0.01, this means `round(step_time / dt)` = 10 integration
    /// steps per activation. A change travels one connection per step, so
    /// a signal crosses at most that many connections in one activation.
    pub step_time: f64,
    /// Probability that a mutation touches the time constants at all
    /// (default 0: τ does not evolve unless asked). When it does, each τ
    /// (inputs excluded) is mutated with probability
    /// `tau_gene_mutation_rate`: `τ ← τ·e^N(0, σ²)` with
    /// `σ = tau_mutate_power`, clipped to `[tau_min, tau_max]`.
    pub tau_mutation_rate: f64,
    /// Probability that each neuron τ is mutated when the τ are mutated.
    pub tau_gene_mutation_rate: f64,
    /// Standard deviation σ of the log-normal τ perturbation.
    pub tau_mutate_power: f64,
    /// Mutated τ never go below this value (default 0.001, must be > 0).
    pub tau_min: f64,
    /// Mutated τ never go above this value (default 1.0).
    pub tau_max: f64,
    /// How output neurons are read: `σ(y + θ)` (default) or linear `y + θ`
    /// for regression.
    pub output_activation: OutputActivation,
    /// How organisms integrate their CTRNN (default
    /// `Integrator::ExponentialEuler`, stable for any τ > 0).
    /// `Integrator::Euler` (explicit Euler) is only stable for dt/τ < 2:
    /// with τ genes evolving down to `tau_min` it can make networks
    /// diverge. Kept to reproduce older results.
    pub integrator: Integrator,
    /// A species whose best fitness ever has not improved for this many
    /// generations is stagnant and is removed at the start of the next
    /// `evolve`; its share of the offspring goes to the other species
    /// (NEAT paper: 15).
    pub species_stagnation_generations: usize,
    /// The species holding the best organisms of the population (this many
    /// species, ranked by their current champion) are never removed for
    /// stagnation, so the best solution found is never lost.
    pub species_elitism: usize,
    /// For noisy fitness (default false). When false, the elite copied
    /// into the next generation keeps its fitness and is not evaluated
    /// again, and `Population::champion` is the best organism ever
    /// evaluated: right for deterministic environments, but a single lucky
    /// score then keeps an organism as elite and champion for ever.
    ///
    /// When true:
    /// - the elites are evaluated again every generation and their fitness
    ///   is the mean of all their evaluations (`Organism::evaluations`), so
    ///   it converges to their true fitness;
    /// - `Population::champion` is the organism with the best fitness of
    ///   the latest evaluation among those evaluated at least twice (a
    ///   confirmed score), or the best one if none is confirmed yet (the
    ///   first evaluation). It is no longer a historical maximum, so its
    ///   fitness can go down. `Population::epochs_without_improvements`
    ///   still counts against the best champion fitness seen.
    pub reevaluate_elites: bool,
}

/// Probability that a gene disabled in either parent stays disabled.
const DISABLED_GENE_INHERITANCE: f64 = 0.75;
/// Probability that a crossover is between two species.
const INTERSPECIES_MATE_RATE: f64 = 0.001;
/// Generations without improvement before a species is removed.
const SPECIES_STAGNATION_GENERATIONS: usize = 15;
/// Best species protected from the stagnation removal.
const SPECIES_ELITISM: usize = 2;
/// Probability of mutation without crossover (the NEAT paper uses 0.25; see
/// `MutationConfig::mutation_probability` for the measurements).
const MUTATION_PROBABILITY: f64 = 0.4;
/// Probability that a crossover averages the matching weights.
const CROSSOVER_WEIGHT_AVERAGING_RATE: f64 = 0.4;
/// Fraction of each species that reproduces (see
/// `MutationConfig::survival_threshold`).
const SURVIVAL_THRESHOLD: f64 = 0.2;
/// Smallest offspring of a species that keeps an unchanged champion.
const CHAMPION_COPY_MIN_OFFSPRING: usize = 2;
/// Initial weights (and replaced weights and biases) are drawn from
/// `[-WEIGHT_INIT_RANGE, WEIGHT_INIT_RANGE]`.
pub(crate) const WEIGHT_INIT_RANGE: f64 = 1.0;
/// Default CTRNN time constant τ.
pub(crate) const TAU: f64 = 0.01;
/// Default CTRNN integration step.
pub(crate) const DT: f64 = 0.01;
/// Default simulated time per activation.
pub(crate) const STEP_TIME: f64 = 0.1;

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
            compatibility_tau_coefficient: COMPATIBILITY_TAU_COEFFICIENT,
            compatibility_small_genome_genes: COMPATIBILITY_SMALL_GENOME_GENES,
            target_species: Some(TARGET_SPECIES),
            compatibility_threshold_step: COMPATIBILITY_THRESHOLD_STEP,
            compatibility_threshold_min: COMPATIBILITY_THRESHOLD_MIN,
            compatibility_threshold_max: f64::INFINITY,
            species_assignment: SpeciesAssignment::FirstCompatible,
            mutation_probability: MUTATION_PROBABILITY,
            mate_only_probability: 0f64,
            mate_selection: MateSelection::Uniform,
            disabled_gene_inheritance: DISABLED_GENE_INHERITANCE,
            interspecies_mate_rate: INTERSPECIES_MATE_RATE,
            crossover_weight_averaging_rate: CROSSOVER_WEIGHT_AVERAGING_RATE,
            survival_threshold: SURVIVAL_THRESHOLD,
            champion_copy_min_offspring: CHAMPION_COPY_MIN_OFFSPRING,
            split_input_weight: SPLIT_INPUT_WEIGHT,
            split_bias: SPLIT_BIAS,
            split_input_range: (0f64, 1f64),
            weight_init_range: WEIGHT_INIT_RANGE,
            weight_mutate_power: WEIGHT_MUTATE_POWER,
            tau: TAU,
            dt: DT,
            step_time: STEP_TIME,
            tau_mutation_rate: MUTATE_TAU,
            tau_gene_mutation_rate: TAU_GENE_MUTATION_RATE,
            tau_mutate_power: TAU_MUTATE_POWER,
            tau_min: TAU_MIN,
            tau_max: TAU_MAX,
            output_activation: OutputActivation::Sigmoid,
            integrator: Integrator::ExponentialEuler,
            species_stagnation_generations: SPECIES_STAGNATION_GENERATIONS,
            species_elitism: SPECIES_ELITISM,
            reevaluate_elites: false,
        }
    }
}

impl MutationConfig {
    /// A builder starting from `MutationConfig::default()`: set the values
    /// that differ and call `build()`.
    ///
    /// ```
    /// use rustneat::MutationConfig;
    /// let config = MutationConfig::builder().add_neuron_rate(0.1).build();
    /// assert_eq!(config.add_neuron_rate, 0.1);
    /// ```
    pub fn builder() -> MutationConfigBuilder {
        MutationConfigBuilder::default()
    }

    /// The same as `MutationConfig::builder()`: despite its name it returns
    /// a `MutationConfigBuilder`, not a `MutationConfig`.
    // Public API: `new()` has always returned the builder.
    #[allow(clippy::new_ret_no_self)]
    #[deprecated(
        note = "returns a builder, not a `MutationConfig`: use `MutationConfig::builder()` (or `MutationConfig::default()`)"
    )]
    pub fn new() -> MutationConfigBuilder {
        MutationConfig::builder()
    }
}

/// Builder for MutationConfig
#[derive(Debug, Clone, Copy, Default)]
pub struct MutationConfigBuilder {
    config: MutationConfig,
}

impl MutationConfigBuilder {
    /// Set the probability that a gene disabled in either parent is
    /// disabled in the child
    pub fn disabled_gene_inheritance(mut self, probability: f64) -> Self {
        self.config.disabled_gene_inheritance = probability;
        self
    }

    /// Set the probability that a crossover takes its second parent from
    /// another species
    pub fn interspecies_mate_rate(mut self, rate: f64) -> Self {
        self.config.interspecies_mate_rate = rate;
        self
    }

    /// Set the probability that a crossover averages the weights (biases,
    /// τ) of the matching genes instead of taking those of one parent
    pub fn crossover_weight_averaging_rate(mut self, rate: f64) -> Self {
        self.config.crossover_weight_averaging_rate = rate;
        self
    }

    /// Set the fraction of each species (the fittest) allowed to reproduce
    pub fn survival_threshold(mut self, threshold: f64) -> Self {
        self.config.survival_threshold = threshold;
        self
    }

    /// Set the smallest number of offspring for which a species keeps an
    /// unchanged copy of its champion (`usize::MAX`: no elitism)
    pub fn champion_copy_min_offspring(mut self, offspring: usize) -> Self {
        self.config.champion_copy_min_offspring = offspring;
        self
    }

    /// Set the weight of the connection into a neuron added by a split and
    /// the bias of that neuron
    pub fn split_neuron(mut self, input_weight: f64, bias: f64) -> Self {
        self.config.split_input_weight = input_weight;
        self.config.split_bias = bias;
        self
    }

    /// Set the range `[lo, hi]` of the input values the split of a
    /// connection from an input is fit to (see
    /// `MutationConfig::split_input_range`).
    ///
    /// # Panics
    ///
    /// If `lo >= hi` or either bound is not finite.
    pub fn split_input_range(mut self, lo: f64, hi: f64) -> Self {
        assert!(
            lo.is_finite() && hi.is_finite() && lo < hi,
            "split_input_range: need finite lo < hi, got [{}, {}]",
            lo,
            hi
        );
        self.config.split_input_range = (lo, hi);
        self
    }

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

    /// Set the maximum of the dynamic threshold
    pub fn compatibility_threshold_max(mut self, max: f64) -> Self {
        self.config.compatibility_threshold_max = max;
        self
    }

    /// Set how an organism compatible with several species chooses one
    pub fn species_assignment(mut self, assignment: SpeciesAssignment) -> Self {
        self.config.species_assignment = assignment;
        self
    }

    /// Set mutation vs crossover probability (0.4 = 40% mutation, 60% crossover)
    pub fn mutation_probability(mut self, prob: f64) -> Self {
        self.config.mutation_probability = prob;
        self
    }

    /// Set the probability that a crossover child is not mutated
    pub fn mate_only_probability(mut self, prob: f64) -> Self {
        self.config.mate_only_probability = prob;
        self
    }

    /// Set how a crossover chooses the second parent
    pub fn mate_selection(mut self, selection: MateSelection) -> Self {
        self.config.mate_selection = selection;
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

    /// Set the CTRNN integration step dt
    pub fn dt(mut self, dt: f64) -> Self {
        self.config.dt = dt;
        self
    }

    /// Set the probability of mutating the neuron time constants τ
    pub fn tau_mutation_rate(mut self, rate: f64) -> Self {
        self.config.tau_mutation_rate = rate;
        self
    }

    /// Set the probability that each neuron τ is mutated when the τ are
    /// mutated
    pub fn tau_gene_mutation_rate(mut self, rate: f64) -> Self {
        self.config.tau_gene_mutation_rate = rate;
        self
    }

    /// Set σ of the log-normal τ perturbation `τ·e^N(0, σ²)`
    pub fn tau_mutate_power(mut self, power: f64) -> Self {
        self.config.tau_mutate_power = power;
        self
    }

    /// Set the range mutated τ are clipped to, `0 < min <= max`
    pub fn tau_range(mut self, min: f64, max: f64) -> Self {
        assert!(
            min > 0f64 && min <= max,
            "tau range must satisfy 0 < min <= max, got [{}, {}]",
            min,
            max
        );
        self.config.tau_min = min;
        self.config.tau_max = max;
        self
    }

    /// Set the weight of the τ differences in the compatibility distance
    pub fn compatibility_tau_coefficient(mut self, coefficient: f64) -> Self {
        self.config.compatibility_tau_coefficient = coefficient;
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

    /// Set how organisms integrate their CTRNN (see
    /// `MutationConfig::integrator`)
    pub fn integrator(mut self, integrator: Integrator) -> Self {
        self.config.integrator = integrator;
        self
    }

    /// Remove a species after `generations` generations without improving
    /// its best fitness, except the `protected` species with the best
    /// organisms
    pub fn species_stagnation(mut self, generations: usize, protected: usize) -> Self {
        self.config.species_stagnation_generations = generations;
        self.config.species_elitism = protected;
        self
    }

    /// Evaluate the elites again every generation, for noisy fitness
    /// (see `MutationConfig::reevaluate_elites`)
    pub fn reevaluate_elites(mut self, reevaluate: bool) -> Self {
        self.config.reevaluate_elites = reevaluate;
        self
    }

    /// Build the configuration
    pub fn build(self) -> MutationConfig {
        self.config
    }
}
