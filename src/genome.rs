use crate::gene::Gene;
use crate::innovation::InnovationRegistry;
use crate::mutation::Mutation;
use crate::mutation_config::MutationConfig;
use rand::Rng;
use std::cmp;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

/// Identifies the contents of a genome: a new genome and every change to a
/// genome take a value never used before, and a clone keeps it (same
/// contents). `Organism` compares it to know if its compiled phenotype is
/// still the one of its genome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Revision(u64);

impl Revision {
    fn next() -> Revision {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        Revision(NEXT.fetch_add(1, AtomicOrdering::Relaxed))
    }
}

impl Default for Revision {
    fn default() -> Revision {
        Revision::next()
    }
}

/// Vector of Genes
///
/// Neuron ids are the historical markings: inside a `Population` the id of a
/// new neuron comes from its `InnovationRegistry`, so a genome's hidden ids
/// need not be consecutive. A genome mutated on its own (without registry)
/// numbers its new neurons after its highest id.
#[derive(Default, Clone)]
pub struct Genome {
    genes: Vec<Gene>,
    last_neuron_id: usize,
    /// Neuron (node) genes: the bias θ of neuron `id` is `biases[id]`; a
    /// neuron past the end has bias 0.
    biases: Vec<f64>,
    /// Neuron genes too: the time constant τ of neuron `id` is `taus[id]`;
    /// `None` (or past the end) means the default τ of the organism
    /// (`MutationConfig::tau`).
    taus: Vec<Option<f64>>,
    /// Neurons `0..input_neurons` are inputs.
    input_neurons: usize,
    /// Neurons `input_neurons..input_neurons + output_neurons` are outputs.
    output_neurons: usize,
    /// Changes with every change to the genome (see `Revision`).
    revision: Revision,
}

pub(crate) const MUTATE_CONNECTION_WEIGHT: f64 = 0.90f64;
// Structural rates of the NEAT paper (Stanley & Miikkulainen 2002).
pub(crate) const MUTATE_ADD_CONNECTION: f64 = 0.05f64;
pub(crate) const MUTATE_ADD_NEURON: f64 = 0.03f64;
// Splitting a connection `i -> o` of weight w adds `i -> new` with weight
// SPLIT_INPUT_WEIGHT, a new neuron with bias SPLIT_BIAS and `new -> o` with
// weight w. The new neuron relays σ(a·s + θ) where the old connection
// carried s: σ(y + θ) of a CTRNN neuron, in (0, 1), or the raw value of an
// input. a = 5, θ = -2.5 is the least-squares fit of σ(a·s - a/2) to the
// identity on [0, 1] (a = 4.93; a = 4, θ = -2 is the tangent at 0.5), so
// the new neuron changes the output little. The paper's choice (weight 1,
// no bias) relays σ(s), in (0.5, 0.73), not s.
//
// Measured with `split_probe 50 1 2 3 4 5` (every enabled gene of the
// organisms of 5 seeds after 50 generations of bench A, XOR outputs):
// mean |Δoutput| 0.091 (weight 1, bias 0), 0.025 (a = 4), 0.016 (a = 5);
// splits that lower the fitness by more than 0.5: 38 %, 12 %, 7 %.
// Treating input sources apart (weight 1, new -> o weight 4w, bias of o
// lowered by 2w) is not better on XOR inputs (0.026) and needs a weight 4
// times larger.
// Bench A medians, seeds 1/101/201/301: 66/64/63/64, 53/52/57/54,
// 50/52/52/53. (a = 7 is a bit faster in XOR, 48/51/52/52, but less
// neutral: it makes the new neuron a step.)
pub(crate) const SPLIT_INPUT_WEIGHT: f64 = 5f64;
pub(crate) const SPLIT_BIAS: f64 = -2.5f64;
pub(crate) const MUTATE_TOGGLE_EXPRESSION: f64 = 0.001f64;
pub(crate) const MUTATE_BIAS: f64 = 0.90f64;
// Weight and bias mutation (issue #64). When a mutation touches the weights
// (probability MUTATE_CONNECTION_WEIGHT) each gene mutates independently
// with probability WEIGHT_GENE_MUTATION_RATE, so a mutation changes about
// 45% of the weights instead of all of them; the same for the biases. A
// mutated value gets a gaussian perturbation N(0, σ²) or, rarely, is
// replaced, and is then clipped to [-max, max].
//
// Measured with bench A (`xor_bench 50 500 init reset=1`, solved runs and
// median generations, seeds 1 / 101; develop before #64: 50/50 139 / 134):
// * σ_w 0.5 (neat-python): 146 / 152; σ_w 1.0: 103 / 98; σ_w 1.5: 88 / 86
//   (all with per-gene 0.5, replacement 0.1 of the mutated genes, max 30).
// * Replacement is what hurts most: with σ_w = σ_b = 1.0, a replacement
//   probability of 0.2 gives 168 / 157, 0.1 gives 103 / 98, 0.05 gives
//   85 / 83, 0.02 gives 72 / 76, 0 gives 70 / 70. A replaced weight is drawn
//   from [-1, 1] while useful XOR weights are around 10, so it mostly
//   destroys; 0.02 keeps a small escape hatch.
// * Per-gene rate (σ 1.0, no replacement): 0.3 gives 85 / 81, 0.5 gives
//   70 / 70, 0.8 gives 65 / 61, 1.0 (old behaviour, unbounded) 122 / 111.
// * Limit (σ 1.0, no replacement): 30 gives 70 / 70, 10 gives 73 / 76;
//   5 is too tight (σ 1.0, replacement 0.1): 47/50 221, 50/50 256.
// * σ_b 0.5 instead of 1.0 (no replacement): 64 / 65 instead of 70 / 70.
// σ_w = 1.0 is the scale of the initial weights; 1.5 and 2.0 are faster in
// XOR (56 / 56 without replacement) but make bigger jumps, so they are left
// to the problems that want them.
pub(crate) const MUTATE_CONNECTION_WEIGHT_PERTURBED_PROBABILITY: f64 = 0.98f64;
pub(crate) const WEIGHT_GENE_MUTATION_RATE: f64 = 0.5f64;
pub(crate) const WEIGHT_MUTATE_POWER: f64 = 1f64;
pub(crate) const WEIGHT_MAX: f64 = 30f64;
pub(crate) const BIAS_GENE_MUTATION_RATE: f64 = 0.5f64;
pub(crate) const BIAS_PERTURBED_PROBABILITY: f64 = 0.98f64;
pub(crate) const BIAS_MUTATE_POWER: f64 = 0.5f64;
pub(crate) const BIAS_MAX: f64 = 30f64;
// Time constant τ per neuron (issue #69). A mutated τ is multiplied by
// e^N(0, σ²) (τ > 0 and its effect depends on dt/τ, so it is perturbed in
// log scale) and clipped to [TAU_MIN, TAU_MAX]. TAU_MIN = dt/10: with
// exponential Euler a neuron with dt/τ = 10 reaches 99.995 % of its target
// in one step, so a smaller τ changes nothing. TAU_MAX = 10 times the
// default step_time: such a neuron moves about 10 % of the way per
// activation.
//
// Measured with bench A (`xor_bench 50 500 init reset=1`), medians of seeds
// 1/101/201/301/401/501/601/701 (develop before #69, explicit Euler:
// 50 52 52 53 56 52 55 52, mean 52.8):
// * exponential Euler, τ = dt = 0.01, τ not mutated (the defaults):
//   54 54 55 55 55 51 56 53 (54.1). τ = 0.001 (dt/τ = 10, the old
//   discrete map y ← W·σ(y + θ) + I): 50 52 52 54 55 52 54 52 (52.6), so the
//   integrator itself costs nothing; the 1.3 generations are the price of
//   real dynamics with τ = dt (63 % of the way per step).
// * τ mutation 0.1 (per neuron 0.5): σ 0.5 56 55 56 55 56 53 52 54 (54.6),
//   σ 0.25 53 53 55 53 55 55 55 54 (54.1); 0.3 and 0.9 (4 seeds) 54-58.
//   With σ 0.25 and a τ term in the distance (c_τ 0.2): 56 54 53 54 56 52
//   56 57 (54.75); 4 seeds with c_τ 0.5: 53-58.
// XOR with the state reset before every pattern has no use for τ, so τ
// mutation is neutral there at best: it is off by default (no random draw
// either, so seeds reproduce) and so is its distance term. Turn it on for
// tasks with memory (e.g. `tau_mutation_rate(0.1)`).
pub(crate) const MUTATE_TAU: f64 = 0f64;
pub(crate) const TAU_GENE_MUTATION_RATE: f64 = 0.5f64;
pub(crate) const TAU_MUTATE_POWER: f64 = 0.25f64;
pub(crate) const TAU_MIN: f64 = 0.001f64;
pub(crate) const TAU_MAX: f64 = 1f64;
// τ in the compatibility distance: off by default (see MutationConfig).
pub(crate) const COMPATIBILITY_TAU_COEFFICIENT: f64 = 0f64;
// Initial compatibility threshold. The paper uses 3.0, but with N = 1 two
// genomes without genes in common are at distance >= c1 + c2 = 2 (one gene
// each), so 2.0 is the largest value that keeps them apart. With a
// target number of species it is only the starting point.
pub(crate) const COMPATIBILITY_THRESHOLD: f64 = 2f64;
// Compatibility distance of the NEAT paper (Stanley & Miikkulainen 2002,
// eq. 1 and section 4.1): c1 = c2 = 1, c3 = 0.4.
pub(crate) const COMPATIBILITY_EXCESS_COEFFICIENT: f64 = 1f64;
pub(crate) const COMPATIBILITY_DISJOINT_COEFFICIENT: f64 = 1f64;
pub(crate) const COMPATIBILITY_WEIGHT_COEFFICIENT: f64 = 0.4f64;
// A bias is one more weight (that of the neuron), so it counts like c3.
pub(crate) const COMPATIBILITY_BIAS_COEFFICIENT: f64 = 0.4f64;
// The paper uses N = 1 below 20 genes and N = genes of the larger genome
// from there on (set 20 to get it). That jump divides the distance by about
// 20 as soon as one of the two genomes reaches 20 genes: in XOR, where
// genomes grow through 9-22 genes, a representative with 20 genes swallowed
// the whole population every other generation (bench A seed=1: species from
// generation 50 between 1 and 22). Never normalizing (N = 1) keeps the
// distance continuous; the dynamic threshold takes care of the scale.
pub(crate) const COMPATIBILITY_SMALL_GENOME_GENES: usize = usize::MAX;
pub(crate) const TARGET_SPECIES: usize = 10;
pub(crate) const COMPATIBILITY_THRESHOLD_STEP: f64 = 0.3f64;
pub(crate) const COMPATIBILITY_THRESHOLD_MIN: f64 = 0.3f64;

/// A mutated weight or bias: with probability `perturbation_rate` the value
/// plus a gaussian `N(0, power²)` perturbation, otherwise a new value drawn
/// uniformly from `[-init_range, init_range]`.
fn mutated_value<R: Rng + ?Sized>(
    value: f64,
    perturbation_rate: f64,
    power: f64,
    init_range: f64,
    rng: &mut R,
) -> f64 {
    if rng.gen::<f64>() < perturbation_rate {
        value + Gene::generate_gaussian_with_rng(power, rng)
    } else {
        Gene::generate_weight_in_range_with_rng(init_range, rng)
    }
}

/// The genes; the revision is bookkeeping, not part of the genome.
impl std::fmt::Debug for Genome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Genome")
            .field("genes", &self.genes)
            .field("last_neuron_id", &self.last_neuron_id)
            .field("biases", &self.biases)
            .field("taus", &self.taus)
            .field("input_neurons", &self.input_neurons)
            .field("output_neurons", &self.output_neurons)
            .finish()
    }
}

impl Genome {
    /// Create a genome from serialized gene data.
    ///
    /// The genome does not know which neurons are inputs or outputs (both
    /// counts are 0), so mutation may connect into an input and crossover
    /// may drop an unconnected output: use `from_genes_with_io` when they
    /// are known.
    ///
    /// The genome keeps one gene per connection `(in, out)`. Of several
    /// genes with the same ends it keeps the last enabled one or, if none
    /// is enabled, the last one: the gene the phenotype already used, so a
    /// list with duplicates gives the same network as before. The genome
    /// length is at least the highest neuron id of the genes plus one.
    pub fn from_genes(genes: Vec<Gene>, last_neuron_id: usize) -> Genome {
        Genome::from_genes_with_io(genes, last_neuron_id, 0, 0)
    }

    /// Like `from_genes`, for a genome with `input_neurons` inputs (ids
    /// `0..input_neurons`) and `output_neurons` outputs (the ids right
    /// after), as `new_unconnected` would create it: mutation never adds a
    /// connection into an input and crossover keeps every input and output,
    /// connected or not.
    pub fn from_genes_with_io(
        genes: Vec<Gene>,
        last_neuron_id: usize,
        input_neurons: usize,
        output_neurons: usize,
    ) -> Genome {
        let mut genome = Genome::new_unconnected(input_neurons, output_neurons);
        genome.last_neuron_id = genome.last_neuron_id.max(last_neuron_id);
        for gene in genes {
            genome.last_neuron_id = genome
                .last_neuron_id
                .max(gene.in_neuron_id())
                .max(gene.out_neuron_id());
            match genome.genes.binary_search(&gene) {
                // A later gene replaces an earlier one unless it is
                // disabled and the earlier one is enabled.
                Ok(pos) => {
                    if gene.enabled() || !genome.genes[pos].enabled() {
                        genome.genes[pos] = gene;
                    }
                }
                Err(pos) => genome.genes.insert(pos, gene),
            }
        }
        genome
    }

    ///Add initial input and output neurons interconnected
    pub fn new_initialized(input_neurons: usize, output_neurons: usize) -> Genome {
        Genome::new_initialized_with_rng(input_neurons, output_neurons, &mut rand::thread_rng())
    }

    /// Like `new_initialized`, drawing the initial weights in [-1, 1] from `rng`
    pub fn new_initialized_with_rng<R: Rng + ?Sized>(
        input_neurons: usize,
        output_neurons: usize,
        rng: &mut R,
    ) -> Genome {
        let mut genome = Genome::new_unconnected(input_neurons, output_neurons);
        for i in 0..input_neurons {
            for o in 0..output_neurons {
                let weight = Gene::generate_weight_with_rng(rng);
                genome.add_gene(Gene::new(i, input_neurons + o, weight, true));
            }
        }
        genome
    }

    /// Create genome with input/output neuron IDs but NO connections.
    /// NEAT will discover connections through mutation.
    ///
    /// The genome remembers how many inputs and outputs it has: mutation
    /// never adds a connection that ends in an input nor mutates the bias of
    /// an input, and crossover keeps every input and output neuron.
    pub fn new_unconnected(input_neurons: usize, output_neurons: usize) -> Genome {
        Genome {
            genes: Vec::new(),
            last_neuron_id: if input_neurons + output_neurons > 0 {
                input_neurons + output_neurons - 1
            } else {
                0
            },
            biases: Vec::new(),
            taus: Vec::new(),
            input_neurons,
            output_neurons,
            revision: Revision::next(),
        }
    }

    /// Number of input neurons (ids `0..input_neurons()`), 0 if unknown.
    pub fn input_neurons(&self) -> usize {
        self.input_neurons
    }

    /// Number of output neurons (the ids right after the inputs), 0 if unknown.
    pub fn output_neurons(&self) -> usize {
        self.output_neurons
    }

    /// Changes whenever the genome changes; equal for a clone.
    pub(crate) fn revision(&self) -> Revision {
        self.revision
    }

    /// Record a change of the genome.
    fn touch(&mut self) {
        self.revision = Revision::next();
    }

    /// Bias θ of the neuron `neuron_id` (0 for a neuron without one).
    pub fn bias(&self, neuron_id: usize) -> f64 {
        self.biases.get(neuron_id).copied().unwrap_or(0f64)
    }

    /// Ids of the neurons of the genome, in increasing order: its inputs and
    /// outputs, connected or not, and every neuron some gene connects.
    pub fn neurons(&self) -> Vec<usize> {
        let mut ids: Vec<usize> = (0..self.input_neurons + self.output_neurons)
            .chain(
                self.genes
                    .iter()
                    .flat_map(|gene| [gene.in_neuron_id(), gene.out_neuron_id()]),
            )
            .collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    /// True if `neuron_id` is one of `neurons()`.
    pub fn has_neuron(&self, neuron_id: usize) -> bool {
        neuron_id < self.input_neurons + self.output_neurons
            || self
                .genes
                .iter()
                .any(|gene| gene.in_neuron_id() == neuron_id || gene.out_neuron_id() == neuron_id)
    }

    /// Time constant τ of the neuron `neuron_id`, `None` if it has no τ
    /// gene and uses the default τ of the organism.
    pub fn tau(&self, neuron_id: usize) -> Option<f64> {
        self.taus.get(neuron_id).copied().flatten()
    }

    /// Set the time constant τ of the neuron `neuron_id`.
    pub fn set_tau(&mut self, neuron_id: usize, tau: f64) {
        self.store_tau(neuron_id, Some(tau));
    }

    fn store_tau(&mut self, neuron_id: usize, tau: Option<f64>) {
        self.touch();
        if self.taus.len() <= neuron_id {
            self.taus.resize(neuron_id + 1, None);
        }
        self.taus[neuron_id] = tau;
    }

    /// Set the bias θ of the neuron `neuron_id`.
    pub fn set_bias(&mut self, neuron_id: usize, bias: f64) {
        self.touch();
        if self.biases.len() <= neuron_id {
            self.biases.resize(neuron_id + 1, 0f64);
        }
        self.biases[neuron_id] = bias;
    }

    /// May add a connection &| neuron &| mutat connection weight &|
    /// enable/disable connection
    pub fn mutate(&mut self) {
        self.mutate_with_config(&MutationConfig::default());
    }

    /// Mutate using specific mutation rates from config
    pub fn mutate_with_config(&mut self, config: &MutationConfig) {
        self.mutate_with_config_and_rng(config, &mut rand::thread_rng());
    }

    /// Mutate using specific mutation rates from config, drawing every random
    /// decision from `rng`
    pub fn mutate_with_config_and_rng<R: Rng + ?Sized>(
        &mut self,
        config: &MutationConfig,
        rng: &mut R,
    ) {
        self.mutate_with_registry(config, None, rng);
    }

    /// Like `mutate_with_config_and_rng`, but a new neuron takes its id from
    /// `innovations`: splitting a connection gives the same id as the same
    /// split anywhere else in the population. The random draws are the same
    /// as without registry.
    pub fn mutate_with_innovations<R: Rng + ?Sized>(
        &mut self,
        config: &MutationConfig,
        innovations: &mut InnovationRegistry,
        rng: &mut R,
    ) {
        self.mutate_with_registry(config, Some(innovations), rng);
    }

    fn mutate_with_registry<R: Rng + ?Sized>(
        &mut self,
        config: &MutationConfig,
        innovations: Option<&mut InnovationRegistry>,
        rng: &mut R,
    ) {
        self.touch();
        if rng.gen::<f64>() < config.add_connection_rate || self.genes.is_empty() {
            self.mutate_add_connection_with_config(config, rng);
        };

        if rng.gen::<f64>() < config.add_neuron_rate {
            self.mutate_add_neuron_with_registry(config, innovations, rng);
        };

        if rng.gen::<f64>() < config.weight_mutation_rate {
            self.mutate_connection_weight_with_config(config, rng);
        };

        if rng.gen::<f64>() < config.toggle_expression_rate {
            self.mutate_toggle_expression(rng);
        };

        if rng.gen::<f64>() < config.bias_mutation_rate {
            self.mutate_biases_with_config(config, rng);
        };

        // No draw at all when τ does not evolve, so the random sequence is
        // the same as before τ genes existed.
        if config.tau_mutation_rate > 0f64 && rng.gen::<f64>() < config.tau_mutation_rate {
            self.mutate_taus_with_config(config, rng);
        };
    }

    /// Mate two genomes. If `self_is_fitter` is true the child inherits the
    /// structure (disjoint and excess genes) of `self`, otherwise of `other`.
    pub fn mate(&self, other: &Genome, self_is_fitter: bool) -> Genome {
        self.mate_with_rng(other, self_is_fitter, &mut rand::thread_rng())
    }

    /// Mate two genomes, drawing every random decision from `rng`.
    /// If `self_is_fitter` is true the child inherits the structure
    /// (disjoint and excess genes) of `self`, otherwise of `other`.
    pub fn mate_with_rng<R: Rng + ?Sized>(
        &self,
        other: &Genome,
        self_is_fitter: bool,
        rng: &mut R,
    ) -> Genome {
        self.mate_with_config_and_rng(other, self_is_fitter, &MutationConfig::default(), rng)
    }

    /// Like `mate_with_rng`, with the crossover settings of `config`
    /// (`disabled_gene_inheritance`).
    pub fn mate_with_config_and_rng<R: Rng + ?Sized>(
        &self,
        other: &Genome,
        self_is_fitter: bool,
        config: &MutationConfig,
        rng: &mut R,
    ) -> Genome {
        if self_is_fitter {
            self.mate_genes(other, false, config, rng)
        } else {
            other.mate_genes(self, false, config, rng)
        }
    }

    /// Mate two genomes of equal fitness, drawing every random decision from
    /// `rng`: disjoint and excess genes are inherited from either parent at
    /// random (NEAT paper, section 3.2).
    pub fn mate_equal_with_rng<R: Rng + ?Sized>(&self, other: &Genome, rng: &mut R) -> Genome {
        self.mate_equal_with_config_and_rng(other, &MutationConfig::default(), rng)
    }

    /// Like `mate_equal_with_rng`, with the crossover settings of `config`
    /// (`disabled_gene_inheritance`).
    pub fn mate_equal_with_config_and_rng<R: Rng + ?Sized>(
        &self,
        other: &Genome,
        config: &MutationConfig,
        rng: &mut R,
    ) -> Genome {
        self.mate_genes(other, true, config, rng)
    }

    /// Builds the child of `self` (structure donor) and `other`. With
    /// `equal_fitness`, every disjoint/excess gene of either parent is
    /// inherited with probability 0.5 instead of taking those of `self`.
    /// A gene disabled in either parent is disabled in the child with
    /// probability `config.disabled_gene_inheritance`, otherwise enabled.
    fn mate_genes<R: Rng + ?Sized>(
        &self,
        other: &Genome,
        equal_fitness: bool,
        config: &MutationConfig,
        rng: &mut R,
    ) -> Genome {
        let disabled_inheritance = config.disabled_gene_inheritance;
        let mut genome = Genome {
            input_neurons: cmp::max(self.input_neurons, other.input_neurons),
            output_neurons: cmp::max(self.output_neurons, other.output_neurons),
            // The child keeps every neuron of the structure donor (inputs and
            // outputs included, even unconnected), so a new neuron never
            // reuses one of their ids.
            last_neuron_id: self.last_neuron_id,
            ..Genome::default()
        };
        if equal_fitness {
            // Keep every neuron of both parents so a randomly inherited gene
            // never references a neuron the child lacks.
            genome.last_neuron_id = cmp::max(self.last_neuron_id, other.last_neuron_id);
        }
        // NEAT paper: some crossovers average the weights of the matching
        // genes (`crossover_weight_averaging_rate`, 0.4 by default).
        let use_average_weights = rng.gen::<f64>() < config.crossover_weight_averaging_rate;

        for gene in &self.genes {
            let mut child_gene = match other.genes.binary_search(gene) {
                Ok(position) => {
                    // Matching gene in both parents
                    if use_average_weights {
                        // Inherit average weight from both parents
                        let mut avg_gene = *gene;
                        avg_gene.set_weight((gene.weight() + other.genes[position].weight()) / 2.0);
                        avg_gene
                    } else if rng.gen::<f64>() > 0.5 {
                        *gene
                    } else {
                        other.genes[position]
                    }
                }
                Err(_) => {
                    // Disjoint/excess gene: inherit from the fitter parent (self),
                    // or from either parent at random when fitness is equal
                    if equal_fitness && rng.gen::<f64>() < 0.5 {
                        continue;
                    }
                    *gene
                }
            };

            // NEAT rule: a gene disabled in either parent is disabled in
            // the child with probability `disabled_inheritance`.
            let other_gene_disabled = other
                .genes
                .binary_search(gene)
                .ok()
                .map(|pos| !other.genes[pos].enabled())
                .unwrap_or(false);

            if !gene.enabled() || other_gene_disabled {
                if rng.gen::<f64>() < disabled_inheritance {
                    child_gene.set_disabled();
                } else {
                    child_gene.set_enabled();
                }
            }

            genome.add_gene(child_gene);
        }

        if equal_fitness {
            for gene in &other.genes {
                if self.genes.binary_search(gene).is_ok() || rng.gen::<f64>() < 0.5 {
                    continue;
                }
                let mut child_gene = *gene;
                if !gene.enabled() {
                    if rng.gen::<f64>() < disabled_inheritance {
                        child_gene.set_disabled();
                    } else {
                        child_gene.set_enabled();
                    }
                }
                genome.add_gene(child_gene);
            }
        }

        // Neuron genes: a neuron of both parents (the same id, so the same
        // innovation) takes the bias of either at random (or their average
        // when weights are averaged); a neuron of only one parent takes its
        // bias.
        let mut biases = vec![0f64; genome.len()];
        for id in genome.neurons() {
            biases[id] = match (self.has_neuron(id), other.has_neuron(id)) {
                (true, true) if use_average_weights => (self.bias(id) + other.bias(id)) / 2.0,
                (true, true) if rng.gen::<f64>() > 0.5 => self.bias(id),
                (true, true) | (false, true) => other.bias(id),
                (true, false) => self.bias(id),
                (false, false) => 0f64,
            };
        }
        genome.biases = biases;

        // τ genes, the same way; `None` (default τ) is inherited as is. Equal
        // τ draw nothing, so without τ mutation the random sequence is the
        // same as before τ genes existed.
        let mut taus = vec![None; genome.len()];
        for id in genome.neurons() {
            taus[id] = match (self.has_neuron(id), other.has_neuron(id)) {
                (true, true) => match (self.tau(id), other.tau(id)) {
                    (a, b) if a == b => a,
                    (Some(a), Some(b)) if use_average_weights => Some((a + b) / 2.0),
                    (a, b) => {
                        if rng.gen::<f64>() > 0.5 {
                            a
                        } else {
                            b
                        }
                    }
                },
                (false, true) => other.tau(id),
                (true, false) => self.tau(id),
                (false, false) => None,
            };
        }
        genome.taus = taus;
        genome
    }
    /// Get vector of all genes in this genome
    pub fn get_genes(&self) -> &Vec<Gene> {
        &self.genes
    }

    /// only allow connected nodes
    #[deprecated(since = "0.3.0", note = "please use `add_gene` instead")]
    pub fn inject_gene(&mut self, in_neuron_id: usize, out_neuron_id: usize, weight: f64) {
        let gene = Gene::new(in_neuron_id, out_neuron_id, weight, true);
        self.add_gene(gene);
    }
    /// Highest neuron id + 1. With ids from an `InnovationRegistry` it is not
    /// the number of neurons: that is `neurons().len()`.
    pub fn len(&self) -> usize {
        self.last_neuron_id + 1 // first neuron id is 0
    }
    /// is genome empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn mutate_add_connection_with_config<R: Rng + ?Sized>(
        &mut self,
        config: &MutationConfig,
        rng: &mut R,
    ) {
        let candidates = self.new_connection_candidates();
        if candidates.is_empty() {
            return;
        }
        let (in_neuron_id, out_neuron_id) = candidates[rng.gen_range(0..candidates.len())];
        let weight = Gene::generate_weight_in_range_with_rng(config.weight_init_range, rng);
        let gene = Gene::new(
            in_neuron_id,
            out_neuron_id,
            weight.clamp(-config.weight_max, config.weight_max),
            true,
        );
        self.add_gene(gene);
    }

    /// Every connection the add-connection mutation may create, as
    /// `(in_neuron_id, out_neuron_id)`: between two different neurons, not
    /// ending in an input, and not already in the genome (enabled or not).
    ///
    /// Picking one of them uniformly is the same as drawing pairs uniformly
    /// and retrying until the pair is new. This replaces the reservoir
    /// sampling kept from rand 0.4 (`sample_iter`), whose pair order depended
    /// on the ids (with 3 neurons, 1 -> 2 was never chosen), and the old
    /// behaviour of giving up when the drawn connection already existed.
    ///
    /// Only neurons of the genome (`neurons()`) are connected: with ids from
    /// an `InnovationRegistry` the ids in between belong to other genomes.
    /// A genome that does not know its inputs (`Genome::default`) with a
    /// single neuron can still connect it to itself, as before.
    fn new_connection_candidates(&self) -> Vec<(usize, usize)> {
        let exists = |from: usize, to: usize| {
            self.genes
                .binary_search(&Gene::new_connection(from, to))
                .is_ok()
        };
        if self.input_neurons == 0 && self.len() == 1 {
            return if exists(0, 0) { vec![] } else { vec![(0, 0)] };
        }
        let neurons = self.neurons();
        let destinations: Vec<usize> = neurons
            .iter()
            .copied()
            .filter(|&id| id >= self.input_neurons)
            .collect();
        neurons
            .iter()
            .flat_map(|&from| {
                destinations
                    .iter()
                    .filter(move |&&to| to != from)
                    .map(move |&to| (from, to))
            })
            .filter(|&(from, to)| !exists(from, to))
            .collect()
    }

    fn mutate_connection_weight_with_config<R: Rng + ?Sized>(
        &mut self,
        config: &MutationConfig,
        rng: &mut R,
    ) {
        for gene in &mut self.genes {
            if rng.gen::<f64>() < config.weight_gene_mutation_rate {
                let weight = mutated_value(
                    gene.weight(),
                    config.weight_perturbation_rate,
                    config.weight_mutate_power,
                    config.weight_init_range,
                    rng,
                );
                gene.set_weight(weight.clamp(-config.weight_max, config.weight_max));
            }
        }
    }

    fn mutate_toggle_expression<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        if self.genes.is_empty() {
            return;
        }
        let selected_gene = rng.gen_range(0..self.genes.len());
        <dyn Mutation>::toggle_expression(&mut self.genes[selected_gene]);
    }

    /// Mutates each neuron bias with probability `bias_gene_mutation_rate`,
    /// except those of the inputs, which are sources without bias.
    fn mutate_biases_with_config<R: Rng + ?Sized>(&mut self, config: &MutationConfig, rng: &mut R) {
        self.biases.resize(self.len(), 0f64);
        for id in self.neurons() {
            if id < self.input_neurons {
                continue;
            }
            let bias = &mut self.biases[id];
            if rng.gen::<f64>() < config.bias_gene_mutation_rate {
                let value = mutated_value(
                    *bias,
                    config.bias_perturbation_rate,
                    config.bias_mutate_power,
                    config.weight_init_range,
                    rng,
                );
                *bias = value.clamp(-config.bias_max, config.bias_max);
            }
        }
    }

    /// Mutates each neuron τ with probability `tau_gene_mutation_rate`,
    /// except those of the inputs: `τ ← τ·e^N(0, σ²)`, clipped to
    /// `[tau_min, tau_max]`. A neuron without τ gene starts from the
    /// default `config.tau`.
    fn mutate_taus_with_config<R: Rng + ?Sized>(&mut self, config: &MutationConfig, rng: &mut R) {
        for id in self.neurons() {
            if id < self.input_neurons {
                continue;
            }
            if rng.gen::<f64>() < config.tau_gene_mutation_rate {
                let tau = self.tau(id).unwrap_or(config.tau);
                let factor = Gene::generate_gaussian_with_rng(config.tau_mutate_power, rng).exp();
                self.set_tau(id, (tau * factor).clamp(config.tau_min, config.tau_max));
            }
        }
    }

    #[cfg(test)]
    fn mutate_add_neuron<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        self.mutate_add_neuron_with_registry(&MutationConfig::default(), None, rng);
    }

    fn mutate_add_neuron_with_registry<R: Rng + ?Sized>(
        &mut self,
        config: &MutationConfig,
        innovations: Option<&mut InnovationRegistry>,
        rng: &mut R,
    ) {
        // Only an expressed connection is split: splitting a disabled one
        // would put back a path that evolution switched off.
        let enabled: Vec<usize> = (0..self.genes.len())
            .filter(|&index| self.genes[index].enabled())
            .collect();
        if enabled.is_empty() {
            return;
        }
        let selected_gene = enabled[rng.gen_range(0..enabled.len())];
        self.split_gene(selected_gene, config, innovations);
    }

    /// Split the gene at `index`: disable it and add a new neuron between its
    /// ends, nearly neutral with the default `config.split_input_weight` and
    /// `config.split_bias` (see SPLIT_INPUT_WEIGHT). The neuron id comes
    /// from `innovations` or, without registry, is the next one after the
    /// highest id of the genome. Returns the id.
    fn split_gene(
        &mut self,
        index: usize,
        config: &MutationConfig,
        innovations: Option<&mut InnovationRegistry>,
    ) -> usize {
        let (in_neuron_id, out_neuron_id) = (
            self.genes[index].in_neuron_id(),
            self.genes[index].out_neuron_id(),
        );
        let neuron_id = match innovations {
            Some(registry) => registry.split_neuron_id(self, in_neuron_id, out_neuron_id),
            None => self.last_neuron_id + 1,
        };
        let (gene1, gene2) = <dyn Mutation>::add_neuron(
            &mut self.genes[index],
            neuron_id,
            config.split_input_weight,
        );
        // The new neuron is born with the split bias (see SPLIT_BIAS), even
        // if its id held a bias before.
        self.set_bias(neuron_id, config.split_bias);
        // ... and with the default τ.
        self.store_tau(neuron_id, None);
        self.add_gene(gene1);
        self.add_gene(gene2);
        neuron_id
    }

    /// Add a new gene and checks if is allowd. Only can connect next neuron or already connected
    /// neurons.
    pub fn add_gene(&mut self, gene: Gene) {
        self.touch();
        let max_neuron_id = self.last_neuron_id + 1;

        if gene.in_neuron_id() == gene.out_neuron_id() && gene.in_neuron_id() > max_neuron_id {
            panic!(
                "Try to create a gene neuron unconnected, max neuron id {}, {} -> {}",
                max_neuron_id,
                gene.in_neuron_id(),
                gene.out_neuron_id()
            );
        }

        if gene.in_neuron_id() > self.last_neuron_id {
            self.last_neuron_id = gene.in_neuron_id();
        }
        if gene.out_neuron_id() > self.last_neuron_id {
            self.last_neuron_id = gene.out_neuron_id();
        }
        // The genes stay sorted by (in, out): insert in place.
        match self.genes.binary_search(&gene) {
            Ok(pos) => self.genes[pos].set_enabled(),
            Err(pos) => self.genes.insert(pos, gene),
        }
    }

    /// Compare another Genome for species equality
    // TODO This should be impl Eq
    pub fn is_same_specie(&self, other: &Genome) -> bool {
        self.is_same_specie_with_threshold(other, COMPATIBILITY_THRESHOLD)
    }

    /// Compare another Genome for species equality using a custom threshold
    /// and the default coefficients
    pub fn is_same_specie_with_threshold(&self, other: &Genome, threshold: f64) -> bool {
        self.compatibility_distance(other) < threshold
    }

    /// Compare another Genome for species equality with the coefficients and
    /// the threshold of `config`
    pub fn is_same_specie_with_config(&self, other: &Genome, config: &MutationConfig) -> bool {
        self.compatibility_distance_with_config(other, config) < config.compatibility_threshold
    }

    /// Total weigths of all genes
    pub fn total_weights(&self) -> f64 {
        let mut total = 0f64;
        for gene in &self.genes {
            total += gene.weight();
        }
        total
    }

    /// Total num genes
    // TODO len() is enough
    pub fn total_genes(&self) -> usize {
        self.genes.len()
    }

    /// Compatibility distance with the default coefficients
    pub fn compatibility_distance(&self, other: &Genome) -> f64 {
        self.compatibility_distance_with_config(other, &MutationConfig::default())
    }

    /// Compatibility distance with the coefficients of `config` (NEAT paper,
    /// Stanley & Miikkulainen 2002, eq. 1, plus a bias term):
    ///
    /// `δ = c1·E/N + c2·D/N + c3·W̄ + c_bias·B̄ + c_τ·T̄`
    ///
    /// Genes are aligned by `(in, out)`, the order they are kept in: with
    /// neuron ids from an `InnovationRegistry` the pair identifies the
    /// innovation. `E` counts the excess genes (beyond the last gene of the
    /// other genome in that order), `D` the disjoint ones (the rest of the
    /// unmatched genes), `W̄` is the average weight difference of the matching
    /// genes and `B̄` the average bias difference of the neurons both genomes
    /// have (same id), inputs excluded; `T̄` is the average `|ln τa - ln τb|`
    /// of those neurons (the default τ of `config` for a neuron without τ
    /// gene). Each average is 0 when there is nothing to average. `N = 1` if both genomes have fewer genes than
    /// `compatibility_small_genome_genes`, otherwise the number of genes of
    /// the larger genome.
    pub fn compatibility_distance_with_config(
        &self,
        other: &Genome,
        config: &MutationConfig,
    ) -> f64 {
        let (a, b) = (&self.genes, &other.genes);
        let (mut i, mut j) = (0, 0);
        let (mut disjoint, mut matching) = (0usize, 0usize);
        let mut weight_difference = 0f64;
        // Both gene lists are sorted: walk them together.
        while i < a.len() && j < b.len() {
            match a[i].cmp(&b[j]) {
                cmp::Ordering::Equal => {
                    weight_difference += (a[i].weight() - b[j].weight()).abs();
                    matching += 1;
                    i += 1;
                    j += 1;
                }
                cmp::Ordering::Less => {
                    disjoint += 1;
                    i += 1;
                }
                cmp::Ordering::Greater => {
                    disjoint += 1;
                    j += 1;
                }
            }
        }
        // Whatever is left in one genome lies beyond the last gene of the
        // other one.
        let excess = (a.len() - i) + (b.len() - j);

        let larger = cmp::max(a.len(), b.len());
        let small = config.compatibility_small_genome_genes;
        let n = if larger < small {
            1f64
        } else {
            larger.max(1) as f64
        };

        let average_weight_difference = if matching > 0 {
            weight_difference / matching as f64
        } else {
            0f64
        };

        // Neurons both genomes have (same id, same innovation), inputs
        // excluded.
        let first = cmp::max(self.input_neurons, other.input_neurons);
        let other_neurons = other.neurons();
        let common: Vec<usize> = self
            .neurons()
            .into_iter()
            .filter(|&id| id >= first && other_neurons.binary_search(&id).is_ok())
            .collect();
        let average_bias_difference = if common.is_empty() {
            0f64
        } else {
            common
                .iter()
                .map(|&id| (self.bias(id) - other.bias(id)).abs())
                .sum::<f64>()
                / common.len() as f64
        };

        // τ acts through dt/τ, so its differences count in log scale.
        let average_tau_difference = if common.is_empty() {
            0f64
        } else {
            let log_tau = |genome: &Genome, id| genome.tau(id).unwrap_or(config.tau).ln();
            common
                .iter()
                .map(|&id| (log_tau(self, id) - log_tau(other, id)).abs())
                .sum::<f64>()
                / common.len() as f64
        };

        config.compatibility_excess_coefficient * excess as f64 / n
            + config.compatibility_disjoint_coefficient * disjoint as f64 / n
            + config.compatibility_weight_coefficient * average_weight_difference
            + config.compatibility_bias_coefficient * average_bias_difference
            + config.compatibility_tau_coefficient * average_tau_difference
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::f64::EPSILON;

    fn rng() -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(0)
    }

    impl Genome {
        fn mutate_add_connection(&mut self) {
            self.mutate_add_connection_with_config(&MutationConfig::default(), &mut rng());
        }

        /// Mutates every weight (default config except the per-gene rate).
        fn mutate_connection_weight(&mut self) {
            let config = MutationConfig::new()
                .weight_gene_mutation_rate(1f64)
                .build();
            self.mutate_connection_weight_with_config(&config, &mut rng());
        }

        fn add_connection(&mut self, in_neuron_id: usize, out_neuron_id: usize) {
            let gene = <dyn Mutation>::add_connection(in_neuron_id, out_neuron_id, &mut rng());
            self.add_gene(gene);
        }
    }

    fn connections(genome: &Genome) -> Vec<(usize, usize)> {
        genome
            .get_genes()
            .iter()
            .map(|g| (g.in_neuron_id(), g.out_neuron_id()))
            .collect()
    }

    #[test]
    fn default_structural_rates_are_the_neat_paper_ones() {
        let config = MutationConfig::default();
        assert_eq!(config.add_connection_rate, 0.05);
        assert_eq!(config.add_neuron_rate, 0.03);
    }

    #[test]
    fn default_weight_and_bias_mutation_is_per_gene_gaussian_and_bounded() {
        let config = MutationConfig::default();
        assert_eq!(config.weight_gene_mutation_rate, 0.5);
        assert_eq!(config.weight_mutate_power, 1.0);
        assert_eq!(config.weight_perturbation_rate, 0.98);
        assert_eq!(config.weight_max, 30.0);
        assert_eq!(config.bias_gene_mutation_rate, 0.5);
        assert_eq!(config.bias_mutate_power, 0.5);
        assert_eq!(config.bias_perturbation_rate, 0.98);
        assert_eq!(config.bias_max, 30.0);
    }

    #[test]
    fn genome_knows_its_inputs_and_outputs() {
        let genome = Genome::new_initialized(2, 3);
        assert_eq!((genome.input_neurons(), genome.output_neurons()), (2, 3));
        assert_eq!(genome.len(), 5);
        let genome = Genome::new_unconnected(3, 1);
        assert_eq!((genome.input_neurons(), genome.output_neurons()), (3, 1));
        assert_eq!(genome.len(), 4);
        let genome = Genome::default();
        assert_eq!((genome.input_neurons(), genome.output_neurons()), (0, 0));
    }

    #[test]
    fn add_connection_never_ends_in_an_input() {
        let mut rng = rng();
        for _ in 0..200 {
            let mut genome = Genome::new_unconnected(3, 2);
            for _ in 0..30 {
                genome.mutate_add_connection_with_config(&MutationConfig::default(), &mut rng);
                if rng.gen::<f64>() < 0.2 {
                    genome.mutate_add_neuron(&mut rng);
                }
            }
            for (from, to) in connections(&genome) {
                assert!(to >= 3, "connection {} -> {} ends in an input", from, to);
            }
        }
    }

    #[test]
    fn add_connection_retries_until_a_new_connection_is_found() {
        // Neurons 0 (input), 1 and 2 (outputs); 0->1 and 0->2 exist, so the
        // only new connections are 1->2 and 2->1.
        for seed in 0..50 {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let mut genome = Genome::new_initialized(1, 2);
            genome.mutate_add_connection_with_config(&MutationConfig::default(), &mut rng);
            assert_eq!(genome.total_genes(), 3, "seed {}", seed);
            genome.mutate_add_connection_with_config(&MutationConfig::default(), &mut rng);
            assert_eq!(
                connections(&genome),
                vec![(0, 1), (0, 2), (1, 2), (2, 1)],
                "seed {}",
                seed
            );
            // Nothing left to connect: no-op, no panic.
            let before = connections(&genome);
            genome.mutate_add_connection_with_config(&MutationConfig::default(), &mut rng);
            assert_eq!(connections(&genome), before);
        }
    }

    #[test]
    fn add_connection_picks_pairs_uniformly() {
        // Neurons 0 (input), 1 and 2 (outputs), no connections: the possible
        // ones are 0->1, 0->2, 1->2, 2->1, each with probability 1/4.
        let mut rng = rng();
        let mut seen = std::collections::BTreeMap::new();
        for _ in 0..4000 {
            let mut genome = Genome::new_unconnected(1, 2);
            genome.mutate_add_connection_with_config(&MutationConfig::default(), &mut rng);
            *seen.entry(connections(&genome)).or_insert(0) += 1;
        }
        assert_eq!(seen.len(), 4, "{:?}", seen);
        for &count in seen.values() {
            assert!(count > 850 && count < 1150, "{:?}", seen);
        }
    }

    #[test]
    fn bias_mutation_leaves_inputs_untouched() {
        let mut genome = Genome::new_initialized(2, 1);
        let mut rng = rng();
        for _ in 0..20 {
            genome.mutate_with_config_and_rng(&only_bias_mutation(1f64), &mut rng);
        }
        assert_eq!(genome.bias(0), 0f64);
        assert_eq!(genome.bias(1), 0f64);
        assert!(genome.bias(2) != 0f64);
    }

    /// Two genomes with inputs 0, 1 and outputs 2, 3, where only output 2
    /// is connected.
    fn io_parents() -> (Genome, Genome) {
        let mut a = Genome::new_unconnected(2, 2);
        a.add_gene(Gene::new(0, 2, 1f64, true));
        let mut b = Genome::new_unconnected(2, 2);
        b.add_gene(Gene::new(0, 2, -1f64, true));
        b.add_gene(Gene::new(1, 2, 0.5f64, true));
        (a, b)
    }

    #[test]
    fn crossover_keeps_unconnected_inputs_and_outputs() {
        let (a, b) = io_parents();
        let mut rng = rng();
        for _ in 0..50 {
            for mut child in [
                a.mate_with_rng(&b, true, &mut rng),
                a.mate_with_rng(&b, false, &mut rng),
                a.mate_equal_with_rng(&b, &mut rng),
            ] {
                assert_eq!((child.input_neurons(), child.output_neurons()), (2, 2));
                assert_eq!(child.len(), 4);
                // A new neuron never reuses the id of the unconnected output.
                child.mutate_add_neuron(&mut rng);
                assert_eq!(child.len(), 5);
                assert!(connections(&child).iter().any(|&(f, t)| f == 4 || t == 4));
            }
        }
    }

    #[test]
    fn mutation_connection_weight() {
        let mut genome = Genome::default();
        genome.add_gene(Gene::new(0, 0, 1f64, true));
        let orig_gene = genome.genes[0];
        genome.mutate_connection_weight();
        // These should not be same size
        assert!((genome.genes[0].weight() - orig_gene.weight()).abs() > EPSILON);
    }

    #[test]
    fn mutation_add_connection() {
        let mut genome = Genome::default();
        genome.add_connection(1, 2);

        assert!(genome.genes[0].in_neuron_id() == 1);
        assert!(genome.genes[0].out_neuron_id() == 2);
    }

    #[test]
    fn mutation_add_neuron() {
        let mut genome = Genome::default();
        genome.mutate_add_connection();
        genome.mutate_add_neuron(&mut rng());
        assert!(!genome.genes[0].enabled());
        assert!(genome.genes[1].in_neuron_id() == genome.genes[0].in_neuron_id());
        assert!(genome.genes[1].out_neuron_id() == 1);
        assert!(genome.genes[2].in_neuron_id() == 1);
        assert!(genome.genes[2].out_neuron_id() == genome.genes[0].out_neuron_id());
    }

    #[test]
    #[should_panic(expected = "Try to create a gene neuron unconnected, max neuron id 1, 2 -> 2")]
    fn try_to_inject_a_unconnected_neuron_gene_should_panic() {
        let mut genome1 = Genome::default();
        genome1.add_gene(Gene::new(2, 2, 0.5f64, true));
    }

    #[test]
    fn two_genomes_without_differences_should_be_in_same_specie() {
        let mut genome1 = Genome::default();
        genome1.add_gene(Gene::new(0, 0, 1f64, true));
        genome1.add_gene(Gene::new(0, 1, 1f64, true));
        let mut genome2 = Genome::default();
        genome2.add_gene(Gene::new(0, 0, 0f64, true));
        genome2.add_gene(Gene::new(0, 1, 0f64, true));
        genome2.add_gene(Gene::new(0, 2, 0f64, true));
        assert!(genome1.is_same_specie(&genome2));
    }

    #[test]
    fn two_genomes_with_enought_difference_should_be_in_different_species() {
        let mut genome1 = Genome::default();
        genome1.add_gene(Gene::new(0, 0, 1f64, true));
        genome1.add_gene(Gene::new(0, 1, 1f64, true));
        let mut genome2 = Genome::default();
        genome2.add_gene(Gene::new(0, 0, 20f64, true));
        genome2.add_gene(Gene::new(0, 1, 20f64, true));
        genome2.add_gene(Gene::new(0, 2, 1f64, true));
        genome2.add_gene(Gene::new(0, 3, 1f64, true));
        assert!(!genome1.is_same_specie(&genome2));
    }

    #[test]
    fn already_existing_gene_should_be_not_duplicated() {
        let mut genome1 = Genome::default();
        genome1.add_gene(Gene::new(0, 0, 1f64, true));
        genome1.add_connection(0, 0);
        assert_eq!(genome1.genes.len(), 1);
        assert!((genome1.get_genes()[0].weight() - 1f64).abs() < EPSILON);
    }

    #[test]
    fn adding_an_existing_gene_disabled_should_enable_original() {
        let mut genome1 = Genome::default();
        genome1.add_gene(Gene::new(0, 1, 0f64, true));
        genome1.mutate_add_neuron(&mut rng());
        assert!(!genome1.genes[0].enabled());
        assert!(genome1.genes.len() == 3);
        genome1.add_connection(0, 1);
        assert!(genome1.genes[0].enabled());
        assert!((genome1.genes[0].weight() - 0f64).abs() < EPSILON);
        assert_eq!(genome1.genes.len(), 3);
    }

    #[test]
    fn genomes_with_same_genes_with_little_differences_on_weight_should_be_in_same_specie() {
        let mut genome1 = Genome::default();
        genome1.add_gene(Gene::new(0, 0, 16f64, true));
        let mut genome2 = Genome::default();
        genome2.add_gene(Gene::new(0, 0, 16.1f64, true));
        assert!(genome1.is_same_specie(&genome2));
    }

    #[test]
    fn genomes_with_same_genes_with_big_differences_on_weight_should_be_in_other_specie() {
        let mut genome1 = Genome::default();
        genome1.add_gene(Gene::new(0, 0, 5f64, true));
        let mut genome2 = Genome::default();
        genome2.add_gene(Gene::new(0, 0, 25f64, true));
        assert!(!genome1.is_same_specie(&genome2));
    }

    /// Genome with neurons `0..inputs` as inputs and one connection per
    /// pair in `connections`, all with weight `weight`.
    fn genome_with(inputs: usize, connections: &[(usize, usize)], weight: f64) -> Genome {
        let mut genome = Genome::new_unconnected(inputs, 1);
        for &(from, to) in connections {
            genome.add_gene(Gene::new(from, to, weight, true));
        }
        genome
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-9,
            "distance {} != {}",
            actual,
            expected
        );
    }

    #[test]
    fn genomes_without_common_genes_are_in_different_species() {
        // a: connections 0 -> k; b: connections 1 -> k. No gene in common.
        // With the default N = 1 the distance is at least c1 + c2 = 2.
        for size_a in 1..40 {
            for size_b in 1..40 {
                let a_conn: Vec<(usize, usize)> = (0..size_a).map(|k| (0, 2 + k)).collect();
                let b_conn: Vec<(usize, usize)> = (0..size_b).map(|k| (1, 2 + k)).collect();
                let a = genome_with(2, &a_conn, 1f64);
                let b = genome_with(2, &b_conn, 1f64);
                assert!(
                    !a.is_same_specie(&b),
                    "{} vs {} genes: distance {}",
                    size_a,
                    size_b,
                    a.compatibility_distance(&b)
                );
            }
        }
    }

    #[test]
    fn compatibility_distance_of_disjoint_genomes_has_no_weight_term() {
        // One gene each, none in common: one disjoint and one excess gene,
        // W = 0 (no special value when nothing matches), N = 1.
        let a = genome_with(2, &[(0, 2)], 5f64);
        let b = genome_with(2, &[(1, 2)], -5f64);
        assert_close(a.compatibility_distance(&b), 2f64);
    }

    #[test]
    fn compatibility_distance_follows_the_paper_formula() {
        let config = MutationConfig::new()
            .compatibility_coefficients(2f64, 3f64, 0.5f64, 0f64)
            .build();
        // Genes aligned by (in, out), the order they are kept in.
        // a: 0->3, 0->4, 1->3 ; b: 0->3, 0->5, 1->4, 2->3.
        // Matching: 0->3 (|1 - 3| = 2). Disjoint (within the range of the
        // other genome): a's 0->4, 1->3 and b's 0->5. Excess (beyond a's
        // last gene 1->3): b's 1->4, 2->3.
        let a = genome_with(3, &[(0, 3), (0, 4), (1, 3)], 1f64);
        let mut b = genome_with(3, &[(0, 5), (1, 4), (2, 3)], 0f64);
        b.add_gene(Gene::new(0, 3, 3f64, true));
        // E = 2, D = 3, W = 2, N = 1
        let expected = 2f64 * 2f64 + 3f64 * 3f64 + 0.5f64 * 2f64;
        assert_close(a.compatibility_distance_with_config(&b, &config), expected);
        assert_close(b.compatibility_distance_with_config(&a, &config), expected);
    }

    #[test]
    fn compatibility_distance_normalizes_large_genomes() {
        // The paper criterion: N = 1 if both genomes have fewer than 20 genes.
        let config = MutationConfig::new()
            .compatibility_coefficients(1f64, 1f64, 0f64, 0f64)
            .compatibility_small_genome_genes(20)
            .build();
        // a: 25 genes 0 -> k; b: the same 25 genes plus 5 genes 1 -> k.
        let common: Vec<(usize, usize)> = (0..25).map(|k| (0, 2 + k)).collect();
        let mut more = common.clone();
        more.extend((0..5).map(|k| (1, 2 + k)));
        let a = genome_with(2, &common, 1f64);
        let b = genome_with(2, &more, 1f64);
        // E = 5, N = 30
        assert_close(
            a.compatibility_distance_with_config(&b, &config),
            5f64 / 30f64,
        );
        // N criterion disabled for small genomes: N = max always.
        let small_a = genome_with(2, &[(0, 2)], 1f64);
        let small_b = genome_with(2, &[(0, 2), (1, 2)], 1f64);
        assert_close(
            small_a.compatibility_distance_with_config(&small_b, &config),
            1f64,
        );
        let always = MutationConfig::new()
            .compatibility_coefficients(1f64, 1f64, 0f64, 0f64)
            .compatibility_small_genome_genes(0)
            .build();
        assert_close(
            small_a.compatibility_distance_with_config(&small_b, &always),
            0.5f64,
        );
        // Default: never normalized (N = 1).
        let default = MutationConfig::new()
            .compatibility_coefficients(1f64, 1f64, 0f64, 0f64)
            .build();
        assert_close(a.compatibility_distance_with_config(&b, &default), 5f64);
    }

    #[test]
    fn compatibility_distance_counts_bias_differences_of_common_neurons() {
        let config = MutationConfig::new()
            .compatibility_coefficients(1f64, 1f64, 0.4f64, 0.5f64)
            .build();
        // Same genes and weights; neurons 0, 1 inputs, 2 output, 3 hidden.
        let mut a = genome_with(2, &[(0, 2), (1, 3), (3, 2)], 1f64);
        let mut b = a.clone();
        assert_close(a.compatibility_distance_with_config(&b, &config), 0f64);
        a.set_bias(2, 1f64);
        b.set_bias(2, -1f64);
        a.set_bias(3, 0.5f64);
        // Input biases never count.
        a.set_bias(0, 100f64);
        // B = (|1 - (-1)| + |0.5 - 0|) / 2 = 1.25
        assert_close(
            a.compatibility_distance_with_config(&b, &config),
            0.5f64 * 1.25f64,
        );
    }

    #[test]
    fn genomes_initialized_has_correct_neurons() {
        let genome1 = Genome::new_initialized(2, 3);
        assert_eq!(genome1.total_genes(), 6);
        assert_eq!(genome1.genes[0].in_neuron_id(), 0);
        assert_eq!(genome1.genes[0].out_neuron_id(), 2);
        assert_eq!(genome1.genes[1].in_neuron_id(), 0);
        assert_eq!(genome1.genes[1].out_neuron_id(), 3);
        assert_eq!(genome1.genes[2].in_neuron_id(), 0);
        assert_eq!(genome1.genes[2].out_neuron_id(), 4);
        assert_eq!(genome1.genes[3].in_neuron_id(), 1);
        assert_eq!(genome1.genes[3].out_neuron_id(), 2);
        assert_eq!(genome1.genes[4].in_neuron_id(), 1);
        assert_eq!(genome1.genes[4].out_neuron_id(), 3);
        assert_eq!(genome1.genes[5].in_neuron_id(), 1);
        assert_eq!(genome1.genes[5].out_neuron_id(), 4);
    }

    fn only_bias_mutation(rate: f64) -> MutationConfig {
        MutationConfig::new()
            .add_connection_rate(0f64)
            .add_neuron_rate(0f64)
            .weight_mutation_rate(0f64)
            .toggle_expression_rate(0f64)
            .bias_mutation_rate(rate)
            .bias_gene_mutation_rate(1f64)
            .bias_perturbation_rate(1f64)
            .build()
    }

    #[test]
    fn neuron_without_bias_gene_has_neutral_bias() {
        let mut genome = Genome::default();
        genome.add_gene(Gene::new(0, 2, 1f64, true));
        for id in 0..10 {
            assert_eq!(genome.bias(id), 0f64);
        }
        genome.set_bias(1, -2.5);
        assert_eq!(genome.bias(1), -2.5);
        assert_eq!(genome.bias(2), 0f64);
    }

    #[test]
    fn bias_mutation_with_gene_rate_one_perturbs_every_bias_and_can_go_negative() {
        let mut genome = Genome::default();
        genome.add_gene(Gene::new(0, 1, 1f64, true));
        genome.add_gene(Gene::new(0, 2, 1f64, true));
        let config = only_bias_mutation(1f64);
        let mut rng = rng();
        let mut seen_negative = false;
        for _ in 0..20 {
            let before: Vec<f64> = (0..genome.len()).map(|id| genome.bias(id)).collect();
            genome.mutate_with_config_and_rng(&config, &mut rng);
            for (id, old) in before.iter().enumerate() {
                let new = genome.bias(id);
                assert!(new != *old, "bias of neuron {} not perturbed", id);
                seen_negative |= new < 0f64;
            }
        }
        assert!(seen_negative);
    }

    #[test]
    fn bias_mutation_rate_zero_leaves_biases_untouched() {
        let mut genome = Genome::default();
        genome.add_gene(Gene::new(0, 1, 1f64, true));
        genome.set_bias(1, 0.7);
        let mut rng = rng();
        for _ in 0..50 {
            genome.mutate_with_config_and_rng(&only_bias_mutation(0f64), &mut rng);
        }
        assert_eq!(genome.bias(0), 0f64);
        assert_eq!(genome.bias(1), 0.7);
    }

    #[test]
    fn bias_mutation_is_reproducible_with_same_seed() {
        let run = |seed| {
            let mut genome = Genome::default();
            genome.add_gene(Gene::new(0, 1, 1f64, true));
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            for _ in 0..10 {
                genome.mutate_with_config_and_rng(&only_bias_mutation(1f64), &mut rng);
            }
            (genome.bias(0), genome.bias(1))
        };
        assert_eq!(run(3), run(3));
        assert!(run(3) != run(4));
    }

    #[test]
    fn new_neuron_is_born_with_the_split_bias() {
        let mut genome = Genome::default();
        genome.add_gene(Gene::new(0, 1, 1f64, true));
        genome.set_bias(0, 3f64);
        genome.set_bias(1, -3f64);
        // A stale bias stored for the id the new neuron will take.
        genome.set_bias(2, 5f64);
        genome.mutate_add_neuron(&mut rng());
        assert_eq!(genome.len(), 3);
        assert_eq!(genome.bias(2), SPLIT_BIAS);
        assert_eq!(genome.bias(0), 3f64);
        assert_eq!(genome.bias(1), -3f64);
    }

    #[test]
    fn split_neuron_is_close_to_the_identity_on_the_unit_interval() {
        // in -> new has weight SPLIT_INPUT_WEIGHT, the new neuron has bias
        // SPLIT_BIAS and new -> out keeps the old weight.
        let mut genome = Genome::new_initialized(2, 1);
        genome.genes[0].set_weight(-3.0);
        let id = split(&mut genome, 0, 2, None);
        let weight = |from, to| {
            let pos = genome
                .genes
                .binary_search(&Gene::new_connection(from, to))
                .unwrap();
            genome.genes[pos].weight()
        };
        assert_eq!(weight(0, id), SPLIT_INPUT_WEIGHT);
        assert_eq!(weight(id, 2), -3.0);
        assert_eq!(genome.bias(id), SPLIT_BIAS);
        // σ(a·s + θ) stays within 0.1 of s for s in [0, 1].
        for k in 0..=100 {
            let s = k as f64 / 100.0;
            let relayed = crate::Ctrnn::sigmoid(SPLIT_INPUT_WEIGHT * s + SPLIT_BIAS);
            assert!((relayed - s).abs() < 0.1, "s {}: {}", s, relayed);
        }
    }

    #[test]
    fn splitting_a_connection_barely_changes_the_output() {
        use crate::Organism;
        let output = |genome: &Genome, a: f64, b: f64| {
            let mut organism = Organism::new(genome.clone());
            let mut out = vec![0f64];
            organism.activate(vec![a, b], &mut out);
            out[0]
        };
        // Inputs 0, 1, hidden 3, output 2: split an input connection and a
        // connection between CTRNN neurons.
        let mut genome = Genome::new_initialized(2, 1);
        genome.genes[0].set_weight(3.0);
        genome.genes[1].set_weight(-2.0);
        split(&mut genome, 1, 2, None);
        genome.set_bias(3, 0.5);
        let index = genome
            .genes
            .binary_search(&Gene::new_connection(3, 2))
            .unwrap();
        genome.genes[index].set_weight(4.0);
        for (from, to) in [(0, 2), (3, 2)] {
            let mut child = genome.clone();
            split(&mut child, from, to, None);
            for (a, b) in [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)] {
                let (before, after) = (output(&genome, a, b), output(&child, a, b));
                assert!(
                    (before - after).abs() < 0.05,
                    "split {} -> {}, input ({}, {}): {} before, {} after",
                    from,
                    to,
                    a,
                    b,
                    before,
                    after
                );
            }
        }
    }

    /// Split the connection `from -> to` of `genome` (it must exist).
    fn split(
        genome: &mut Genome,
        from: usize,
        to: usize,
        innovations: Option<&mut InnovationRegistry>,
    ) -> usize {
        let index = genome
            .genes
            .binary_search(&Gene::new_connection(from, to))
            .unwrap();
        genome.split_gene(index, &MutationConfig::default(), innovations)
    }

    #[test]
    fn splitting_the_same_connection_twice_gives_two_neurons_shared_by_lineages() {
        let mut registry = InnovationRegistry::new(3);
        let mut a = Genome::new_initialized(2, 1);
        let mut b = a.clone();
        let first = split(&mut a, 0, 2, Some(&mut registry));
        // 0 -> 2 is now disabled; splitting it again must not collide.
        let second = split(&mut a, 0, 2, Some(&mut registry));
        assert!(first != second);
        assert!(a.has_neuron(first) && a.has_neuron(second));
        for gene in [(0, first), (first, 2), (0, second), (second, 2)] {
            assert!(connections(&a).contains(&gene), "{:?} missing", gene);
        }
        // Another lineage doing the same two splits gets the same two ids.
        assert_eq!(split(&mut b, 0, 2, Some(&mut registry)), first);
        assert_eq!(split(&mut b, 0, 2, Some(&mut registry)), second);
        assert_eq!(connections(&a), connections(&b));
        assert_eq!(registry.split_of(second), Some((0, 2)));
        assert_eq!(registry.len(), 2);
    }

    #[test]
    fn splitting_without_registry_numbers_after_the_highest_id() {
        let mut genome = Genome::new_initialized(2, 1);
        let mut registry = InnovationRegistry::new(3);
        // Another genome takes id 3, so this one gets 4 and has no neuron 3.
        let mut other = Genome::new_initialized(2, 1);
        split(&mut other, 1, 2, Some(&mut registry));
        let far = split(&mut genome, 0, 2, Some(&mut registry));
        assert_eq!(far, 4);
        assert_eq!(split(&mut genome, 1, 2, None), 5);
        assert_eq!(split(&mut genome, 0, far, None), 6);
        assert_eq!(genome.neurons(), vec![0, 1, 2, 4, 5, 6]);
    }

    #[test]
    fn registry_never_gives_an_id_the_genome_already_has() {
        let mut genome = Genome::new_initialized(2, 1);
        // Neurons 3 and 4 numbered without registry.
        split(&mut genome, 0, 2, None);
        split(&mut genome, 1, 2, None);
        let mut registry = InnovationRegistry::new(3);
        let id = split(&mut genome, 0, 3, Some(&mut registry));
        assert_eq!(id, 5);
    }

    /// A: neurons 0, 1; B: neurons 0, 1, 2. Different biases on 1, B has
    /// its own bias on 2.
    fn bias_parents() -> (Genome, Genome) {
        let mut a = Genome::default();
        a.add_gene(Gene::new(0, 1, 1f64, true));
        a.set_bias(1, -1f64);
        let mut b = Genome::default();
        b.add_gene(Gene::new(0, 1, 1f64, true));
        b.add_gene(Gene::new(0, 2, 1f64, true));
        b.add_gene(Gene::new(2, 1, 1f64, true));
        b.set_bias(1, 2f64);
        b.set_bias(2, -4f64);
        (a, b)
    }

    #[test]
    fn crossover_inherits_biases_of_matching_neurons_from_both_parents() {
        let (a, b) = bias_parents();
        let mut rng = rng();
        let mut from = std::collections::BTreeMap::new();
        for (fitter, child) in (0..1000).map(|i| {
            if i % 2 == 0 {
                ("a", a.mate_with_rng(&b, true, &mut rng))
            } else {
                ("b", a.mate_with_rng(&b, false, &mut rng))
            }
        }) {
            let bias = child.bias(1);
            assert!(
                bias == -1f64 || bias == 2f64 || bias == 0.5f64,
                "bias {} not inherited",
                bias
            );
            *from.entry((fitter, bias.to_string())).or_insert(0) += 1;
        }
        for fitter in ["a", "b"] {
            for bias in ["-1", "2"] {
                assert!(
                    from.get(&(fitter, bias.to_string())).copied().unwrap_or(0) > 100,
                    "{:?}",
                    from
                );
            }
        }
    }

    #[test]
    fn crossover_inherits_bias_of_neuron_only_in_one_parent() {
        let (a, b) = bias_parents();
        let mut rng = rng();
        for _ in 0..200 {
            assert_eq!(a.mate_with_rng(&b, false, &mut rng).bias(2), -4f64);
            let child = a.mate_equal_with_rng(&b, &mut rng);
            if child.has_neuron(2) {
                assert_eq!(child.bias(2), -4f64);
            }
        }
    }

    /// Fraction of `trials` children of `a` (fitter) and `b` whose gene
    /// `0 -> 1` is disabled.
    fn disabled_ratio(a: &Genome, b: &Genome, config: &MutationConfig, equal: bool) -> f64 {
        let mut rng = rng();
        let trials = 2000;
        let disabled = (0..trials)
            .filter(|_| {
                let child = if equal {
                    a.mate_equal_with_config_and_rng(b, config, &mut rng)
                } else {
                    a.mate_with_config_and_rng(b, true, config, &mut rng)
                };
                let gene = child
                    .genes
                    .binary_search(&Gene::new_connection(0, 1))
                    .map(|pos| child.genes[pos]);
                matches!(gene, Ok(g) if !g.enabled())
            })
            .count();
        disabled as f64 / trials as f64
    }

    #[test]
    fn default_disabled_gene_inheritance_is_the_neat_paper_one() {
        assert_eq!(MutationConfig::default().disabled_gene_inheritance, 0.75);
    }

    #[test]
    fn gene_disabled_in_either_parent_is_inherited_disabled_with_configured_probability() {
        let mut disabled = Genome::default();
        disabled.add_gene(Gene::new(0, 1, 1.0, false));
        let mut enabled = Genome::default();
        enabled.add_gene(Gene::new(0, 1, 1.0, true));
        let default = MutationConfig::default();
        // Disabled in the fitter parent, in the other one, or in both.
        for (a, b) in [
            (&disabled, &enabled),
            (&enabled, &disabled),
            (&disabled, &disabled),
        ] {
            let ratio = disabled_ratio(a, b, &default, false);
            assert!(ratio > 0.71 && ratio < 0.79, "default: {}", ratio);
            for p in [0.0, 0.3, 1.0] {
                let config = MutationConfig::new().disabled_gene_inheritance(p).build();
                let ratio = disabled_ratio(a, b, &config, false);
                assert!((ratio - p).abs() < 0.04, "p {}: {}", p, ratio);
            }
        }
        // Enabled in both parents: always enabled.
        assert_eq!(disabled_ratio(&enabled, &enabled, &default, false), 0.0);
        // The old entry points use the default probability.
        let mut rng = rng();
        let old = (0..2000)
            .filter(|_| !disabled.mate_with_rng(&enabled, true, &mut rng).genes[0].enabled())
            .count() as f64
            / 2000.0;
        assert!(old > 0.71 && old < 0.79, "mate_with_rng: {}", old);
    }

    #[test]
    fn disabled_disjoint_gene_of_equal_parents_follows_configured_probability() {
        // 0 -> 1 is only in `a` (disabled) and is inherited half of the
        // time; when it is, it is disabled with probability p.
        let mut a = Genome::default();
        a.add_gene(Gene::new(0, 1, 1.0, false));
        a.add_gene(Gene::new(0, 2, 1.0, true));
        let mut b = Genome::default();
        b.add_gene(Gene::new(0, 2, 1.0, true));
        for p in [0.0, 0.75, 1.0] {
            let config = MutationConfig::new().disabled_gene_inheritance(p).build();
            let ratio = disabled_ratio(&a, &b, &config, true);
            assert!(
                (ratio - p / 2.0).abs() < 0.04,
                "a first, p {}: {}",
                p,
                ratio
            );
            let ratio = disabled_ratio(&b, &a, &config, true);
            assert!(
                (ratio - p / 2.0).abs() < 0.04,
                "b first, p {}: {}",
                p,
                ratio
            );
        }
    }

    #[test]
    fn add_neuron_only_splits_enabled_genes() {
        // Inputs 0, 1, output 2: 0 -> 2 is disabled, 1 -> 2 is enabled.
        let mut genome = Genome::new_initialized(2, 1);
        genome.genes[0].set_disabled();
        let mut rng = rng();
        for _ in 0..200 {
            let mut child = genome.clone();
            child.mutate_add_neuron(&mut rng);
            assert_eq!(
                connections(&child),
                vec![(0, 2), (1, 2), (1, 3), (3, 2)],
                "a disabled gene was split"
            );
            assert!(!child.genes[0].enabled(), "0 -> 2 was re-enabled");
        }
        // With every gene disabled there is nothing to split.
        genome.genes[1].set_disabled();
        let mut child = genome.clone();
        child.mutate_add_neuron(&mut rng);
        assert_eq!(connections(&child), connections(&genome));
        assert!(child.genes.iter().all(|gene| !gene.enabled()));
        assert_eq!(child.len(), genome.len());
    }

    #[test]
    fn add_connection_never_reenables_a_disabled_gene() {
        // Inputs 0, 1, output 2: 0 -> 2 disabled. The only new connections
        // would end in an input or be 2 -> 2, so there is nothing to add.
        let mut genome = Genome::new_initialized(2, 1);
        genome.genes[0].set_disabled();
        let mut rng = rng();
        for _ in 0..100 {
            genome.mutate_add_connection_with_config(&MutationConfig::default(), &mut rng);
        }
        assert_eq!(connections(&genome), vec![(0, 2), (1, 2)]);
        assert!(!genome.genes[0].enabled());
        // With a hidden neuron there are new connections: they are added
        // and the disabled gene stays disabled.
        let mut genome = Genome::new_initialized(2, 1);
        split(&mut genome, 0, 2, None);
        for _ in 0..20 {
            genome.mutate_add_connection_with_config(&MutationConfig::default(), &mut rng);
            let old = genome
                .genes
                .binary_search(&Gene::new_connection(0, 2))
                .unwrap();
            assert!(!genome.genes[old].enabled(), "0 -> 2 was re-enabled");
        }
        assert_eq!(genome.total_genes(), 6);
    }

    fn only_weight_mutation(gene_rate: f64) -> MutationConfig {
        MutationConfig::new()
            .add_connection_rate(0f64)
            .add_neuron_rate(0f64)
            .weight_mutation_rate(1f64)
            .weight_gene_mutation_rate(gene_rate)
            .toggle_expression_rate(0f64)
            .bias_mutation_rate(0f64)
            .build()
    }

    #[test]
    fn weight_mutation_changes_the_configured_fraction_of_genes() {
        let mut rng = rng();
        for &rate in &[0.1, 0.5, 0.8] {
            // 40 inputs x 25 outputs = 1000 connection genes.
            let mut genome = Genome::new_initialized_with_rng(40, 25, &mut rng);
            let before: Vec<f64> = genome.genes.iter().map(|g| g.weight()).collect();
            genome.mutate_with_config_and_rng(&only_weight_mutation(rate), &mut rng);
            let changed = genome
                .genes
                .iter()
                .zip(before.iter())
                .filter(|(g, w)| g.weight() != **w)
                .count();
            let fraction = changed as f64 / before.len() as f64;
            assert!(
                (fraction - rate).abs() <= 0.05,
                "rate {}: {} of the weights changed",
                rate,
                fraction
            );
        }
    }

    #[test]
    fn weights_never_exceed_weight_max() {
        let config = MutationConfig::new()
            .add_connection_rate(0.5)
            .add_neuron_rate(0.2)
            .weight_mutation_rate(1f64)
            .weight_gene_mutation_rate(1f64)
            .weight_mutate_power(5f64)
            .weight_init_range(10f64)
            .weight_perturbation_rate(0.8)
            .weight_max(3f64)
            .build();
        let mut rng = rng();
        let mut genome = Genome::new_initialized_with_rng(3, 2, &mut rng);
        let mut seen_at_limit = false;
        for _ in 0..500 {
            genome.mutate_with_config_and_rng(&config, &mut rng);
            for gene in &genome.genes {
                assert!(gene.weight().abs() <= 3f64, "weight {}", gene.weight());
                seen_at_limit |= gene.weight().abs() == 3f64;
            }
        }
        assert!(seen_at_limit);
    }

    #[test]
    fn weight_perturbation_is_gaussian_with_the_configured_deviation() {
        let config = MutationConfig::new()
            .weight_gene_mutation_rate(1f64)
            .weight_perturbation_rate(1f64)
            .weight_mutate_power(0.5)
            .build();
        let mut rng = rng();
        let mut genome = Genome::new_initialized_with_rng(40, 25, &mut rng);
        let before: Vec<f64> = genome.genes.iter().map(|g| g.weight()).collect();
        genome.mutate_connection_weight_with_config(&config, &mut rng);
        let deltas: Vec<f64> = genome
            .genes
            .iter()
            .zip(before.iter())
            .map(|(g, w)| g.weight() - w)
            .collect();
        let n = deltas.len() as f64;
        let mean = deltas.iter().sum::<f64>() / n;
        let sd = (deltas.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / n).sqrt();
        assert!(mean.abs() < 0.05, "mean {}", mean);
        assert!((sd - 0.5).abs() < 0.05, "sd {}", sd);
        // Unbounded tails: a uniform perturbation of ±0.5 never exceeds 0.5.
        assert!(deltas.iter().any(|d| d.abs() > 1f64));
    }

    #[test]
    fn bias_mutation_changes_the_configured_fraction_of_biases() {
        let mut rng = rng();
        for &rate in &[0.1, 0.5, 0.8] {
            let mut genome = Genome::new_unconnected(2, 1000);
            let config = MutationConfig::new()
                .add_connection_rate(0f64)
                .add_neuron_rate(0f64)
                .weight_mutation_rate(0f64)
                .toggle_expression_rate(0f64)
                .bias_mutation_rate(1f64)
                .bias_gene_mutation_rate(rate)
                .build();
            genome.add_gene(Gene::new(0, 2, 1f64, true));
            genome.mutate_with_config_and_rng(&config, &mut rng);
            let changed = (2..1002).filter(|&id| genome.bias(id) != 0f64).count();
            let fraction = changed as f64 / 1000f64;
            assert!(
                (fraction - rate).abs() <= 0.05,
                "rate {}: {} of the biases changed",
                rate,
                fraction
            );
            assert_eq!((genome.bias(0), genome.bias(1)), (0f64, 0f64));
        }
    }

    #[test]
    fn biases_never_exceed_bias_max() {
        let config = MutationConfig::new()
            .add_connection_rate(0f64)
            .add_neuron_rate(0f64)
            .weight_mutation_rate(0f64)
            .toggle_expression_rate(0f64)
            .bias_mutation_rate(1f64)
            .bias_gene_mutation_rate(1f64)
            .bias_perturbation_rate(0.8)
            .bias_mutate_power(5f64)
            .weight_init_range(10f64)
            .bias_max(2f64)
            .build();
        let mut rng = rng();
        let mut genome = Genome::new_initialized_with_rng(2, 3, &mut rng);
        let mut seen_at_limit = false;
        for _ in 0..500 {
            genome.mutate_with_config_and_rng(&config, &mut rng);
            for id in 0..genome.len() {
                assert!(genome.bias(id).abs() <= 2f64, "bias {}", genome.bias(id));
                seen_at_limit |= genome.bias(id).abs() == 2f64;
            }
        }
        assert!(seen_at_limit);
    }

    // Time constant τ per neuron (issue #69).

    fn only_tau_mutation(rate: f64) -> MutationConfig {
        MutationConfig::new()
            .add_connection_rate(0f64)
            .add_neuron_rate(0f64)
            .weight_mutation_rate(0f64)
            .toggle_expression_rate(0f64)
            .bias_mutation_rate(0f64)
            .tau_mutation_rate(rate)
            .tau_gene_mutation_rate(1f64)
            .build()
    }

    #[test]
    fn neuron_without_tau_gene_uses_the_default_tau() {
        let mut genome = Genome::new_initialized(2, 1);
        for id in 0..10 {
            assert_eq!(genome.tau(id), None);
        }
        genome.set_tau(2, 0.3);
        assert_eq!(genome.tau(2), Some(0.3));
        assert_eq!(genome.tau(3), None);
    }

    #[test]
    fn tau_mutation_perturbs_every_tau_within_bounds_except_inputs() {
        let config = MutationConfig::new().tau_range(0.005, 0.05).build();
        let config = MutationConfig {
            tau_mutation_rate: 1f64,
            tau_gene_mutation_rate: 1f64,
            tau_mutate_power: 2f64,
            ..only_tau_mutation_fields(config)
        };
        let mut genome = Genome::new_initialized(2, 1);
        let id = split(&mut genome, 0, 2, None);
        let mut rng = rng();
        let (mut seen_min, mut seen_max) = (false, false);
        for _ in 0..200 {
            let before = [genome.tau(2), genome.tau(id)];
            genome.mutate_with_config_and_rng(&config, &mut rng);
            assert_eq!(genome.tau(0), None);
            assert_eq!(genome.tau(1), None);
            for (n, &neuron) in [2, id].iter().enumerate() {
                let tau = genome.tau(neuron).expect("mutated tau");
                let at_bound = matches!(before[n], Some(t) if t == 0.005 || t == 0.05);
                assert!(
                    at_bound || Some(tau) != before[n],
                    "tau of {} not perturbed",
                    neuron
                );
                assert!((0.005..=0.05).contains(&tau), "tau {} out of range", tau);
                seen_min |= tau == 0.005;
                seen_max |= tau == 0.05;
            }
        }
        assert!(seen_min && seen_max);
    }

    /// `config` with every mutation but τ switched off.
    fn only_tau_mutation_fields(config: MutationConfig) -> MutationConfig {
        MutationConfig {
            add_connection_rate: 0f64,
            add_neuron_rate: 0f64,
            weight_mutation_rate: 0f64,
            toggle_expression_rate: 0f64,
            bias_mutation_rate: 0f64,
            ..config
        }
    }

    #[test]
    fn first_tau_mutation_starts_from_the_default_tau() {
        // A small perturbation of a neuron without τ gene lands near the
        // default τ, not near 0.
        let config = MutationConfig {
            tau: 0.2,
            tau_mutate_power: 0.01,
            ..only_tau_mutation(1f64)
        };
        let mut genome = Genome::new_initialized(1, 1);
        genome.mutate_with_config_and_rng(&config, &mut rng());
        let tau = genome.tau(1).unwrap();
        assert!((tau - 0.2).abs() < 0.02, "{}", tau);
    }

    #[test]
    fn tau_mutation_rate_zero_draws_nothing_and_changes_nothing() {
        // With τ mutation off the random draws are those of a genome
        // without τ genes (one per kind of mutation), so older seeds
        // reproduce.
        let mut genome = Genome::new_initialized(2, 1);
        genome.set_tau(2, 0.3);
        let mut used = rng();
        genome.mutate_with_config_and_rng(&only_tau_mutation(0f64), &mut used);
        assert_eq!(genome.tau(2), Some(0.3));
        assert_eq!(genome.tau(1), None);
        let mut expected = rng();
        for _ in 0..5 {
            expected.gen::<f64>();
        }
        assert_eq!(used.gen::<u64>(), expected.gen::<u64>());
    }

    #[test]
    fn tau_mutation_is_reproducible_with_same_seed() {
        let run = |seed| {
            let mut genome = Genome::new_initialized(2, 2);
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            for _ in 0..20 {
                genome.mutate_with_config_and_rng(&only_tau_mutation(1f64), &mut rng);
            }
            (genome.tau(2), genome.tau(3))
        };
        assert_eq!(run(3), run(3));
        assert!(run(3) != run(4));
    }

    #[test]
    fn new_neuron_is_born_with_the_default_tau() {
        let mut genome = Genome::default();
        genome.add_gene(Gene::new(0, 1, 1f64, true));
        genome.set_tau(1, 0.5);
        // A stale τ stored for the id the new neuron will take.
        genome.set_tau(2, 0.7);
        genome.mutate_add_neuron(&mut rng());
        assert_eq!(genome.len(), 3);
        assert_eq!(genome.tau(2), None);
        assert_eq!(genome.tau(1), Some(0.5));
    }

    /// A: neurons 0, 1; B: neurons 0, 1, 2. Different τ on 1, B has its own
    /// τ on 2.
    fn tau_parents() -> (Genome, Genome) {
        let (mut a, mut b) = bias_parents();
        a.set_tau(1, 0.1);
        b.set_tau(1, 0.3);
        b.set_tau(2, 0.7);
        (a, b)
    }

    #[test]
    fn crossover_inherits_tau_of_matching_neurons_from_both_parents() {
        let (a, b) = tau_parents();
        let mut rng = rng();
        let mut seen = std::collections::BTreeMap::new();
        for i in 0..1000 {
            let child = a.mate_with_rng(&b, i % 2 == 0, &mut rng);
            let tau = child.tau(1).unwrap();
            assert!(
                tau == 0.1 || tau == 0.3 || (tau - 0.2).abs() < 1e-12,
                "tau {} not inherited",
                tau
            );
            *seen.entry(format!("{:.3}", tau)).or_insert(0) += 1;
            assert_eq!(child.tau(2), b.tau(2).filter(|_| child.has_neuron(2)));
        }
        assert!(seen["0.100"] > 100 && seen["0.300"] > 100, "{:?}", seen);
    }

    #[test]
    fn crossover_keeps_tau_of_neuron_only_in_one_parent_or_default() {
        let (mut a, b) = tau_parents();
        a.set_tau(1, 0.1);
        let mut rng = rng();
        for _ in 0..200 {
            assert_eq!(a.mate_with_rng(&b, false, &mut rng).tau(2), Some(0.7));
            let child = a.mate_equal_with_rng(&b, &mut rng);
            if child.has_neuron(2) {
                assert_eq!(child.tau(2), Some(0.7));
            }
        }
        // A neuron without τ gene in both parents keeps the default.
        let (c, d) = bias_parents();
        assert_eq!(c.mate_with_rng(&d, false, &mut rng).tau(1), None);
    }

    #[test]
    fn compatibility_distance_counts_log_tau_differences_only_with_its_coefficient() {
        let mut a = genome_with(2, &[(0, 2), (1, 3), (3, 2)], 1f64);
        let mut b = a.clone();
        a.set_tau(2, 0.1);
        b.set_tau(2, 0.01);
        // Neuron 3 has the default τ in b.
        a.set_tau(3, 0.04);
        // Input τ never counts.
        a.set_tau(0, 100f64);
        assert_close(a.compatibility_distance(&b), 0f64);
        let config = MutationConfig {
            compatibility_tau_coefficient: 0.5,
            tau: 0.01,
            ..MutationConfig::default()
        };
        // T = (|ln 0.1 - ln 0.01| + |ln 0.04 - ln 0.01|) / 2 = (ln 10 + ln 4) / 2
        assert_close(
            a.compatibility_distance_with_config(&b, &config),
            0.5 * (10f64.ln() + 4f64.ln()) / 2f64,
        );
    }

    /// 30 genes over 8 distinct (in, out) pairs, in a shuffled order.
    fn thirty_genes_eight_pairs() -> Vec<Gene> {
        let pairs = [
            (0, 2),
            (1, 2),
            (0, 3),
            (1, 3),
            (2, 4),
            (4, 3),
            (0, 4),
            (3, 2),
        ];
        (0..30)
            .map(|k| {
                let (i, o) = pairs[(k * 5) % 8];
                Gene::new(i, o, k as f64, true)
            })
            .collect()
    }

    #[test]
    fn from_genes_keeps_one_gene_per_connection() {
        let genome = Genome::from_genes(thirty_genes_eight_pairs(), 4);
        assert_eq!(genome.get_genes().len(), 8);
        let mut sorted = connections(&genome);
        sorted.sort();
        sorted.dedup();
        assert_eq!(connections(&genome), sorted);
    }

    #[test]
    fn from_genes_keeps_the_last_enabled_duplicate_as_the_phenotype_did() {
        let genes = vec![
            Gene::new(0, 2, 1.0, true),
            Gene::new(0, 2, 2.0, true),
            Gene::new(0, 2, 3.0, false),
            Gene::new(1, 2, 4.0, false),
            Gene::new(1, 2, 5.0, false),
            Gene::new(0, 3, 6.0, true),
        ];
        let genome = Genome::from_genes(genes, 3);
        let kept: Vec<(usize, usize, f64, bool)> = genome
            .get_genes()
            .iter()
            .map(|g| (g.in_neuron_id(), g.out_neuron_id(), g.weight(), g.enabled()))
            .collect();
        assert_eq!(
            kept,
            vec![(0, 2, 2.0, true), (0, 3, 6.0, true), (1, 2, 5.0, false)]
        );
    }

    #[test]
    fn from_genes_with_duplicates_activates_like_the_original_genes() {
        // Before deduplicating, the phenotype used the last enabled gene of
        // each connection; the deduplicated genome must give the same outputs.
        let genes = thirty_genes_eight_pairs()
            .into_iter()
            .enumerate()
            .map(|(k, g)| {
                Gene::new(
                    g.in_neuron_id(),
                    g.out_neuron_id(),
                    (k as f64 - 15.0) / 10.0,
                    k % 3 != 0,
                )
            })
            .collect::<Vec<_>>();
        let mut expected = Genome::new_unconnected(2, 2);
        for gene in &genes {
            if gene.enabled() {
                expected.genes.retain(|g| g != gene);
                expected.genes.push(*gene);
                expected.genes.sort();
            }
        }
        expected.last_neuron_id = 4;
        let genome = Genome::from_genes_with_io(genes, 4, 2, 2);
        let mut a = crate::Organism::new(expected);
        let mut b = crate::Organism::new(genome);
        let (mut out_a, mut out_b) = (vec![0f64; 2], vec![0f64; 2]);
        for _ in 0..5 {
            a.activate(vec![0.3, 0.9], &mut out_a);
            b.activate(vec![0.3, 0.9], &mut out_b);
            assert_eq!(out_a, out_b);
        }
    }

    #[test]
    fn from_genes_with_io_knows_its_inputs_and_outputs() {
        let genes = vec![Gene::new(0, 2, 1.0, true), Gene::new(1, 2, 1.0, true)];
        let genome = Genome::from_genes_with_io(genes, 2, 2, 1);
        assert_eq!(genome.input_neurons(), 2);
        assert_eq!(genome.output_neurons(), 1);
        assert_eq!(genome.neurons(), vec![0, 1, 2]);
        // Mutation never connects into an input once the genome knows them.
        let mut rng = rng();
        let mut genome = genome;
        for _ in 0..200 {
            genome.mutate_add_connection_with_config(&MutationConfig::default(), &mut rng);
        }
        assert!(genome.get_genes().iter().all(|g| g.out_neuron_id() >= 2));
    }

    #[test]
    fn from_genes_with_io_keeps_unconnected_outputs() {
        let genome = Genome::from_genes_with_io(vec![Gene::new(0, 2, 1.0, true)], 0, 2, 2);
        assert_eq!(genome.neurons(), vec![0, 1, 2, 3]);
        assert_eq!(genome.len(), 4);
    }

    #[test]
    fn from_genes_never_leaves_last_neuron_id_below_a_gene() {
        let genome = Genome::from_genes(vec![Gene::new(0, 7, 1.0, true)], 1);
        assert_eq!(genome.len(), 8);
    }

    #[test]
    fn add_gene_keeps_genes_sorted_and_unique() {
        let mut genome = Genome::new_unconnected(3, 2);
        let order = [(2, 3), (0, 4), (1, 3), (0, 3), (2, 4), (1, 4), (0, 3)];
        for &(i, o) in order.iter() {
            genome.add_gene(Gene::new(i, o, 1.0, true));
        }
        assert_eq!(
            connections(&genome),
            vec![(0, 3), (0, 4), (1, 3), (1, 4), (2, 3), (2, 4)]
        );
    }
}
