use serde_derive::{Deserialize, Serialize};

/// Contains all parameters for the NEAT algorithm. A reference to `NeatParams`
/// will be passed around internally. Usually you only need to give it to
/// `Population::evolve`.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct NeatParams {
    /// Number of inputs to the neural network. Only used to ensure a certain
    /// amount of neurons.
    pub n_inputs: usize,
    /// Number of outputs to the neural network. Only used to ensure a certain
    /// amount of neurons.
    pub n_outputs: usize,
    // In `Population`
    /// Maximum number of generations without improvement in a species, before
    /// that species is removed.
    pub remove_after_n_generations: usize,
    /// Number of best species that cannot be removed due to stagnation
    pub species_elite: usize,

    // In `Specie`
    /// The probability of just mutating (as opposed to mating), during
    /// selection
    pub mutation_pr: f64,
    /// The probability that an organism mates with an organism outside of their
    /// species
    pub interspecie_mate_pr: f64,
    /// The fraction of organisms to cull from a species before selection
    /// (worst-performing ones)
    pub cull_fraction: f64,
    // TODO: n_elites or elite_fraction

    // Topological mutations
    /// The probability of adding a connection during mutation
    pub mutate_add_conn_pr: f64,
    /// The probability of deleting a connection during mutation
    pub mutate_del_conn_pr: f64,
    /// The probability of adding a neuron during mutation
    pub mutate_add_neuron_pr: f64,
    /// The probability of deleting a neuron during mutation
    pub mutate_del_neuron_pr: f64,

    /// The mean (normal distribution) of the weight of a new connection
    pub weight_init_mean: f64,
    /// The variance (normal distribution) of the weight of a new connection
    pub weight_init_var: f64,
    /// The variance (normal distribution) of a mutation of the weight of an
    /// existing connection
    pub weight_mutate_var: f64,
    /// The probability to perturb the weights of a connection (simulated for
    /// each connection individually) when mutating
    pub weight_mutate_pr: f64,
    /// The probability to replace the weights of a connection (simulated for
    /// each connection individually) when mutating
    pub weight_replace_pr: f64,

    /// The mean (normal distribution) of the bias of a new connection
    pub bias_init_mean: f64,
    /// The variance (normal distribution) of the bias of a new connection
    pub bias_init_var: f64,
    /// The variance (normal distribution) of a mutation of the bias of an
    /// existing connection
    pub bias_mutate_var: f64,
    /// The probability to perturb the biases of a connection (simulated for
    /// each connection individually) when mutating
    pub bias_mutate_pr: f64,
    /// The probability to replace the biases of a connection (simulated for
    /// each connection individually) when mutating
    pub bias_replace_pr: f64,

    /// The probability, during mating, of including a gene that is disjoint or
    /// excess, from the organisms that is least fit
    pub include_weak_disjoint_gene: f64,

    // (TODO: tau and n_steps in `activate()`

    // Other
    /// Threshold for distance (compatibility) between organisms,
    /// under which the organisms are considered 'compatible', i.e. belonging to
    /// the same species.
    pub compatibility_threshold: f64,
    /// How much connection weights and node biases contribute to the distance.
    pub distance_weight_coef: f64,
    /// How much disjoint/excess (not in common) connections and neurons
    /// contribute to the distance
    pub distance_disjoint_coef: f64,
}

impl NeatParams {
    /// Sane default parameters
    pub fn default(n_inputs: usize, n_outputs: usize) -> NeatParams {
        NeatParams {
            n_inputs,
            n_outputs,
            // population
            remove_after_n_generations: 20,
            species_elite: 2,

            mutation_pr: 0.5,
            interspecie_mate_pr: 0.001,
            cull_fraction: 0.2,

            mutate_add_conn_pr: 0.5,
            mutate_del_conn_pr: 0.5,
            mutate_add_neuron_pr: 0.1,
            mutate_del_neuron_pr: 0.1,

            weight_init_mean: 0.0,
            weight_init_var: 1.0,
            weight_mutate_var: 0.5,
            weight_mutate_pr: 0.8,
            weight_replace_pr: 0.1,

            bias_init_mean: 0.0,
            bias_init_var: 1.0,
            bias_mutate_var: 0.5,
            bias_mutate_pr: 0.7,
            bias_replace_pr: 0.1,

            include_weak_disjoint_gene: 0.2,

            // other
            compatibility_threshold: 3.0,
            distance_weight_coef: 0.5,
            distance_disjoint_coef: 1.0,
        }
    }
}

impl NeatParams {
    // temporary
    /// 100gen, 26pop, 250iter
    pub fn optimized_for_xor3(n_inputs: usize, n_outputs: usize) -> NeatParams {
        NeatParams {
            n_inputs,
            n_outputs,
            remove_after_n_generations: 25,
            species_elite: 4,
            mutation_pr: 0.817_948_608_000_698_1,
            interspecie_mate_pr: 0.000_786_747_963_916_689_3,
            cull_fraction: 0.170_965_494_802_234_66,
            mutate_add_conn_pr: 0.447_539_529_772_954_75,
            mutate_del_conn_pr: 0.124_879_731_795_234_51,
            mutate_add_neuron_pr: 0.018_564_851_821_478_344,
            mutate_del_neuron_pr: 0.032_637_713_799_404_23,
            weight_init_mean: 0.0,
            weight_init_var: 0.941_304_288_479_847_3,
            weight_mutate_var: 0.853_903_593_419_955_7,
            weight_mutate_pr: 0.293_849_632_324_998_7,
            weight_replace_pr: 0.020_513_723_672_854_978,
            bias_init_mean: 0.0,
            bias_init_var: 0.934_694_004_792_945_3,
            bias_mutate_var: 0.251_537_605_304_202_27,
            bias_mutate_pr: 0.256_824_665_804_256_3,
            bias_replace_pr: 0.137_209_850_104_071_94,
            include_weak_disjoint_gene: 0.292_298_273_802_692_9,
            compatibility_threshold: 3.077_294_494_323_634_7,
            distance_weight_coef: 0.322_727_707_366_624_26,
            distance_disjoint_coef: 0.745_728_980_671_972_9,
        }
    }
}
