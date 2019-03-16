use serde_derive::{Deserialize, Serialize};

/// Contains all parameters for the NEAT algorithm. A reference to `Params` will be passed around
/// internally. Usually you only need to give it to `Population::evolve`.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Params {
    // In `Population`
    /// Maximum nuumber of generations without improvement before proceed with only the `n_to_prune` best
    /// species.
    pub prune_after_n_generations: usize,
    /// Maximum amount of species that survive a 'pruning'
    pub n_to_prune: usize,

    // In `Specie`
    /// The probability of just mutating (as opposed to mating), during selection
    pub mutation_pr: f64,
    /// The probability that an organism mates with an organism outside of their species
    pub interspecie_mate_pr: f64,
    /// The fraction of organisms to cull from a species before selection (worst-performing ones)
    pub cull_fraction: f64,
    // TODO: n_elites or elite_fraction

    // In `NeuralNetwork`
    /// The maximum number of connections to mutate. A number `x <= 0` means `n_connections - x`.
    pub n_conn_to_mutate: i32,
    /// The probability of adding a connection during mutation
    pub mutate_add_conn_pr: f64,
    /// The probability of adding a neuron during mutation
    pub mutate_add_neuron_pr: f64,
    /// The probability of toggling a connection during mutation
    pub mutate_toggle_expr_pr: f64,


    /// The mean (normal distribution) of the weight of a new connection
    pub weight_init_mean: f64, 
    /// The variance (normal distribution) of the weight of a new connection
    pub weight_init_var: f64, 
    /// The variance (normal distribution) of a mutation of the weight of an existing connection
    pub weight_mutate_var: f64,
    /// The probability to perturb the weights of a connection (simulated for each connection
    /// individually) when mutating
    pub weight_mutate_pr: f64,
    /// The probability to replace the weights of a connection (simulated for each connection
    /// individually) when mutating
    pub weight_replace_pr: f64,

    /// The mean (normal distribution) of the bias of a new connection
    pub bias_init_mean: f64, 
    /// The variance (normal distribution) of the bias of a new connection
    pub bias_init_var: f64, 
    /// The variance (normal distribution) of a mutation of the bias of an existing connection
    pub bias_mutate_var: f64,
    /// The probability to perturb the biases of a connection (simulated for each connection
    /// individually) when mutating
    pub bias_mutate_pr: f64,
    /// The probability to replace the biases of a connection (simulated for each connection
    /// individually) when mutating
    pub bias_replace_pr: f64,

    /// The probability, during mating, of including a gene that is disjoint or excess,
    /// from the organisms that is least fit
    pub include_weak_disjoint_gene: f64,

    // (TODO: tau and n_steps in `activate()`

    // Other
    /// Threshold for distance (compatibility) between organisms,
    /// under which the organisms are considered 'compatible', i.e. belonging to the same species.
    pub compatibility_threshold: f64,
    /// How much connection weights and node biases contribute to the distance.
    pub distance_weight_coef: f64,
    /// How much disjoint/excess (not in common) connections and neurons contribute to the distance
    pub distance_disjoint_coef: f64,
}

impl Default for Params {
    /// Sane default parameters
    fn default() -> Params {
        Params {
            // population
            prune_after_n_generations: 20,
            /// Maximum amount of species that survive a 'pruning'
            n_to_prune: 3,

            // specie
            mutation_pr: 0.25,
            interspecie_mate_pr: 0.001,
            cull_fraction: 0.1,

            // neural network
            n_conn_to_mutate: 0,
            mutate_add_conn_pr: 0.05,
            mutate_add_neuron_pr: 0.01,
            mutate_toggle_expr_pr: 0.001,

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



impl Params {
    /// temporary
    pub fn optimized_for_xor() -> Params {
        Params {
            prune_after_n_generations: 19,
            n_to_prune: 2,
            mutation_pr: 0.886247407101759,
            interspecie_mate_pr: 0.001989480993503591,
            cull_fraction: 0.1678207776504504,
            n_conn_to_mutate: 0,
            mutate_add_conn_pr: 0.003705672926030052,
            mutate_add_neuron_pr: 0.001,
            mutate_toggle_expr_pr: 0.004741272230391003,

            weight_init_mean: 0.0,
            weight_init_var: 1.63124335619358,
            weight_mutate_var: 0.7944613405433263,
            weight_mutate_pr: 0.22264676958558907,
            weight_replace_pr: 0.07487760506419014,

            bias_init_mean: 0.0,
            bias_init_var: 1.1368762515242272,
            bias_mutate_var: 0.5519888080987314,
            bias_mutate_pr: 0.2533586967820932,
            bias_replace_pr: 0.18821040418714985,
            include_weak_disjoint_gene: 0.22339770722706298,
            compatibility_threshold: 2.087256136498854,
            distance_weight_coef: 0.2717466608295739,
            distance_disjoint_coef: 0.9412445790399938
        }
    }
}
