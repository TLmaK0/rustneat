use serde_derive::{Deserialize, Serialize};

/// Contains all parameters for the NEAT algorithm. A reference to `Params` will be passed around
/// internally. Usually you only need to give it to `Population::evolve`.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Params {
    /// Number of inputs to the neural network
    pub n_inputs: usize,
    /// Number of outputs to the neural network
    pub n_outputs: usize,
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

impl Params {
    /// Sane default parameters
    pub fn default(n_inputs: usize, n_outputs: usize) -> Params {
        Params {
            n_inputs,
            n_outputs,
            // population
            prune_after_n_generations: 100,
            /// Maximum amount of species that survive a 'pruning'
            n_to_prune: 3,

            mutation_pr: 0.5,
            interspecie_mate_pr: 0.001,
            cull_fraction: 0.1,

            mutate_add_conn_pr: 0.5,
            mutate_del_conn_pr: 0.5,
            mutate_add_neuron_pr: 0.1,
            mutate_del_neuron_pr: 0.08,

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
    /// 30gen, 30pop, 300iter
    pub fn optimized_for_xor(n_inputs: usize, n_outputs: usize) -> Params {
        Params {
            n_inputs,
            n_outputs,
            prune_after_n_generations: 24,
            n_to_prune: 3,
            mutation_pr: 0.7789911380525976,
            interspecie_mate_pr: 0.00011142344146628424,
            cull_fraction: 0.08461952672427815,
            mutate_add_conn_pr: 0.034419321300764874,
            mutate_del_conn_pr: 0.02,
            mutate_add_neuron_pr: 0.010348211728832088,
            mutate_del_neuron_pr: 0.001,
            weight_init_mean: 0.0,
            weight_init_var: 0.739315871769454,
            weight_mutate_var: 0.6278347591284132,
            weight_mutate_pr: 0.5178002229984818,
            weight_replace_pr: 0.14323022713090744,
            bias_init_mean: 0.0,
            bias_init_var: 0.6638680611444077,
            bias_mutate_var: 0.3262182872253431,
            bias_mutate_pr: 0.46351108180984113,
            bias_replace_pr: 0.15485210902008673,
            include_weak_disjoint_gene: 0.2592218898273462,
            compatibility_threshold: 2.6013055853422187,
            distance_weight_coef: 0.43937041525713183,
            distance_disjoint_coef: 0.8007059311648334
        }
    }
    ///
    pub fn optimized_for_xor2(n_inputs: usize, n_outputs: usize) -> Params {
        Params {
            n_inputs,
            n_outputs,
            prune_after_n_generations: 18,
            n_to_prune: 2,
            mutation_pr: 0.42511383143306747,
            interspecie_mate_pr: 0.0011253527448187332,
            cull_fraction: 0.26378491719510166,
            mutate_add_conn_pr: 0.03846985213713843,
            mutate_del_conn_pr: 0.02,
            mutate_add_neuron_pr: 0.011185122817130088,
            mutate_del_neuron_pr: 0.001,
            weight_init_mean: 0.0,
            weight_init_var: 1.6933681857715341,
            weight_mutate_var: 0.6132724760136441,
            weight_mutate_pr: 0.3793931785569132,
            weight_replace_pr: 0.07344761865136407,
            bias_init_mean: 0.0,
            bias_init_var: 1.576323126898198,
            bias_mutate_var: 0.33893450130417346,
            bias_mutate_pr: 0.6606222432143025,
            bias_replace_pr: 0.020981083224497044,
            include_weak_disjoint_gene: 0.2742797470379294,
            compatibility_threshold: 3.111548605546438,
            distance_weight_coef: 0.27947178737291656,
            distance_disjoint_coef: 0.7320224906467249
        }
    }
}
