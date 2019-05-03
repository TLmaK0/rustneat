use serde_derive::{Deserialize, Serialize};

/// Contains all parameters for the NEAT algorithm. A reference to `NeatParams` will be passed around
/// internally. Usually you only need to give it to `Population::evolve`.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct NeatParams {
    /// Number of inputs to the neural network. Only used to ensure a certain amount of neurons.
    pub n_inputs: usize,
    /// Number of outputs to the neural network. Only used to ensure a certain amount of neurons.
    pub n_outputs: usize,
    // In `Population`
    /// Maximum number of generations without improvement in a species, before that species is
    /// removed.
    pub remove_after_n_generations: usize,
    /// Number of best species that cannot be removed due to stagnation
    pub species_elite: usize,

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
    /// temporary
    /// 30gen, 30pop, 300iter
    pub fn optimized_for_xor(n_inputs: usize, n_outputs: usize) -> NeatParams {
        NeatParams {
            n_inputs,
            n_outputs,
            remove_after_n_generations: 24,
            species_elite: 2,
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
    pub fn optimized_for_xor2(n_inputs: usize, n_outputs: usize) -> NeatParams {
        NeatParams {
            n_inputs,
            n_outputs,
            remove_after_n_generations: 18,
            species_elite: 2,
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
    /// 100gen, 26pop, 250iter
    pub fn optimized_for_xor3(n_inputs: usize, n_outputs: usize) -> NeatParams {
        NeatParams {
            n_inputs,
            n_outputs,
            remove_after_n_generations: 25,
            species_elite: 4,
            mutation_pr: 0.8179486080006981,
            interspecie_mate_pr: 0.0007867479639166893,
            cull_fraction: 0.17096549480223466,
            mutate_add_conn_pr: 0.44753952977295475,
            mutate_del_conn_pr: 0.12487973179523451,
            mutate_add_neuron_pr: 0.018564851821478344,
            mutate_del_neuron_pr: 0.03263771379940423,
            weight_init_mean: 0.0,
            weight_init_var: 0.9413042884798473,
            weight_mutate_var: 0.8539035934199557,
            weight_mutate_pr: 0.2938496323249987,
            weight_replace_pr: 0.020513723672854978,
            bias_init_mean: 0.0,
            bias_init_var: 0.9346940047929453,
            bias_mutate_var: 0.25153760530420227,
            bias_mutate_pr: 0.2568246658042563,
            bias_replace_pr: 0.13720985010407194,
            include_weak_disjoint_gene: 0.2922982738026929,
            compatibility_threshold: 3.0772944943236347,
            distance_weight_coef: 0.32272770736662426,
            distance_disjoint_coef: 0.7457289806719729
        }
    }
    ///
    pub fn optimized_for_xor4(n_inputs: usize, n_outputs: usize) -> NeatParams {
        NeatParams {
            n_inputs,
            n_outputs,
            remove_after_n_generations: 24,
            species_elite: 4,
            mutation_pr: 0.7377448588347766,
            interspecie_mate_pr: 0.00035009502538494,
            cull_fraction: 0.23275864645590494,
            mutate_add_conn_pr: 0.4265089877216002,
            mutate_del_conn_pr: 0.10236231655587034,
            mutate_add_neuron_pr: 0.01646785755288386,
            mutate_del_neuron_pr: 0.026939146200355306,
            weight_init_mean: 0.0,
            weight_init_var: 1.13057738327287,
            weight_mutate_var: 0.7654186135703698,
            weight_mutate_pr: 0.5121623674085568,
            weight_replace_pr: 0.035396420187588075,
            bias_init_mean: 0.0,
            bias_init_var: 0.9190528106125618,
            bias_mutate_var: 0.3354386887591801,
            bias_mutate_pr: 0.25787915314264925,
            bias_replace_pr: 0.12951042590409398,
            include_weak_disjoint_gene: 0.14737065873007102,
            compatibility_threshold: 2.89666508232241,
            distance_weight_coef: 0.1973310444507289,
            distance_disjoint_coef: 0.6121188685737646
        }
    }
    ///
    pub fn optimized_for_xor3_200pop(n_inputs: usize, n_outputs: usize) -> NeatParams {
        NeatParams {
            n_inputs,
            n_outputs,
            remove_after_n_generations: 32,
            species_elite: 1,
            mutation_pr: 0.5327950655843168,
            interspecie_mate_pr: 0.00007966830274460968,
            cull_fraction: 0.22026134738323333,
            mutate_add_conn_pr: 0.4503863062666368,
            mutate_del_conn_pr: 0.10673123924170867,
            mutate_add_neuron_pr: 0.011488878905223616,
            mutate_del_neuron_pr: 0.03517207664947583,
            weight_init_mean: 0.0,
            weight_init_var: 1.7662735379957946,
            weight_mutate_var: 1.6383118400258092,
            weight_mutate_pr: 0.6752697941225041,
            weight_replace_pr: 0.12828394343804145,
            bias_init_mean: 0.0,
            bias_init_var: 1.3050124837892862,
            bias_mutate_var: 0.22056449217883092,
            bias_mutate_pr: 0.4522477279407073,
            bias_replace_pr: 0.09809841413336916,
            include_weak_disjoint_gene: 0.2662004228007475,
            compatibility_threshold: 2.7137103625511902,
            distance_weight_coef: 0.06185651136671744,
            distance_disjoint_coef: 0.7670793572284181
        }
    }
}
