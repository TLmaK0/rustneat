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
    /// For measuring distance between neural networks (cf. paper)
    pub c2: f64,
    /// For measuring distance between neural networks
    pub c3: f64,
    /// The probability of mutating connection weights during mutation
    pub mutate_conn_weight_pr: f64,
    /// Once mutating the connection weight: the probability of _adding_ to the current weight
    /// rather than _assigning_ to it. (For now, this also applies to mutating neuron bias)
    pub mutate_conn_weight_perturbed_pr: f64,
    /// The maximum number of connections to mutate. A number `x <= 0` means `n_connections - x`.
    pub n_conn_to_mutate: i32,
    /// The probability of adding a connection during mutation
    pub mutate_add_conn_pr: f64,
    /// The probability of adding a neuron during mutation
    pub mutate_add_neuron_pr: f64,
    /// The probability of toggling a connection during mutation
    pub mutate_toggle_expr_pr: f64,
    /// The probability of mutating the bias of a neuron during mutation
    pub mutate_bias_pr: f64,

    /// The probability, during mating, of including a gene that is disjoint or excess,
    /// from the organisms that is least fit
    pub include_weak_disjoint_gene: f64,

    // (TODO: tau and n_steps in `activate()`

    // Other
    /// Threshold for distance between organisms, under which the organisms are considered
    /// 'compatible', i.e. belonging to the same species.
    pub compatibility_threshold: f64,
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
            c2: 1.0,
            c3: 0.0,
            mutate_conn_weight_pr: 0.9,
            mutate_conn_weight_perturbed_pr: 0.9,
            n_conn_to_mutate: 0,
            mutate_add_conn_pr: 0.005,
            mutate_add_neuron_pr: 0.004,
            mutate_toggle_expr_pr: 0.001,
            mutate_bias_pr: 0.001,
            include_weak_disjoint_gene: 0.2,

            // other
            compatibility_threshold: 3.0,
        }
    }
}
impl Params {
    /// Temporary...
    pub fn optimized_for_xor() -> Params {
        Params {
            prune_after_n_generations: 32,
            n_to_prune: 2,

            mutation_pr: 0.32906693983692487,
            interspecie_mate_pr: 0.001,
            cull_fraction: 0.2,

            c2: 0.8,
            c3: 0.16,
            mutate_conn_weight_pr: 0.7925931762803848,
            mutate_conn_weight_perturbed_pr: 0.8070611818456519,
            n_conn_to_mutate: 0,
            mutate_add_conn_pr: 0.0036613530568043416,
            mutate_add_neuron_pr: 0.001,
            mutate_toggle_expr_pr: 0.001,
            mutate_bias_pr: 0.0015733578021236633,
            include_weak_disjoint_gene: 0.14114733660591033,
            compatibility_threshold: 3.2385579112148832
        }
    }
    ///
    pub fn optimized_for_xor2() -> Params {
        Params {
            prune_after_n_generations: 37,
            n_to_prune: 3,
            mutation_pr: 0.7409024658448657,
            interspecie_mate_pr: 0.0009617417457758326,
            cull_fraction: 0.12205116814436302,

            c2: 0.8,
            c3: 0.16,
            mutate_conn_weight_pr: 0.38835869194909484,
            mutate_conn_weight_perturbed_pr: 0.8932268971133872,
            n_conn_to_mutate: 0,
            mutate_add_conn_pr: 0.003539267758935184,
            mutate_add_neuron_pr: 0.001,
            mutate_toggle_expr_pr: 0.001707570363667156,
            mutate_bias_pr: 0.022186130799031763,
            include_weak_disjoint_gene: 0.18318240866880092,
            compatibility_threshold: 3.1724911445144377
        }
    }
    /// From hyper opt where we start only with 1 neuron, no connections.
    pub fn optimized_for_xor3() -> Params {
        Params {
            prune_after_n_generations: 12,
            n_to_prune: 3,
            mutation_pr: 0.5498701920443196,
            interspecie_mate_pr: 0.001944535580070709,
            cull_fraction: 0.22915365169642715,

            c2: 0.8,
            c3: 0.16,
            mutate_conn_weight_pr: 0.8969159885233875,
            mutate_conn_weight_perturbed_pr: 0.8894245496918445,
            n_conn_to_mutate: 0,
            mutate_add_conn_pr: 0.0034247772002724414,
            mutate_add_neuron_pr: 0.001,
            mutate_toggle_expr_pr: 0.0014479539087863408,
            mutate_bias_pr: 0.04062414601000685,
            include_weak_disjoint_gene: 0.157536468189799,
            compatibility_threshold: 3.2857599138723064
        }
    }
}


