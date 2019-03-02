use serde_derive::{Deserialize, Serialize};

/// Contains all parameters for the NEAT algorithm. A reference to `Params` will be passed around
/// internally. Usually you only need to give it to `Population::evolve`.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Params {
    // In `Population`
    /// Maximum nuumber of generations without improvement before proceed with only the two best
    /// species.
    pub max_generations_without_improvement: usize,

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
            max_generations_without_improvement: 20,
            mutation_pr: 0.25,
            interspecie_mate_pr: 0.001,
            cull_fraction: 0.1,

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

            compatibility_threshold: 3.0,
        }
    }
}
