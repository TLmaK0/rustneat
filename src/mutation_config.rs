use crate::genome::{
    COMPATIBILITY_THRESHOLD, MUTATE_ADD_CONNECTION, MUTATE_ADD_NEURON,
    MUTATE_CONNECTION_WEIGHT, MUTATE_CONNECTION_WEIGHT_PERTURBED_PROBABILITY,
    MUTATE_TOGGLE_BIAS, MUTATE_TOGGLE_EXPRESSION,
};

/// Configuration for mutation rates in NEAT
///
/// Allows customizing mutation probabilities per problem.
/// Use `MutationConfig::default()` for standard NEAT values defined in `genome.rs`.
#[derive(Debug, Clone, Copy)]
pub struct MutationConfig {
    /// Probability of mutating connection weights
    pub weight_mutation_rate: f64,
    /// Probability of adding a new connection
    pub add_connection_rate: f64,
    /// Probability of adding a new neuron
    pub add_neuron_rate: f64,
    /// Probability of toggling a connection's enabled state
    pub toggle_expression_rate: f64,
    /// Probability of perturbing vs replacing weight
    pub weight_perturbation_rate: f64,
    /// Probability of toggling bias
    pub toggle_bias_rate: f64,
    /// Compatibility threshold for speciation
    pub compatibility_threshold: f64,
}

impl Default for MutationConfig {
    fn default() -> Self {
        MutationConfig {
            weight_mutation_rate: MUTATE_CONNECTION_WEIGHT,
            add_connection_rate: MUTATE_ADD_CONNECTION,
            add_neuron_rate: MUTATE_ADD_NEURON,
            toggle_expression_rate: MUTATE_TOGGLE_EXPRESSION,
            weight_perturbation_rate: MUTATE_CONNECTION_WEIGHT_PERTURBED_PROBABILITY,
            toggle_bias_rate: MUTATE_TOGGLE_BIAS,
            compatibility_threshold: COMPATIBILITY_THRESHOLD,
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

    /// Set toggle bias rate
    pub fn toggle_bias_rate(mut self, rate: f64) -> Self {
        self.config.toggle_bias_rate = rate;
        self
    }

    /// Set compatibility threshold
    pub fn compatibility_threshold(mut self, threshold: f64) -> Self {
        self.config.compatibility_threshold = threshold;
        self
    }

    /// Build the configuration
    pub fn build(self) -> MutationConfig {
        self.config
    }
}
