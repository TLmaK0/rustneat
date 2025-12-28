use std::ops::RangeInclusive;

/// Defines the search space for hyperparameter optimization.
///
/// Each parameter can have a range of values to explore.
#[derive(Debug, Clone)]
pub struct SearchSpace {
    /// Range for weight mutation rate
    pub weight_mutation_rate: Option<RangeInclusive<f64>>,
    /// Range for add connection rate
    pub add_connection_rate: Option<RangeInclusive<f64>>,
    /// Range for add neuron rate
    pub add_neuron_rate: Option<RangeInclusive<f64>>,
    /// Range for toggle expression rate
    pub toggle_expression_rate: Option<RangeInclusive<f64>>,
    /// Range for weight perturbation rate
    pub weight_perturbation_rate: Option<RangeInclusive<f64>>,
    /// Range for toggle bias rate
    pub toggle_bias_rate: Option<RangeInclusive<f64>>,
    /// Range for compatibility threshold
    pub compatibility_threshold: Option<RangeInclusive<f64>>,
}

impl Default for SearchSpace {
    fn default() -> Self {
        Self::new()
    }
}

impl SearchSpace {
    /// Create a new empty search space
    pub fn new() -> Self {
        SearchSpace {
            weight_mutation_rate: None,
            add_connection_rate: None,
            add_neuron_rate: None,
            toggle_expression_rate: None,
            weight_perturbation_rate: None,
            toggle_bias_rate: None,
            compatibility_threshold: None,
        }
    }

    /// Set range for weight mutation rate
    pub fn weight_mutation_rate(mut self, range: RangeInclusive<f64>) -> Self {
        self.weight_mutation_rate = Some(range);
        self
    }

    /// Set range for add connection rate
    pub fn add_connection_rate(mut self, range: RangeInclusive<f64>) -> Self {
        self.add_connection_rate = Some(range);
        self
    }

    /// Set range for add neuron rate
    pub fn add_neuron_rate(mut self, range: RangeInclusive<f64>) -> Self {
        self.add_neuron_rate = Some(range);
        self
    }

    /// Set range for toggle expression rate
    pub fn toggle_expression_rate(mut self, range: RangeInclusive<f64>) -> Self {
        self.toggle_expression_rate = Some(range);
        self
    }

    /// Set range for weight perturbation rate
    pub fn weight_perturbation_rate(mut self, range: RangeInclusive<f64>) -> Self {
        self.weight_perturbation_rate = Some(range);
        self
    }

    /// Set range for toggle bias rate
    pub fn toggle_bias_rate(mut self, range: RangeInclusive<f64>) -> Self {
        self.toggle_bias_rate = Some(range);
        self
    }

    /// Set range for compatibility threshold
    pub fn compatibility_threshold(mut self, range: RangeInclusive<f64>) -> Self {
        self.compatibility_threshold = Some(range);
        self
    }

    /// Check if any parameter has a range defined
    pub fn is_empty(&self) -> bool {
        self.weight_mutation_rate.is_none()
            && self.add_connection_rate.is_none()
            && self.add_neuron_rate.is_none()
            && self.toggle_expression_rate.is_none()
            && self.weight_perturbation_rate.is_none()
            && self.toggle_bias_rate.is_none()
            && self.compatibility_threshold.is_none()
    }

    /// Count how many parameters are being tuned
    pub fn num_parameters(&self) -> usize {
        let mut count = 0;
        if self.weight_mutation_rate.is_some() { count += 1; }
        if self.add_connection_rate.is_some() { count += 1; }
        if self.add_neuron_rate.is_some() { count += 1; }
        if self.toggle_expression_rate.is_some() { count += 1; }
        if self.weight_perturbation_rate.is_some() { count += 1; }
        if self.toggle_bias_rate.is_some() { count += 1; }
        if self.compatibility_threshold.is_some() { count += 1; }
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_space_builder() {
        let space = SearchSpace::new()
            .add_connection_rate(0.01..=0.10)
            .add_neuron_rate(0.01..=0.05);

        assert!(!space.is_empty());
        assert_eq!(space.num_parameters(), 2);
        assert!(space.add_connection_rate.is_some());
        assert!(space.add_neuron_rate.is_some());
        assert!(space.weight_mutation_rate.is_none());
    }

    #[test]
    fn test_empty_search_space() {
        let space = SearchSpace::new();
        assert!(space.is_empty());
        assert_eq!(space.num_parameters(), 0);
    }
}
