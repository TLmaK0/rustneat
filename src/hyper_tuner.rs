use crate::environment::Environment;
use crate::mutation_config::MutationConfig;
use crate::population::Population;
use crate::search_space::SearchSpace;
use rand::Rng;

/// Result of a single trial during hyperparameter search
#[derive(Debug, Clone)]
pub struct TrialResult {
    /// The configuration used in this trial
    pub config: MutationConfig,
    /// Best fitness achieved
    pub best_fitness: f64,
    /// Number of generations run
    pub generations: usize,
}

/// Result of the hyperparameter optimization
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// The best configuration found
    pub best_config: MutationConfig,
    /// Best fitness achieved with the best config
    pub best_fitness: f64,
    /// All trial results
    pub trials: Vec<TrialResult>,
}

/// Hyperparameter tuner using Random Search strategy
///
/// # Example
/// ```ignore
/// use rustneat::{HyperTuner, SearchSpace, Environment};
///
/// let search_space = SearchSpace::new()
///     .add_connection_rate(0.01..=0.10)
///     .add_neuron_rate(0.01..=0.05);
///
/// let tuner = HyperTuner::new(search_space)
///     .population_size(150)
///     .input_neurons(8)
///     .output_neurons(4)
///     .generations_per_trial(50)
///     .num_trials(10);
///
/// let result = tuner.optimize(&environment);
/// println!("Best config: {:?}", result.best_config);
/// ```
pub struct HyperTuner {
    search_space: SearchSpace,
    population_size: usize,
    input_neurons: usize,
    output_neurons: usize,
    generations_per_trial: usize,
    num_trials: usize,
    early_stop_fitness: Option<f64>,
    verbose: bool,
}

impl HyperTuner {
    /// Create a new HyperTuner with the given search space
    pub fn new(search_space: SearchSpace) -> Self {
        HyperTuner {
            search_space,
            population_size: 150,
            input_neurons: 0,
            output_neurons: 0,
            generations_per_trial: 50,
            num_trials: 10,
            early_stop_fitness: None,
            verbose: true,
        }
    }

    /// Set the population size for each trial
    pub fn population_size(mut self, size: usize) -> Self {
        self.population_size = size;
        self
    }

    /// Set the number of input neurons
    pub fn input_neurons(mut self, n: usize) -> Self {
        self.input_neurons = n;
        self
    }

    /// Set the number of output neurons
    pub fn output_neurons(mut self, n: usize) -> Self {
        self.output_neurons = n;
        self
    }

    /// Set number of generations to run per trial
    pub fn generations_per_trial(mut self, gens: usize) -> Self {
        self.generations_per_trial = gens;
        self
    }

    /// Set number of trials (random configurations to try)
    pub fn num_trials(mut self, trials: usize) -> Self {
        self.num_trials = trials;
        self
    }

    /// Set early stopping fitness threshold
    pub fn early_stop_fitness(mut self, fitness: f64) -> Self {
        self.early_stop_fitness = Some(fitness);
        self
    }

    /// Enable or disable verbose output
    pub fn verbose(mut self, v: bool) -> Self {
        self.verbose = v;
        self
    }

    /// Sample a random value from a range
    fn sample_range(range: &std::ops::RangeInclusive<f64>) -> f64 {
        let mut rng = rand::thread_rng();
        let start = *range.start();
        let end = *range.end();
        // rand 0.4 uses gen_range(low, high) - exclusive on high
        // Add small epsilon to include high end
        start + rng.gen::<f64>() * (end - start)
    }

    /// Generate a random configuration from the search space
    fn sample_config(&self) -> MutationConfig {
        let default = MutationConfig::default();

        let weight_mutation_rate = self.search_space.weight_mutation_rate
            .as_ref()
            .map(Self::sample_range)
            .unwrap_or(default.weight_mutation_rate);

        let add_connection_rate = self.search_space.add_connection_rate
            .as_ref()
            .map(Self::sample_range)
            .unwrap_or(default.add_connection_rate);

        let add_neuron_rate = self.search_space.add_neuron_rate
            .as_ref()
            .map(Self::sample_range)
            .unwrap_or(default.add_neuron_rate);

        let toggle_expression_rate = self.search_space.toggle_expression_rate
            .as_ref()
            .map(Self::sample_range)
            .unwrap_or(default.toggle_expression_rate);

        let weight_perturbation_rate = self.search_space.weight_perturbation_rate
            .as_ref()
            .map(Self::sample_range)
            .unwrap_or(default.weight_perturbation_rate);

        let toggle_bias_rate = self.search_space.toggle_bias_rate
            .as_ref()
            .map(Self::sample_range)
            .unwrap_or(default.toggle_bias_rate);

        let compatibility_threshold = self.search_space.compatibility_threshold
            .as_ref()
            .map(Self::sample_range)
            .unwrap_or(default.compatibility_threshold);

        MutationConfig {
            weight_mutation_rate,
            add_connection_rate,
            add_neuron_rate,
            toggle_expression_rate,
            weight_perturbation_rate,
            toggle_bias_rate,
            compatibility_threshold,
        }
    }

    /// Run a single trial with the given configuration
    fn run_trial(&self, config: MutationConfig, environment: &dyn Environment) -> TrialResult {
        let mut population = if self.input_neurons > 0 && self.output_neurons > 0 {
            Population::create_population_initialized_with_config(
                self.population_size,
                self.input_neurons,
                self.output_neurons,
                config,
            )
        } else {
            Population::create_population_with_config(self.population_size, config)
        };

        let mut best_fitness = f64::NEG_INFINITY;
        let mut generations_run = 0;

        for gen in 0..self.generations_per_trial {
            population.evolve();
            population.evaluate_in(environment);
            generations_run = gen + 1;

            if let Some(ref champion) = population.champion {
                if champion.fitness > best_fitness {
                    best_fitness = champion.fitness;
                }

                // Early stopping if we found a good enough solution
                if let Some(target) = self.early_stop_fitness {
                    if champion.fitness >= target {
                        break;
                    }
                }
            }
        }

        TrialResult {
            config,
            best_fitness,
            generations: generations_run,
        }
    }

    /// Run the hyperparameter optimization
    pub fn optimize(&self, environment: &dyn Environment) -> OptimizationResult {
        let mut trials = Vec::with_capacity(self.num_trials);
        let mut best_config = MutationConfig::default();
        let mut best_fitness = f64::NEG_INFINITY;

        if self.verbose {
            println!("Starting hyperparameter optimization...");
            println!("  Search space parameters: {}", self.search_space.num_parameters());
            println!("  Trials: {}", self.num_trials);
            println!("  Generations per trial: {}", self.generations_per_trial);
            println!();
        }

        for trial_num in 0..self.num_trials {
            let config = self.sample_config();

            if self.verbose {
                println!("Trial {}/{}", trial_num + 1, self.num_trials);
                println!("  add_connection: {:.4}", config.add_connection_rate);
                println!("  add_neuron: {:.4}", config.add_neuron_rate);
                println!("  weight_mutation: {:.4}", config.weight_mutation_rate);
            }

            let result = self.run_trial(config, environment);

            if self.verbose {
                println!("  best_fitness: {:.2} (gen {})", result.best_fitness, result.generations);
                println!();
            }

            if result.best_fitness > best_fitness {
                best_fitness = result.best_fitness;
                best_config = result.config;
            }

            trials.push(result);
        }

        if self.verbose {
            println!("Optimization complete!");
            println!("Best configuration:");
            println!("  add_connection: {:.4}", best_config.add_connection_rate);
            println!("  add_neuron: {:.4}", best_config.add_neuron_rate);
            println!("  weight_mutation: {:.4}", best_config.weight_mutation_rate);
            println!("  best_fitness: {:.2}", best_fitness);
        }

        OptimizationResult {
            best_config,
            best_fitness,
            trials,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::organism::Organism;

    struct SimpleEnv;

    impl Environment for SimpleEnv {
        fn test(&self, organism: &mut Organism) -> f64 {
            // Simple fitness based on number of genes
            organism.genome.total_genes() as f64
        }
    }

    #[test]
    fn test_hyper_tuner_creation() {
        let space = SearchSpace::new()
            .add_connection_rate(0.01..=0.10);

        let tuner = HyperTuner::new(space)
            .population_size(50)
            .generations_per_trial(5)
            .num_trials(2)
            .verbose(false);

        assert_eq!(tuner.population_size, 50);
        assert_eq!(tuner.generations_per_trial, 5);
        assert_eq!(tuner.num_trials, 2);
    }

    #[test]
    fn test_sample_config() {
        let space = SearchSpace::new()
            .add_connection_rate(0.05..=0.10)
            .add_neuron_rate(0.02..=0.04);

        let tuner = HyperTuner::new(space);
        let config = tuner.sample_config();

        assert!(config.add_connection_rate >= 0.05);
        assert!(config.add_connection_rate <= 0.10);
        assert!(config.add_neuron_rate >= 0.02);
        assert!(config.add_neuron_rate <= 0.04);
        // Default values for non-specified params
        assert!((config.weight_mutation_rate - 0.90).abs() < 0.001);
    }

    #[test]
    fn test_optimize_runs() {
        let space = SearchSpace::new()
            .add_connection_rate(0.01..=0.05);

        let tuner = HyperTuner::new(space)
            .population_size(10)
            .generations_per_trial(2)
            .num_trials(2)
            .verbose(false);

        let env = SimpleEnv;
        let result = tuner.optimize(&env);

        assert_eq!(result.trials.len(), 2);
        assert!(result.best_fitness > 0.0);
    }
}
