//! Code shared by the lunar lander examples (`openai_lunar_lander`,
//! `test_champion`) and by `tests/lunar_lander_tests.rs`. It has no Python
//! in it, so the tests can run it without the `openai` feature.

use rustneat::{Gene, Genome, MutationConfig, Organism, OutputActivation};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Default time constant τ of the lander networks (the library default is
/// 0.01): with τ = 0.1 the neurons keep some memory between the 0.5 s
/// activations instead of jumping to their target every step.
pub const LANDER_TAU: f64 = 0.1;
/// Simulated seconds per activation (one environment step). The library
/// default is 0.1.
pub const LANDER_STEP_TIME: f64 = 0.5;

fn default_tau() -> f64 {
    LANDER_TAU
}
fn default_dt() -> f64 {
    MutationConfig::default().dt
}
fn default_step_time() -> f64 {
    LANDER_STEP_TIME
}
fn default_output_activation() -> String {
    output_activation_name(OutputActivation::Sigmoid).to_string()
}

/// Name of an output activation as stored in the files and sent to the
/// Python workers.
pub fn output_activation_name(activation: OutputActivation) -> &'static str {
    match activation {
        OutputActivation::Sigmoid => "sigmoid",
        OutputActivation::Linear => "linear",
    }
}

/// Inverse of `output_activation_name`.
pub fn parse_output_activation(name: &str) -> Result<OutputActivation, String> {
    match name {
        "sigmoid" => Ok(OutputActivation::Sigmoid),
        "linear" => Ok(OutputActivation::Linear),
        other => Err(format!(
            "unknown output activation {:?} (expected \"sigmoid\" or \"linear\")",
            other
        )),
    }
}

/// A gene as stored in the current format: (in, out, weight, enabled).
pub type GeneData = (usize, usize, f64, bool);

/// A gene as stored in the files: the current 4 fields, or the 5 fields
/// written before #60 removed the per-connection `is_bias` flag.
#[derive(Deserialize)]
#[serde(untagged)]
enum StoredGene {
    Current(usize, usize, f64, bool),
    /// (in, out, weight, enabled, is_bias)
    Pre60(usize, usize, f64, bool, bool),
}

/// Reads genes in either format. `is_bias` is dropped: #60 replaced it with
/// a real bias per neuron, and a champion saved before #60 has no biases,
/// so all its biases load as 0. Such a champion loads and runs, but it does
/// not behave as it did when it was saved (#59 changed the output to
/// σ(y + θ) and #60 the bias), so it should be evolved again.
fn deserialize_genes<'de, D>(deserializer: D) -> Result<Vec<GeneData>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let stored = Vec::<StoredGene>::deserialize(deserializer)?;
    Ok(stored
        .into_iter()
        .map(|gene| match gene {
            StoredGene::Current(in_id, out_id, weight, enabled)
            | StoredGene::Pre60(in_id, out_id, weight, enabled, _) => {
                (in_id, out_id, weight, enabled)
            }
        })
        .collect())
}

/// Everything needed to rebuild an organism that behaves exactly like the
/// original: its genome and the CTRNN settings that are not in the genome
/// (default τ, dt, step time, output activation). It is what the Python
/// workers receive and what `champion.json` stores.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrganismData {
    /// `Genome::len()`: highest neuron id + 1
    pub neurons_len: usize,
    /// (in, out, weight, enabled). Champions saved before #60 have a fifth
    /// field, `is_bias`, which is dropped on load (see `deserialize_genes`).
    #[serde(deserialize_with = "deserialize_genes")]
    pub genes: Vec<GeneData>,
    /// Bias of each neuron, by neuron id
    #[serde(default)]
    pub biases: Vec<f64>,
    /// Time constant τ of each neuron, by neuron id; `null` = default τ.
    /// Missing in champions saved before τ was a gene (all default).
    #[serde(default)]
    pub taus: Vec<Option<f64>>,
    /// Number of input neurons (ids `0..input_neurons`). Missing (0) in
    /// champions saved before it was stored.
    #[serde(default)]
    pub input_neurons: usize,
    /// Number of output neurons (the ids right after the inputs).
    #[serde(default)]
    pub output_neurons: usize,
    /// Default τ of the neurons without τ gene. Missing in champions saved
    /// before it was stored, which were all evolved with `LANDER_TAU`.
    #[serde(default = "default_tau")]
    pub tau: f64,
    /// CTRNN integration step
    #[serde(default = "default_dt")]
    pub dt: f64,
    /// Simulated time per activation
    #[serde(default = "default_step_time")]
    pub step_time: f64,
    /// "sigmoid" or "linear"
    #[serde(default = "default_output_activation")]
    pub output_activation: String,
}

impl OrganismData {
    /// Snapshot of an organism of the population.
    pub fn of(organism: &Organism) -> OrganismData {
        let genome = &organism.genome;
        OrganismData {
            neurons_len: genome.len(),
            genes: genome
                .get_genes()
                .iter()
                .map(|g| (g.in_neuron_id(), g.out_neuron_id(), g.weight(), g.enabled()))
                .collect(),
            biases: (0..genome.len()).map(|id| genome.bias(id)).collect(),
            taus: (0..genome.len()).map(|id| genome.tau(id)).collect(),
            input_neurons: genome.input_neurons(),
            output_neurons: genome.output_neurons(),
            tau: organism.tau,
            dt: organism.dt,
            step_time: organism.step_time,
            output_activation: output_activation_name(organism.output_activation).to_string(),
        }
    }

    /// Rebuild the organism. This is what the Python workers run (through
    /// `create_organism`) and what `test_champion` runs, so both evaluate
    /// the same network dynamics the population evolved.
    pub fn to_organism(&self) -> Result<Organism, String> {
        let genes: Vec<Gene> = self
            .genes
            .iter()
            .map(|&(in_id, out_id, weight, enabled)| Gene::new(in_id, out_id, weight, enabled))
            .collect();
        let last_neuron_id = self.neurons_len.saturating_sub(1);
        let mut genome = Genome::from_genes_with_io(
            genes,
            last_neuron_id,
            self.input_neurons,
            self.output_neurons,
        );
        for (id, bias) in self.biases.iter().enumerate() {
            genome.set_bias(id, *bias);
        }
        for (id, tau) in self.taus.iter().enumerate() {
            if let Some(tau) = tau {
                genome.set_tau(id, *tau);
            }
        }
        let mut organism = Organism::new(genome);
        organism.tau = self.tau;
        organism.dt = self.dt;
        organism.step_time = self.step_time;
        organism.output_activation = parse_output_activation(&self.output_activation)?;
        Ok(organism)
    }
}

/// `champion.json`: the best organism found and when.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChampionFile {
    pub fitness: f64,
    pub generation: usize,
    /// Flattened: the genome fields sit at the top level of the JSON, as in
    /// the champions saved before this struct existed.
    #[serde(flatten)]
    pub organism: OrganismData,
}

impl ChampionFile {
    pub fn from_organism(organism: &Organism, fitness: f64, generation: usize) -> Self {
        ChampionFile {
            fitness,
            generation,
            organism: OrganismData::of(organism),
        }
    }

    pub fn save(&self, path: &str) {
        let content = serde_json::to_string_pretty(self).unwrap();
        fs::write(path, content).expect("Failed to save champion");
        println!("Champion saved to {}", path);
    }

    pub fn load(path: &str) -> Result<Self, String> {
        let content = fs::read_to_string(path).map_err(|e| format!("{}: {}", path, e))?;
        serde_json::from_str(&content).map_err(|e| format!("{}: {}", path, e))
    }
}

/// `best_config.json`: the lander's evolution parameters.
///
/// Every field is optional: a missing field takes its value from
/// `ConfigFile::default()` (so config files written by older versions still
/// load) and unknown fields are ignored. The fields the lander tunes have
/// lander-specific defaults; all the others default to
/// `MutationConfig::default()` (see its documentation for their meaning).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ConfigFile {
    // Lander-specific defaults
    pub weight_mutation_rate: f64,
    pub add_connection_rate: f64,
    pub add_neuron_rate: f64,
    pub toggle_expression_rate: f64,
    pub compatibility_threshold: f64,
    pub weight_init_range: f64,
    /// σ of the gaussian weight perturbation (see `LANDER_WEIGHT_MUTATE_POWER`)
    pub weight_mutate_power: f64,
    /// Default τ of the neurons (`LANDER_TAU`)
    pub tau: f64,
    /// Simulated time per activation (`LANDER_STEP_TIME`)
    pub step_time: f64,
    /// Fitness the config was tuned with (informative only)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub best_fitness: Option<f64>,

    // MutationConfig::default()
    pub weight_perturbation_rate: f64,
    pub weight_gene_mutation_rate: f64,
    pub weight_max: f64,
    pub bias_mutation_rate: f64,
    pub bias_gene_mutation_rate: f64,
    pub bias_perturbation_rate: f64,
    pub bias_mutate_power: f64,
    pub bias_max: f64,
    pub compatibility_excess_coefficient: f64,
    pub compatibility_disjoint_coefficient: f64,
    pub compatibility_weight_coefficient: f64,
    pub compatibility_bias_coefficient: f64,
    pub compatibility_tau_coefficient: f64,
    pub compatibility_small_genome_genes: usize,
    /// `null` keeps the compatibility threshold fixed
    pub target_species: Option<usize>,
    pub compatibility_threshold_step: f64,
    pub compatibility_threshold_min: f64,
    pub mutation_probability: f64,
    pub disabled_gene_inheritance: f64,
    pub interspecies_mate_rate: f64,
    pub crossover_weight_averaging_rate: f64,
    pub survival_threshold: f64,
    pub champion_copy_min_offspring: usize,
    pub split_input_weight: f64,
    pub split_bias: f64,
    /// Range `[lo, hi]` of the observations a split from an input is fit
    /// to (see `MutationConfig::split_input_range`), as a JSON array
    pub split_input_range: (f64, f64),
    pub dt: f64,
    pub tau_mutation_rate: f64,
    pub tau_gene_mutation_rate: f64,
    pub tau_mutate_power: f64,
    pub tau_min: f64,
    pub tau_max: f64,
    pub species_stagnation_generations: usize,
    pub species_elitism: usize,
    /// Evaluate the elites again every generation and average their
    /// evaluations, so a lucky episode does not keep an organism as elite
    /// and champion (see `MutationConfig::reevaluate_elites`). The lander
    /// is noisy (random terrain and initial push), so this is where it
    /// matters; off by default like in the library.
    pub reevaluate_elites: bool,
}

/// σ of the gaussian weight perturbation `N(0, σ²)` for the lander.
///
/// Until #64 `weight_mutate_power = 1.5` meant a uniform perturbation in
/// [-1.5, 1.5], whose standard deviation is 1.5/√3 ≈ 0.87. Since #64 it is
/// the σ of a gaussian, so 1.5 would make every perturbation ≈ 1.7× (2.6×
/// in variance) larger than the one the lander was tuned with. 0.87 keeps
/// the old spread. It also fits the scale of the problem: the observations
/// are mostly in [-1, 1] and an output switches action band (σ(y+θ) at
/// 0.33 / 0.5 / 0.66) when y moves by ≈ 0.7, so a typical step of 0.87 on
/// one weight can change the chosen action without saturating the output,
/// while a σ of 1.5 (3σ ≈ 4.5) would often saturate it.
pub const LANDER_WEIGHT_MUTATE_POWER: f64 = 0.87;

/// Fraction of each species that reproduces in the lander: the fittest
/// half, the library default the lander was tuned with before #108 lowered
/// it to 0.2 (measured on XOR only; the lander is noisy and was not
/// measured again, so it keeps its value).
pub const LANDER_SURVIVAL_THRESHOLD: f64 = 0.5;

impl Default for ConfigFile {
    fn default() -> Self {
        let base = MutationConfig::default();
        ConfigFile {
            weight_mutation_rate: 0.80,
            add_connection_rate: 0.20,
            add_neuron_rate: 0.10,
            toggle_expression_rate: 0.005,
            compatibility_threshold: 1.5,
            weight_init_range: 3.0,
            weight_mutate_power: LANDER_WEIGHT_MUTATE_POWER,
            tau: LANDER_TAU,
            step_time: LANDER_STEP_TIME,
            best_fitness: None,

            weight_perturbation_rate: base.weight_perturbation_rate,
            weight_gene_mutation_rate: base.weight_gene_mutation_rate,
            weight_max: base.weight_max,
            bias_mutation_rate: base.bias_mutation_rate,
            bias_gene_mutation_rate: base.bias_gene_mutation_rate,
            bias_perturbation_rate: base.bias_perturbation_rate,
            bias_mutate_power: base.bias_mutate_power,
            bias_max: base.bias_max,
            compatibility_excess_coefficient: base.compatibility_excess_coefficient,
            compatibility_disjoint_coefficient: base.compatibility_disjoint_coefficient,
            compatibility_weight_coefficient: base.compatibility_weight_coefficient,
            compatibility_bias_coefficient: base.compatibility_bias_coefficient,
            compatibility_tau_coefficient: base.compatibility_tau_coefficient,
            compatibility_small_genome_genes: base.compatibility_small_genome_genes,
            target_species: base.target_species,
            compatibility_threshold_step: base.compatibility_threshold_step,
            compatibility_threshold_min: base.compatibility_threshold_min,
            mutation_probability: base.mutation_probability,
            disabled_gene_inheritance: base.disabled_gene_inheritance,
            interspecies_mate_rate: base.interspecies_mate_rate,
            crossover_weight_averaging_rate: base.crossover_weight_averaging_rate,
            survival_threshold: LANDER_SURVIVAL_THRESHOLD,
            champion_copy_min_offspring: base.champion_copy_min_offspring,
            split_input_weight: base.split_input_weight,
            split_bias: base.split_bias,
            split_input_range: base.split_input_range,
            dt: base.dt,
            tau_mutation_rate: base.tau_mutation_rate,
            tau_gene_mutation_rate: base.tau_gene_mutation_rate,
            tau_mutate_power: base.tau_mutate_power,
            tau_min: base.tau_min,
            tau_max: base.tau_max,
            species_stagnation_generations: base.species_stagnation_generations,
            species_elitism: base.species_elitism,
            reevaluate_elites: base.reevaluate_elites,
        }
    }
}

impl ConfigFile {
    /// `Ok(None)` if the file does not exist; an error if it exists but is
    /// not a valid config (rather than silently running with the defaults).
    pub fn load(path: &str) -> Result<Option<Self>, String> {
        if !Path::new(path).exists() {
            return Ok(None);
        }
        let content = fs::read_to_string(path).map_err(|e| format!("{}: {}", path, e))?;
        Self::from_json(&content)
            .map(Some)
            .map_err(|e| format!("{}: {}", path, e))
    }

    pub fn from_json(json: &str) -> Result<Self, String> {
        serde_json::from_str(json).map_err(|e| e.to_string())
    }

    pub fn to_mutation_config(&self) -> MutationConfig {
        MutationConfig {
            weight_mutation_rate: self.weight_mutation_rate,
            weight_gene_mutation_rate: self.weight_gene_mutation_rate,
            add_connection_rate: self.add_connection_rate,
            add_neuron_rate: self.add_neuron_rate,
            toggle_expression_rate: self.toggle_expression_rate,
            weight_perturbation_rate: self.weight_perturbation_rate,
            weight_max: self.weight_max,
            bias_mutation_rate: self.bias_mutation_rate,
            bias_gene_mutation_rate: self.bias_gene_mutation_rate,
            bias_perturbation_rate: self.bias_perturbation_rate,
            bias_mutate_power: self.bias_mutate_power,
            bias_max: self.bias_max,
            compatibility_threshold: self.compatibility_threshold,
            compatibility_excess_coefficient: self.compatibility_excess_coefficient,
            compatibility_disjoint_coefficient: self.compatibility_disjoint_coefficient,
            compatibility_weight_coefficient: self.compatibility_weight_coefficient,
            compatibility_bias_coefficient: self.compatibility_bias_coefficient,
            compatibility_tau_coefficient: self.compatibility_tau_coefficient,
            compatibility_small_genome_genes: self.compatibility_small_genome_genes,
            target_species: self.target_species,
            compatibility_threshold_step: self.compatibility_threshold_step,
            compatibility_threshold_min: self.compatibility_threshold_min,
            // Not in the JSON config: the library defaults.
            compatibility_threshold_max: MutationConfig::default().compatibility_threshold_max,
            species_assignment: MutationConfig::default().species_assignment,
            mutation_probability: self.mutation_probability,
            mate_only_probability: MutationConfig::default().mate_only_probability,
            mate_selection: MutationConfig::default().mate_selection,
            disabled_gene_inheritance: self.disabled_gene_inheritance,
            interspecies_mate_rate: self.interspecies_mate_rate,
            crossover_weight_averaging_rate: self.crossover_weight_averaging_rate,
            survival_threshold: self.survival_threshold,
            champion_copy_min_offspring: self.champion_copy_min_offspring,
            split_input_weight: self.split_input_weight,
            split_bias: self.split_bias,
            split_input_range: self.split_input_range,
            weight_init_range: self.weight_init_range,
            weight_mutate_power: self.weight_mutate_power,
            tau: self.tau,
            dt: self.dt,
            step_time: self.step_time,
            tau_mutation_rate: self.tau_mutation_rate,
            tau_gene_mutation_rate: self.tau_gene_mutation_rate,
            tau_mutate_power: self.tau_mutate_power,
            tau_min: self.tau_min,
            tau_max: self.tau_max,
            // The workers pick the action with thresholds on (0, 1) outputs
            output_activation: OutputActivation::Sigmoid,
            // Not in the JSON config: the library default.
            integrator: MutationConfig::default().integrator,
            species_stagnation_generations: self.species_stagnation_generations,
            species_elitism: self.species_elitism,
            reevaluate_elites: self.reevaluate_elites,
        }
    }
}
