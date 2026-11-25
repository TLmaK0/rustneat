//! The lunar lander example (issue #94): the Python workers rebuild every
//! organism from `OrganismData` (`examples/lunar_lander_common`), and that
//! rebuilt organism must behave exactly like the population's one; the
//! example's `best_config.json` must keep loading old files and default
//! every new `MutationConfig` parameter to the library default.

#[path = "../examples/lunar_lander_common/mod.rs"]
#[allow(dead_code)]
mod lunar_lander_common;

use lunar_lander_common::{
    ChampionFile, ConfigFile, OrganismData, LANDER_STEP_TIME, LANDER_TAU,
    LANDER_WEIGHT_MUTATE_POWER,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rustneat::{Environment, MutationConfig, Organism, OutputActivation, Population};

const INPUTS: usize = 8;
const OUTPUTS: usize = 2;

/// Deterministic stand-in for the lander: rewards outputs that follow a
/// fixed pattern, so the population grows hidden neurons and moves biases.
struct Pattern;

impl Environment for Pattern {
    fn test(&self, organism: &mut Organism) -> f64 {
        organism.reset_state();
        let mut fitness = 0f64;
        for step in 0..6 {
            let x = step as f64 / 5.0;
            let sensors = (0..INPUTS).map(|i| x - i as f64 * 0.1).collect();
            let mut outputs = vec![0f64; OUTPUTS];
            organism.activate(sensors, &mut outputs);
            fitness -= (outputs[0] - x).powi(2) + (outputs[1] - (1.0 - x)).powi(2);
        }
        fitness
    }

    fn threads(&self) -> usize {
        1
    }
}

/// Activate both organisms with the same random observation sequence and
/// require bit-identical outputs.
fn assert_same_outputs(population: &mut Organism, worker: &mut Organism, seed: u64, context: &str) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    population.reset_state();
    worker.reset_state();
    for step in 0..40 {
        let observation: Vec<f64> = (0..INPUTS).map(|_| rng.gen_range(-1.5..1.5)).collect();
        let mut expected = vec![0f64; OUTPUTS];
        let mut actual = vec![0f64; OUTPUTS];
        population.activate(observation.clone(), &mut expected);
        worker.activate(observation, &mut actual);
        assert_eq!(
            expected, actual,
            "{}: outputs differ at step {}",
            context, step
        );
    }
}

/// Evolve a population with `config` and check every organism against its
/// worker copy.
fn check_population(config: MutationConfig, seed: u64) {
    let mut population =
        Population::create_population_initialized_with_config(60, INPUTS, OUTPUTS, config)
            .with_seed(seed);
    for _ in 0..15 {
        population.evolve();
        population.evaluate_in(&Pattern).unwrap();
    }
    let organisms = population.get_organisms();
    assert!(
        organisms.iter().any(|o| o.genome.len() > INPUTS + OUTPUTS),
        "the test needs organisms with hidden neurons"
    );
    for (i, organism) in organisms.into_iter().enumerate() {
        assert_eq!(
            (organism.tau, organism.dt, organism.step_time),
            (config.tau, config.dt, config.step_time)
        );
        let data = OrganismData::of(&organism);
        let mut worker = data.to_organism().expect("valid organism data");
        let mut original = organism;
        assert_same_outputs(
            &mut original,
            &mut worker,
            i as u64,
            &format!("organism {}", i),
        );
    }
}

#[test]
fn worker_organisms_match_the_population_with_the_lander_config() {
    let config = ConfigFile::default().to_mutation_config();
    // The lander's dynamics differ from Organism::new's defaults, so a
    // worker that ignored them would fail the comparison.
    assert_ne!(
        (config.tau, config.step_time),
        (
            Organism::new(Default::default()).tau,
            Organism::new(Default::default()).step_time
        )
    );
    check_population(config, 1);
}

#[test]
fn worker_organisms_match_the_population_with_every_ctrnn_setting_changed() {
    let mut config = ConfigFile::default().to_mutation_config();
    config.tau = 0.3;
    config.dt = 0.02;
    config.step_time = 0.26;
    config.output_activation = OutputActivation::Linear;
    // τ genes too, so the per-neuron τ travel with the genome
    config.tau_mutation_rate = 0.5;
    check_population(config, 7);
}

#[test]
fn organism_data_survives_a_json_round_trip() {
    let mut population = Population::create_population_initialized_with_config(
        10,
        INPUTS,
        OUTPUTS,
        ConfigFile::default().to_mutation_config(),
    )
    .with_seed(3);
    population.evolve();
    let mut organism = population.get_organisms().remove(0);
    organism.genome.set_tau(INPUTS, 0.37);
    let champion = ChampionFile::from_organism(&organism, 512.5, 42);
    let json = serde_json::to_string(&champion).unwrap();
    let loaded: ChampionFile = serde_json::from_str(&json).unwrap();
    assert_eq!(loaded.organism, champion.organism);
    let mut rebuilt = loaded.organism.to_organism().unwrap();
    assert_same_outputs(&mut organism, &mut rebuilt, 9, "champion");
}

#[test]
fn champions_saved_before_the_ctrnn_settings_load_with_the_lander_dynamics() {
    // Format written before #94: no tau/dt/step_time/output_activation
    let json = r#"{
        "fitness": 600.0, "generation": 12, "neurons_len": 3,
        "genes": [[0, 1, 0.5, true], [0, 2, -1.0, true]],
        "biases": [0.0, 0.1, -0.2], "taus": [null, 0.2, null],
        "input_neurons": 1, "output_neurons": 2
    }"#;
    let champion: ChampionFile = serde_json::from_str(json).unwrap();
    let organism = champion.organism.to_organism().unwrap();
    assert_eq!(organism.tau, LANDER_TAU);
    assert_eq!(organism.step_time, LANDER_STEP_TIME);
    assert_eq!(organism.dt, MutationConfig::default().dt);
    assert_eq!(organism.output_activation, OutputActivation::Sigmoid);
    assert_eq!(organism.genome.tau(1), Some(0.2));
}

#[test]
fn unknown_output_activation_is_an_error() {
    let mut data = OrganismData::of(&Organism::new(Default::default()));
    data.output_activation = "relu".to_string();
    assert!(data.to_organism().is_err());
}

#[test]
fn new_config_parameters_default_to_the_library_defaults() {
    let config = ConfigFile::from_json("{}").unwrap().to_mutation_config();
    let base = MutationConfig::default();
    assert_eq!(
        config.weight_gene_mutation_rate,
        base.weight_gene_mutation_rate
    );
    assert_eq!(
        config.weight_perturbation_rate,
        base.weight_perturbation_rate
    );
    assert_eq!(config.weight_max, base.weight_max);
    assert_eq!(config.bias_mutation_rate, base.bias_mutation_rate);
    assert_eq!(config.bias_gene_mutation_rate, base.bias_gene_mutation_rate);
    assert_eq!(config.bias_perturbation_rate, base.bias_perturbation_rate);
    assert_eq!(config.bias_mutate_power, base.bias_mutate_power);
    assert_eq!(config.bias_max, base.bias_max);
    assert_eq!(
        config.compatibility_excess_coefficient,
        base.compatibility_excess_coefficient
    );
    assert_eq!(
        config.compatibility_disjoint_coefficient,
        base.compatibility_disjoint_coefficient
    );
    assert_eq!(
        config.compatibility_weight_coefficient,
        base.compatibility_weight_coefficient
    );
    assert_eq!(
        config.compatibility_bias_coefficient,
        base.compatibility_bias_coefficient
    );
    assert_eq!(
        config.compatibility_tau_coefficient,
        base.compatibility_tau_coefficient
    );
    assert_eq!(
        config.compatibility_small_genome_genes,
        base.compatibility_small_genome_genes
    );
    assert_eq!(config.target_species, base.target_species);
    assert_eq!(
        config.compatibility_threshold_step,
        base.compatibility_threshold_step
    );
    assert_eq!(
        config.compatibility_threshold_min,
        base.compatibility_threshold_min
    );
    assert_eq!(config.mutation_probability, base.mutation_probability);
    assert_eq!(
        config.disabled_gene_inheritance,
        base.disabled_gene_inheritance
    );
    assert_eq!(config.interspecies_mate_rate, base.interspecies_mate_rate);
    assert_eq!(config.dt, base.dt);
    assert_eq!(config.tau_mutation_rate, base.tau_mutation_rate);
    assert_eq!(config.tau_gene_mutation_rate, base.tau_gene_mutation_rate);
    assert_eq!(config.tau_mutate_power, base.tau_mutate_power);
    assert_eq!(config.tau_min, base.tau_min);
    assert_eq!(config.tau_max, base.tau_max);
    assert_eq!(
        config.species_stagnation_generations,
        base.species_stagnation_generations
    );
    assert_eq!(config.species_elitism, base.species_elitism);
    assert_eq!(config.output_activation, OutputActivation::Sigmoid);
    // The lander's own choices
    assert_eq!(config.weight_mutate_power, LANDER_WEIGHT_MUTATE_POWER);
    assert_eq!(
        (config.tau, config.step_time),
        (LANDER_TAU, LANDER_STEP_TIME)
    );
    // An empty file and no file give the same configuration
    assert_eq!(ConfigFile::from_json("{}").unwrap(), ConfigFile::default());
}

#[test]
fn old_config_files_still_load() {
    // A best_config.json as written before #94, plus a field this version
    // does not know: the values present are kept, the rest are defaults.
    let json = r#"{
        "weight_mutation_rate": 0.7, "add_connection_rate": 0.15,
        "add_neuron_rate": 0.05, "toggle_expression_rate": 0.01,
        "weight_perturbation_rate": 0.9, "bias_mutation_rate": 0.3,
        "bias_mutate_power": 0.4, "compatibility_threshold": 2.5,
        "best_fitness": 640.0, "weight_init_range": 2.0,
        "weight_mutate_power": 0.8, "some_removed_field": 1
    }"#;
    let file = ConfigFile::from_json(json).unwrap();
    assert_eq!(file.best_fitness, Some(640.0));
    let config = file.to_mutation_config();
    assert_eq!(config.weight_mutation_rate, 0.7);
    assert_eq!(config.add_connection_rate, 0.15);
    assert_eq!(config.add_neuron_rate, 0.05);
    assert_eq!(config.toggle_expression_rate, 0.01);
    assert_eq!(config.weight_perturbation_rate, 0.9);
    assert_eq!(config.bias_mutation_rate, 0.3);
    assert_eq!(config.bias_mutate_power, 0.4);
    assert_eq!(config.compatibility_threshold, 2.5);
    assert_eq!(config.weight_init_range, 2.0);
    assert_eq!(config.weight_mutate_power, 0.8);
    assert_eq!(config.tau_max, MutationConfig::default().tau_max);
}

#[test]
fn null_target_species_keeps_the_threshold_fixed() {
    let config = ConfigFile::from_json(r#"{"target_species": null}"#)
        .unwrap()
        .to_mutation_config();
    assert_eq!(config.target_species, None);
}

#[test]
fn a_saved_config_loads_back_unchanged() {
    let file = ConfigFile {
        tau_mutation_rate: 0.1,
        species_elitism: 3,
        ..ConfigFile::default()
    };
    let json = serde_json::to_string_pretty(&file).unwrap();
    assert_eq!(ConfigFile::from_json(&json).unwrap(), file);
}
