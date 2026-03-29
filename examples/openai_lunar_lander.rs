extern crate ctrlc;
extern crate pyo3;
extern crate rand;
extern crate rustneat;
extern crate serde;
extern crate serde_json;

use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyModule};
use pyo3::PyResult;
use rustneat::{Environment, Gene, Genome, MutationConfig, Organism, Population};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::process;

/// Compute novelty score: average distance to k-nearest neighbors
fn compute_novelty(behavior: &[f64], all_behaviors: &[&Vec<f64>], k: usize) -> f64 {
    let mut distances: Vec<f64> = all_behaviors
        .iter()
        .map(|other| {
            behavior
                .iter()
                .zip(other.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .collect();

    distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Skip distance=0 (self) and take k nearest
    let nearest: Vec<f64> = distances
        .iter()
        .filter(|&&d| d > 1e-10)
        .take(k)
        .cloned()
        .collect();

    if nearest.is_empty() {
        return 0.0;
    }

    nearest.iter().sum::<f64>() / nearest.len() as f64
}

const CONFIG_FILE: &str = "best_config.json";
const CHAMPION_FILE: &str = "champion.json";

/// Serializable champion genome
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChampionFile {
    fitness: f64,
    generation: usize,
    neurons_len: usize,
    genes: Vec<(usize, usize, f64, bool, bool)>, // (in, out, weight, enabled, bias)
}

impl ChampionFile {
    fn save(&self, path: &str) {
        let content = serde_json::to_string_pretty(self).unwrap();
        fs::write(path, content).expect("Failed to save champion");
        println!("Champion saved to {}", path);
    }

    fn from_organism(organism: &Organism, fitness: f64, generation: usize) -> Self {
        let genes = organism
            .genome
            .get_genes()
            .iter()
            .map(|g| {
                (
                    g.in_neuron_id(),
                    g.out_neuron_id(),
                    g.weight(),
                    g.enabled(),
                    g.is_bias(),
                )
            })
            .collect();
        ChampionFile {
            fitness,
            generation,
            neurons_len: organism.genome.len(),
            genes,
        }
    }
}

/// Serializable configuration file
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ConfigFile {
    weight_mutation_rate: f64,
    add_connection_rate: f64,
    add_neuron_rate: f64,
    toggle_expression_rate: f64,
    weight_perturbation_rate: f64,
    toggle_bias_rate: f64,
    compatibility_threshold: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    best_fitness: Option<f64>,
    #[serde(default = "default_weight_init_range")]
    weight_init_range: f64,
    #[serde(default = "default_weight_mutate_power")]
    weight_mutate_power: f64,
}

fn default_weight_init_range() -> f64 {
    3.0
}
fn default_weight_mutate_power() -> f64 {
    1.5
}

impl ConfigFile {
    fn load(path: &str) -> Option<Self> {
        if Path::new(path).exists() {
            let content = fs::read_to_string(path).ok()?;
            serde_json::from_str(&content).ok()
        } else {
            None
        }
    }

    fn to_mutation_config(&self) -> MutationConfig {
        MutationConfig::new()
            .weight_mutation_rate(self.weight_mutation_rate)
            .add_connection_rate(self.add_connection_rate)
            .add_neuron_rate(self.add_neuron_rate)
            .toggle_expression_rate(self.toggle_expression_rate)
            .weight_perturbation_rate(self.weight_perturbation_rate)
            .toggle_bias_rate(self.toggle_bias_rate)
            .compatibility_threshold(self.compatibility_threshold)
            .weight_init_range(self.weight_init_range)
            .weight_mutate_power(self.weight_mutate_power)
            .tau(0.1)
            .step_time(0.5)
            .build()
    }
}

#[cfg(feature = "telemetry")]
mod telemetry_helper;

// ============================================================================
// PyO3 Wrapper for rustneat - exposes Organism to Python workers
// ============================================================================

#[pyclass]
struct PyOrganism {
    organism: Organism,
}

#[pymethods]
impl PyOrganism {
    fn activate(&mut self, inputs: Vec<f64>) -> PyResult<Vec<f64>> {
        let mut outputs = vec![0.0; 2]; // [main_desire, lateral_direction]
        self.organism.activate(inputs, &mut outputs);
        Ok(outputs)
    }

    fn reset_state(&mut self) {
        self.organism.reset_state();
    }
}

#[pyfunction]
fn create_organism(
    genes: Vec<(usize, usize, f64, bool, bool)>,
    neurons_len: usize,
) -> PyResult<PyOrganism> {
    // Reconstruct genes from serialized data
    let gene_vec: Vec<Gene> = genes
        .into_iter()
        .map(|(in_id, out_id, weight, enabled, is_bias)| {
            Gene::new(in_id, out_id, weight, enabled, is_bias)
        })
        .collect();

    // Reconstruct genome with proper neuron count (neurons_len = last_neuron_id + 1)
    let last_neuron_id = if neurons_len > 0 { neurons_len - 1 } else { 0 };
    let genome = Genome::from_genes(gene_vec, last_neuron_id);
    let organism = Organism::new(genome);

    Ok(PyOrganism { organism })
}

#[pymodule]
fn rustneat_py(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyOrganism>()?;
    m.add_function(wrap_pyfunction!(create_organism, m)?)?;
    Ok(())
}

// ============================================================================
// Lunar Lander Environment with Multiprocessing
// ============================================================================

struct LunarLanderMultiprocess {
    pool: Py<PyAny>,
    worker_module: Py<PyAny>,
    /// Novelty archive: stores behavior descriptors from past evaluations
    novelty_archive: std::sync::RwLock<Vec<Vec<f64>>>,
    /// Latest behaviors from most recent batch evaluation
    latest_behaviors: std::sync::RwLock<Vec<Vec<f64>>>,
}

const NOVELTY_ARCHIVE_MAX: usize = 2000;
const NOVELTY_K: usize = 15; // k-nearest neighbors for novelty score
const NOVELTY_WEIGHT: f64 = 0.1; // 0.0 = pure fitness, 1.0 = pure novelty

impl LunarLanderMultiprocess {
    fn new() -> LunarLanderMultiprocess {
        Python::with_gil(|py| {
            let cpus = num_cpus::get();
            // Match workers to CPUs (CPU-bound Box2D simulation)
            let workers = cpus;

            // Import multiprocessing
            let mp = py.import_bound("multiprocessing").unwrap();

            // Create the rustneat_py module inline
            let module = PyModule::new_bound(py, "rustneat_py").unwrap();
            rustneat_py(&module).unwrap();
            let rustneat_py_module = module;

            // Load worker code from external Python file
            let worker_code = include_str!("lunar_lander_worker.py");

            // Create worker module
            let worker_module = PyModule::from_code_bound(
                py,
                &worker_code,
                "lunar_lander_worker.py",
                "lunar_lander_worker",
            )
            .unwrap();

            // Inject rustneat_py module into worker
            worker_module
                .call_method1("set_rustneat_module", (rustneat_py_module,))
                .unwrap();

            // Get the init_worker function for pool initialization
            let init_worker_fn = worker_module.getattr("init_worker").unwrap();

            // Create multiprocessing pool with 2x workers and initializer
            println!(
                "Creating pool with {} workers (1x CPUs) with persistent environments",
                workers
            );
            let pool = mp
                .call_method(
                    "Pool",
                    (workers,),
                    Some(&[("initializer", init_worker_fn)].into_py_dict_bound(py)),
                )
                .unwrap()
                .unbind();

            LunarLanderMultiprocess {
                pool,
                worker_module: worker_module.unbind().into(),
                novelty_archive: std::sync::RwLock::new(Vec::new()),
                latest_behaviors: std::sync::RwLock::new(Vec::new()),
            }
        })
    }

    pub fn lunar_lander_test(&self, organism: &mut Organism, render: bool) -> f64 {
        let (fitness, _behavior) = self.lunar_lander_test_with_behavior(organism, render);
        fitness
    }

    pub fn lunar_lander_test_with_behavior(
        &self,
        organism: &mut Organism,
        render: bool,
    ) -> (f64, Vec<f64>) {
        Python::with_gil(|py| {
            let neurons_len = organism.genome.len();

            let genes_list = organism
                .genome
                .get_genes()
                .iter()
                .map(|gene| {
                    (
                        gene.in_neuron_id(),
                        gene.out_neuron_id(),
                        gene.weight(),
                        gene.enabled(),
                        gene.is_bias(),
                    )
                })
                .collect::<Vec<_>>();

            let builtins = py.import_bound("builtins").unwrap();
            let dict = builtins.call_method0("dict").unwrap();
            dict.set_item("genes", genes_list).unwrap();
            dict.set_item("neurons_len", neurons_len).unwrap();

            let worker_fn = self
                .worker_module
                .bind(py)
                .getattr("evaluate_organism")
                .unwrap();
            let pool = self.pool.bind(py);
            let async_result = pool
                .call_method1("apply", (worker_fn, (dict, render)))
                .unwrap();

            let result: (f64, Vec<f64>) = async_result.extract().unwrap();
            result
        })
    }

    fn close(&self) {
        Python::with_gil(|py| {
            let pool = self.pool.bind(py);
            pool.call_method0("close").unwrap();
            pool.call_method0("join").unwrap();
        });
    }

    fn threads() -> usize {
        num_cpus::get()
    }
}

const EVALS_PER_ORGANISM: usize = 1;

impl Environment for LunarLanderMultiprocess {
    fn test(&self, organism: &mut Organism) -> f64 {
        self.lunar_lander_test(organism, false)
    }

    fn test_batch(&self, organisms: &mut [Organism]) {
        if organisms.is_empty() {
            return;
        }

        // Collect indices of organisms that need evaluation (skip elite copies)
        let to_evaluate: Vec<usize> = organisms
            .iter()
            .enumerate()
            .filter(|(_, org)| !org.preserve_fitness)
            .map(|(i, _)| i)
            .collect();

        if to_evaluate.is_empty() {
            return;
        }

        Python::with_gil(|py| {
            let mut batch_data: Vec<_> = Vec::with_capacity(to_evaluate.len() * EVALS_PER_ORGANISM);

            for &idx in &to_evaluate {
                let organism = &organisms[idx];
                let neurons_len = organism.genome.len();
                let genes_list = organism
                    .genome
                    .get_genes()
                    .iter()
                    .map(|gene| {
                        (
                            gene.in_neuron_id(),
                            gene.out_neuron_id(),
                            gene.weight(),
                            gene.enabled(),
                            gene.is_bias(),
                        )
                    })
                    .collect::<Vec<_>>();

                for _ in 0..EVALS_PER_ORGANISM {
                    let builtins = py.import_bound("builtins").unwrap();
                    let dict = builtins.call_method0("dict").unwrap();
                    dict.set_item("genes", &genes_list).unwrap();
                    dict.set_item("neurons_len", neurons_len).unwrap();
                    batch_data.push((dict, false));
                }
            }

            let worker_fn = self
                .worker_module
                .bind(py)
                .getattr("evaluate_organism")
                .unwrap();
            let pool = self.pool.bind(py);

            // Extract (fitness, behavior) tuples
            let results = pool
                .call_method1("starmap", (worker_fn, batch_data))
                .unwrap();
            let results_list: Vec<(f64, Vec<f64>)> = results.extract().unwrap();

            // Collect behaviors and fitness
            let mut batch_behaviors: Vec<Vec<f64>> = Vec::new();
            let mut fitness_values: Vec<f64> = Vec::new();

            for (i, &idx) in to_evaluate.iter().enumerate() {
                let start = i * EVALS_PER_ORGANISM;
                let mut avg_fitness = 0.0;
                let mut avg_behavior = vec![0.0; 8];
                for j in start..start + EVALS_PER_ORGANISM {
                    avg_fitness += results_list[j].0;
                    for (k, v) in results_list[j].1.iter().enumerate() {
                        if k < avg_behavior.len() {
                            avg_behavior[k] += v;
                        }
                    }
                }
                avg_fitness /= EVALS_PER_ORGANISM as f64;
                for v in &mut avg_behavior {
                    *v /= EVALS_PER_ORGANISM as f64;
                }

                fitness_values.push(avg_fitness);
                batch_behaviors.push(avg_behavior);
                // Store raw fitness first, novelty adjustment below
                organisms[idx].fitness = avg_fitness;
            }

            // Compute novelty scores using k-nearest neighbors
            let archive = self.novelty_archive.read().unwrap();
            let all_behaviors: Vec<&Vec<f64>> =
                archive.iter().chain(batch_behaviors.iter()).collect();

            if all_behaviors.len() > NOVELTY_K {
                for (i, &idx) in to_evaluate.iter().enumerate() {
                    let behavior = &batch_behaviors[i];
                    let novelty = compute_novelty(behavior, &all_behaviors, NOVELTY_K);
                    let raw_fitness = fitness_values[i];

                    // Combined fitness: blend raw fitness with novelty
                    // Normalize novelty to similar scale as fitness (~0-800)
                    let novelty_scaled = novelty * 200.0; // scale factor
                    organisms[idx].fitness =
                        (1.0 - NOVELTY_WEIGHT) * raw_fitness + NOVELTY_WEIGHT * novelty_scaled;
                }
            }

            // Add behaviors to archive (randomly sample to keep bounded)
            drop(archive);
            let mut archive = self.novelty_archive.write().unwrap();
            for behavior in &batch_behaviors {
                if archive.len() < NOVELTY_ARCHIVE_MAX {
                    archive.push(behavior.clone());
                } else {
                    // Replace random entry
                    let idx = rand::random::<usize>() % NOVELTY_ARCHIVE_MAX;
                    archive[idx] = behavior.clone();
                }
            }

            // Store latest behaviors for logging
            *self.latest_behaviors.write().unwrap() = batch_behaviors;
        });
    }

    fn threads(&self) -> usize {
        LunarLanderMultiprocess::threads()
    }
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    // Fitness is offset by +500 (Lunar Lander returns [-500,300] -> [0,800])
    let max_fitness = 800f64;

    #[allow(unused_must_use)]
    {
        ctrlc::set_handler(move || {
            println!("Exiting...");
            process::exit(130);
        });
    }

    #[cfg(feature = "telemetry")]
    telemetry_helper::enable_telemetry(format!("?max_fitness={}", max_fitness).as_str(), true);

    // Load config from file or use defaults
    let mut population = if let Some(config_file) = ConfigFile::load(CONFIG_FILE) {
        println!("Loaded config from {}", CONFIG_FILE);
        println!(
            "  add_connection_rate: {:.4}",
            config_file.add_connection_rate
        );
        println!("  add_neuron_rate: {:.4}", config_file.add_neuron_rate);
        println!(
            "  weight_mutation_rate: {:.4}",
            config_file.weight_mutation_rate
        );
        if let Some(fitness) = config_file.best_fitness {
            println!("  (tuned with fitness: {:.2})", fitness);
        }
        println!();
        let config = config_file.to_mutation_config();
        println!();
        Population::create_population_initialized_with_config(150, 8, 2, config)
    } else {
        println!("No {} found, using default config\n", CONFIG_FILE);
        let mut config = MutationConfig::default();
        config.add_connection_rate = 0.20;
        config.add_neuron_rate = 0.10;
        config.weight_mutation_rate = 0.80;
        config.toggle_expression_rate = 0.005;
        config.compatibility_threshold = 1.5;
        config.weight_init_range = 3.0;
        config.weight_mutate_power = 1.5;
        config.tau = 0.1;
        config.step_time = 0.5;
        Population::create_population_initialized_with_config(150, 8, 2, config)
    };

    let environment = LunarLanderMultiprocess::new();
    let mut champion: Option<Organism> = None;
    let mut generations = 0;
    let mut best_fitness = 300.0; // Typical starting fitness (-200 + 500 offset)
    let mut last_verified_fitness = 0.0; // Track last fitness we verified to avoid re-verifying same champion
    let improvement_threshold = 10.0; // Show render when fitness improves by at least 10 points

    println!("Starting evolution...");
    println!("Target fitness: {}", max_fitness);
    println!(
        "Improvement threshold for rendering: +{} points\n",
        improvement_threshold
    );

    // Performance tracking
    use std::time::Instant;
    let start_time = Instant::now();
    let mut last_stats_time = Instant::now();
    let mut last_stats_gen = 0;

    while champion.is_none() {
        population.evolve();
        population.evaluate_in(&environment);
        generations += 1;

        // Show performance stats every 50 generations
        if generations % 50 == 0 {
            let elapsed = last_stats_time.elapsed().as_secs_f64();
            let gens_since_last = generations - last_stats_gen;
            let gen_per_sec = gens_since_last as f64 / elapsed;
            let total_elapsed = start_time.elapsed().as_secs_f64();
            let avg_gen_per_sec = generations as f64 / total_elapsed;

            println!(
                "\n[Performance] Gen {}: {:.1} gen/s (avg: {:.1} gen/s, total time: {:.1}min)",
                generations,
                gen_per_sec,
                avg_gen_per_sec,
                total_elapsed / 60.0
            );

            last_stats_time = Instant::now();
            last_stats_gen = generations;
        }

        match population.champion {
            Some(_) => {
                let tmp_champion = population.champion.clone().unwrap();
                let current_fitness = tmp_champion.fitness;

                // Print progress every generation
                println!(
                    "Gen {}: Best fitness = {:.2} (species: {})",
                    generations,
                    current_fitness,
                    population.species.len()
                );

                // Only verify when fitness exceeds a meaningful threshold AND champion has changed
                // With +500 offset: >550 is decent, >600 is good, >700 is landing
                if current_fitness > 550.0
                    && current_fitness > best_fitness + improvement_threshold
                    && current_fitness > last_verified_fitness
                {
                    last_verified_fitness = current_fitness; // Mark as verified

                    println!(
                        "\n=== POTENTIAL IMPROVEMENT: {:.2} -> {:.2} (+{:.2}) ===",
                        best_fitness,
                        current_fitness,
                        current_fitness - best_fitness
                    );
                    println!("Verifying with 5 additional tests...");

                    // Evaluate multiple times to verify it's not just luck
                    let mut total_fitness = current_fitness;
                    for i in 1..=5 {
                        let test_fitness =
                            environment.lunar_lander_test(&mut tmp_champion.clone(), false);
                        println!("  Test {}/5: {:.2}", i, test_fitness);
                        total_fitness += test_fitness;
                    }
                    let average_fitness = total_fitness / 6.0;

                    println!("Average fitness: {:.2}", average_fitness);

                    // Only count as real improvement if average is better
                    if average_fitness > best_fitness + improvement_threshold {
                        println!("✓ CONFIRMED IMPROVEMENT! Rendering best attempt...\n");
                        environment.lunar_lander_test(&mut tmp_champion.clone(), true);
                        best_fitness = average_fitness;

                        // Save champion genome
                        let champion_data = ChampionFile::from_organism(
                            &tmp_champion,
                            current_fitness,
                            generations,
                        );
                        champion_data.save(CHAMPION_FILE);
                    } else {
                        println!("✗ Not consistent enough. Continuing evolution...\n");
                    }
                }

                if tmp_champion.fitness >= max_fitness {
                    println!("\n=== TARGET FITNESS REACHED! ===");
                    println!("Verifying with second test...\n");
                    //check again. At least 2 successfully landing
                    if environment.lunar_lander_test(&mut tmp_champion.clone(), true) >= max_fitness
                    {
                        champion = Some(tmp_champion);
                    }
                }
            }
            None => {
                println!("Gen {}: No champion yet", generations);
            }
        }
    }

    let result = champion.unwrap();
    environment.lunar_lander_test(&mut result.clone(), true);
    println!("{:?}", result.genome);
    environment.close();
}
