extern crate ctrlc;
extern crate pyo3;
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

const CONFIG_FILE: &str = "best_config.json";

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
        let mut outputs = vec![0.0; 4];
        self.organism.activate(inputs, &mut outputs);
        Ok(outputs)
    }

    fn reset_state(&mut self) {
        self.organism.reset_state();
    }
}

#[pyfunction]
fn create_organism(genes: Vec<(usize, usize, f64, bool, bool)>, neurons_len: usize) -> PyResult<PyOrganism> {
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
}

impl LunarLanderMultiprocess {
    fn new() -> LunarLanderMultiprocess {
        Python::with_gil(|py| {
            let cpus = num_cpus::get();
            // Use 2x more workers than CPUs to reduce idle time
            let workers = cpus * 2;

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
                "lunar_lander_worker"
            ).unwrap();

            // Inject rustneat_py module into worker
            worker_module.call_method1("set_rustneat_module", (rustneat_py_module,)).unwrap();

            // Get the init_worker function for pool initialization
            let init_worker_fn = worker_module.getattr("init_worker").unwrap();

            // Create multiprocessing pool with 2x workers and initializer
            println!("Creating pool with {} workers ({}x CPUs) with persistent environments", workers, workers / cpus);
            let pool = mp.call_method(
                "Pool",
                (workers,),
                Some(&[("initializer", init_worker_fn)].into_py_dict_bound(py))
            ).unwrap().unbind();

            LunarLanderMultiprocess {
                pool,
                worker_module: worker_module.unbind().into(),
            }
        })
    }

    pub fn lunar_lander_test(&self, organism: &mut Organism, render: bool) -> f64 {
        Python::with_gil(|py| {
            let neurons_len = organism.genome.len();

            let genes_list = organism.genome.get_genes()
                .iter()
                .map(|gene| {
                    (
                        gene.in_neuron_id(),
                        gene.out_neuron_id(),
                        gene.weight(),
                        gene.enabled(),
                        gene.is_bias()
                    )
                })
                .collect::<Vec<_>>();

            let builtins = py.import_bound("builtins").unwrap();
            let dict = builtins.call_method0("dict").unwrap();
            dict.set_item("genes", genes_list).unwrap();
            dict.set_item("neurons_len", neurons_len).unwrap();

            let worker_fn = self.worker_module.bind(py).getattr("evaluate_organism").unwrap();
            let pool = self.pool.bind(py);
            let async_result = pool.call_method1("apply", (worker_fn, (dict, render))).unwrap();

            async_result.extract::<f64>().unwrap()
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
            // Prepare batch data only for organisms that need evaluation
            let batch_data: Vec<_> = to_evaluate
                .iter()
                .map(|&idx| {
                    let organism = &organisms[idx];
                    let neurons_len = organism.genome.len();
                    let genes_list = organism.genome.get_genes()
                        .iter()
                        .map(|gene| {
                            (
                                gene.in_neuron_id(),
                                gene.out_neuron_id(),
                                gene.weight(),
                                gene.enabled(),
                                gene.is_bias()
                            )
                        })
                        .collect::<Vec<_>>();

                    let builtins = py.import_bound("builtins").unwrap();
                    let dict = builtins.call_method0("dict").unwrap();
                    dict.set_item("genes", genes_list).unwrap();
                    dict.set_item("neurons_len", neurons_len).unwrap();
                    (dict, false)
                })
                .collect();

            let worker_fn = self.worker_module.bind(py).getattr("evaluate_organism").unwrap();
            let pool = self.pool.bind(py);

            // Use starmap for batch evaluation
            let results = pool.call_method1("starmap", (worker_fn, batch_data)).unwrap();

            // Extract fitness values and update only evaluated organisms
            let results_list: Vec<f64> = results.extract().unwrap();
            for (&idx, fitness) in to_evaluate.iter().zip(results_list.iter()) {
                organisms[idx].fitness = *fitness;
            }
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
    let max_fitness = 800f64;  // 300 + 500 offset

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
        println!("  add_connection_rate: {:.4}", config_file.add_connection_rate);
        println!("  add_neuron_rate: {:.4}", config_file.add_neuron_rate);
        println!("  weight_mutation_rate: {:.4}", config_file.weight_mutation_rate);
        if let Some(fitness) = config_file.best_fitness {
            println!("  (tuned with fitness: {:.2})", fitness);
        }
        println!();
        let config = config_file.to_mutation_config();
        Population::create_population_initialized_with_config(150, 8, 4, config)
    } else {
        println!("No {} found, using default config\n", CONFIG_FILE);
        Population::create_population_initialized(150, 8, 4)
    };

    let environment = LunarLanderMultiprocess::new();
    let mut champion: Option<Organism> = None;
    let mut generations = 0;
    let mut best_fitness = 300.0; // Typical starting fitness (-200 + 500 offset)
    let mut last_verified_fitness = 0.0; // Track last fitness we verified to avoid re-verifying same champion
    let improvement_threshold = 20.0; // Show render when fitness improves by at least 20 points

    println!("Starting evolution...");
    println!("Target fitness: {}", max_fitness);
    println!("Improvement threshold for rendering: +{} points\n", improvement_threshold);

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

            println!("\n[Performance] Gen {}: {:.1} gen/s (avg: {:.1} gen/s, total time: {:.1}min)",
                     generations, gen_per_sec, avg_gen_per_sec, total_elapsed / 60.0);

            last_stats_time = Instant::now();
            last_stats_gen = generations;
        }

        match population.champion {
            Some(_) => {
                let tmp_champion = population.champion.clone().unwrap();
                let current_fitness = tmp_champion.fitness;

                // Print progress every generation
                println!("Gen {}: Best fitness = {:.2} (species: {})",
                         generations, current_fitness, population.species.len());

                // Only verify when fitness exceeds a meaningful threshold AND champion has changed
                // With +500 offset: >550 is decent, >600 is good, >700 is landing
                if current_fitness > 550.0
                    && current_fitness > best_fitness + improvement_threshold
                    && current_fitness > last_verified_fitness
                {
                    last_verified_fitness = current_fitness; // Mark as verified

                    println!("\n=== POTENTIAL IMPROVEMENT: {:.2} -> {:.2} (+{:.2}) ===",
                             best_fitness, current_fitness, current_fitness - best_fitness);
                    println!("Verifying with 5 additional tests...");

                    // Evaluate multiple times to verify it's not just luck
                    let mut total_fitness = current_fitness;
                    for i in 1..=5 {
                        let test_fitness = environment.lunar_lander_test(&mut tmp_champion.clone(), false);
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
