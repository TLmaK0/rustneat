extern crate ctrlc;
extern crate pyo3;
extern crate rustneat;
extern crate serde;
extern crate serde_json;

use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyModule};
use rustneat::{Environment, HyperTuner, MutationConfig, Organism, SearchSpace};
use serde::{Deserialize, Serialize};
use std::fs;
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

impl From<&MutationConfig> for ConfigFile {
    fn from(config: &MutationConfig) -> Self {
        ConfigFile {
            weight_mutation_rate: config.weight_mutation_rate,
            add_connection_rate: config.add_connection_rate,
            add_neuron_rate: config.add_neuron_rate,
            toggle_expression_rate: config.toggle_expression_rate,
            weight_perturbation_rate: config.weight_perturbation_rate,
            toggle_bias_rate: config.toggle_bias_rate,
            compatibility_threshold: config.compatibility_threshold,
            best_fitness: None,
        }
    }
}

impl ConfigFile {
    fn save(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self).unwrap();
        fs::write(path, json)
    }
}

// Simplified Lunar Lander environment for tuning
struct LunarLanderEnv {
    pool: Py<PyAny>,
    worker_module: Py<PyAny>,
}

impl LunarLanderEnv {
    fn new() -> Self {
        Python::with_gil(|py| {
            let cpus = num_cpus::get();
            let workers = cpus * 2;

            let mp = py.import_bound("multiprocessing").unwrap();

            // Create rustneat_py module
            let module = PyModule::new_bound(py, "rustneat_py").unwrap();
            rustneat_py(&module).unwrap();

            let worker_code = include_str!("lunar_lander_worker.py");
            let worker_module = PyModule::from_code_bound(
                py,
                &worker_code,
                "lunar_lander_worker.py",
                "lunar_lander_worker"
            ).unwrap();

            worker_module.call_method1("set_rustneat_module", (module,)).unwrap();
            let init_worker_fn = worker_module.getattr("init_worker").unwrap();

            println!("Creating pool with {} workers for tuning", workers);
            let pool = mp.call_method(
                "Pool",
                (workers,),
                Some(&[("initializer", init_worker_fn)].into_py_dict_bound(py))
            ).unwrap().unbind();

            LunarLanderEnv {
                pool,
                worker_module: worker_module.unbind().into(),
            }
        })
    }

    fn close(&self) {
        Python::with_gil(|py| {
            let pool = self.pool.bind(py);
            pool.call_method0("close").unwrap();
            pool.call_method0("join").unwrap();
        });
    }
}

impl Environment for LunarLanderEnv {
    fn test(&self, organism: &mut Organism) -> f64 {
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
            let result = pool.call_method1("apply", (worker_fn, (dict, false))).unwrap();
            result.extract::<f64>().unwrap()
        })
    }

    fn test_batch(&self, organisms: &mut [Organism]) {
        if organisms.is_empty() {
            return;
        }

        Python::with_gil(|py| {
            let batch_data: Vec<_> = organisms
                .iter()
                .map(|organism| {
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
            let results = pool.call_method1("starmap", (worker_fn, batch_data)).unwrap();
            let results_list: Vec<f64> = results.extract().unwrap();

            for (organism, fitness) in organisms.iter_mut().zip(results_list.iter()) {
                organism.fitness = *fitness;
            }
        });
    }

    fn threads(&self) -> usize {
        num_cpus::get()
    }
}

// PyO3 bindings
use rustneat::{Gene, Genome};

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
}

#[pyfunction]
fn create_organism(genes: Vec<(usize, usize, f64, bool, bool)>, neurons_len: usize) -> PyResult<PyOrganism> {
    let gene_vec: Vec<Gene> = genes
        .into_iter()
        .map(|(in_id, out_id, weight, enabled, is_bias)| {
            Gene::new(in_id, out_id, weight, enabled, is_bias)
        })
        .collect();

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

fn main() {
    #[allow(unused_must_use)]
    {
        ctrlc::set_handler(move || {
            println!("\nExiting...");
            process::exit(130);
        });
    }

    println!("=== Lunar Lander Hyperparameter Tuner ===\n");

    // Define search space
    let search_space = SearchSpace::new()
        .add_connection_rate(0.01..=0.10)
        .add_neuron_rate(0.01..=0.05)
        .weight_mutation_rate(0.70..=0.95);

    // Create tuner
    let tuner = HyperTuner::new(search_space)
        .population_size(150)
        .input_neurons(8)
        .output_neurons(4)
        .generations_per_trial(100)
        .num_trials(15)
        .early_stop_fitness(700.0)
        .verbose(true);

    // Create environment
    let environment = LunarLanderEnv::new();

    // Run optimization
    let result = tuner.optimize(&environment);

    println!("\n=== Final Results ===");
    println!("Best configuration found:");
    println!("  add_connection_rate: {:.4}", result.best_config.add_connection_rate);
    println!("  add_neuron_rate: {:.4}", result.best_config.add_neuron_rate);
    println!("  weight_mutation_rate: {:.4}", result.best_config.weight_mutation_rate);
    println!("  best_fitness: {:.2}", result.best_fitness);

    println!("\nAll trials:");
    for (i, trial) in result.trials.iter().enumerate() {
        println!("  Trial {}: fitness={:.2}, conn={:.4}, neuron={:.4}",
            i + 1, trial.best_fitness,
            trial.config.add_connection_rate,
            trial.config.add_neuron_rate
        );
    }

    // Save best config to file
    let mut config_file = ConfigFile::from(&result.best_config);
    config_file.best_fitness = Some(result.best_fitness);
    match config_file.save(CONFIG_FILE) {
        Ok(_) => println!("\n✓ Saved best config to {}", CONFIG_FILE),
        Err(e) => eprintln!("\n✗ Failed to save config: {}", e),
    }

    environment.close();
}
