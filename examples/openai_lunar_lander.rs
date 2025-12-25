extern crate ctrlc;
extern crate pyo3;
extern crate rustneat;

use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyModule};
use pyo3::PyResult;
use rustneat::{Environment, Genome, Organism, Population};
use std::process;

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
}

#[pyfunction]
fn create_organism(_genes: Vec<(usize, usize, f64, bool, bool)>, _neurons_len: usize) -> PyResult<PyOrganism> {
    // Create genome with specified neuron count
    // Note: In a full implementation, we would need to reconstruct the genome
    // from the serialized genes. For now, we create a basic organism with default genome.
    let genome = Genome::default();
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

        Python::with_gil(|py| {
            // Prepare batch data for all organisms
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

            // Use starmap for batch evaluation
            let results = pool.call_method1("starmap", (worker_fn, batch_data)).unwrap();

            // Extract fitness values and update organisms
            let results_list: Vec<f64> = results.extract().unwrap();
            for (organism, fitness) in organisms.iter_mut().zip(results_list.iter()) {
                organism.fitness = *fitness;
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
    let max_fitness = 300f64;

    #[allow(unused_must_use)]
    {
        ctrlc::set_handler(move || {
            println!("Exiting...");
            process::exit(130);
        });
    }

    #[cfg(feature = "telemetry")]
    telemetry_helper::enable_telemetry(format!("?max_fitness={}", max_fitness).as_str(), true);

    let mut population = Population::create_population_initialized(100, 8, 4);
    let mut environment = LunarLanderMultiprocess::new();
    let mut champion: Option<Organism> = None;
    let mut generations = 0;
    let mut best_fitness = f64::NEG_INFINITY;
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
        population.evaluate_in(&mut environment);
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
                println!("Gen {}: Best fitness = {:.2}", generations, current_fitness);

                // Render when there's a significant improvement
                if current_fitness - best_fitness >= improvement_threshold {
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
                    if average_fitness - best_fitness >= improvement_threshold {
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
