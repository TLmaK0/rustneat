extern crate pyo3;
extern crate rustneat;
extern crate serde;
extern crate serde_json;

use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use rustneat::{Gene, Genome, Organism};
use serde::Deserialize;
use std::fs;

const CHAMPION_FILE: &str = "champion.json";

#[derive(Debug, Deserialize)]
struct ChampionFile {
    fitness: f64,
    generation: usize,
    neurons_len: usize,
    genes: Vec<(usize, usize, f64, bool, bool)>,
}

impl ChampionFile {
    fn load(path: &str) -> Option<Self> {
        let content = fs::read_to_string(path).ok()?;
        serde_json::from_str(&content).ok()
    }

    fn to_organism(&self) -> Organism {
        let gene_vec: Vec<Gene> = self
            .genes
            .iter()
            .map(|(i, o, w, e, b)| Gene::new(*i, *o, *w, *e, *b))
            .collect();
        let last_neuron_id = if self.neurons_len > 0 {
            self.neurons_len - 1
        } else {
            0
        };
        let genome = Genome::from_genes(gene_vec, last_neuron_id);
        let mut org = Organism::new(genome);
        org.tau = 0.1;
        org.step_time = 0.5;
        org
    }
}

fn run_episode(
    organism: &mut Organism,
    py: Python,
    gym: &Bound<PyAny>,
    render_mode: &str,
) -> (f64, Vec<PyObject>) {
    let kwargs = [("render_mode", render_mode)].into_py_dict_bound(py);
    let env = gym
        .call_method("make", ("LunarLander-v3",), Some(&kwargs))
        .expect("Failed to create environment");

    let reset_result = env.call_method0("reset").unwrap();
    let observation: Vec<f64> = reset_result.get_item(0).unwrap().extract().unwrap();

    organism.reset_state();
    let mut obs = observation;
    let mut done = false;
    let mut episode_reward = 0.0;
    let mut steps = 0;
    let max_steps = 500;
    let mut frames: Vec<PyObject> = Vec::new();

    let record = render_mode == "rgb_array";

    while !done && steps < max_steps {
        if record {
            let frame = env.call_method0("render").unwrap();
            frames.push(frame.unbind());
        }

        let mut outputs = vec![0.0f64; 2];
        organism.activate(obs.clone(), &mut outputs);

        let lateral = outputs[1];
        let action = if lateral < 0.33 {
            1
        } else if lateral > 0.66 {
            3
        } else if outputs[0] > 0.5 {
            2
        } else {
            0
        };

        let step_result = env.call_method1("step", (action,)).unwrap();
        obs = step_result.get_item(0).unwrap().extract().unwrap();
        let reward: f64 = step_result.get_item(1).unwrap().extract().unwrap();
        let terminated: bool = step_result.get_item(2).unwrap().extract().unwrap();
        let truncated: bool = step_result.get_item(3).unwrap().extract().unwrap();

        episode_reward += reward;
        done = terminated || truncated;
        steps += 1;
    }

    if record {
        let frame = env.call_method0("render").unwrap();
        frames.push(frame.unbind());
    }

    env.call_method0("close").ok();
    (episode_reward, frames)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let record_mode = args.iter().any(|a| a == "--record");

    println!("Loading champion from {}...", CHAMPION_FILE);

    let champion_data = ChampionFile::load(CHAMPION_FILE).expect("Failed to load champion.json");

    println!("Champion info:");
    println!("  Fitness: {:.2}", champion_data.fitness);
    println!("  Generation: {}", champion_data.generation);
    println!("  Neurons: {}", champion_data.neurons_len);
    println!("  Genes: {}", champion_data.genes.len());
    println!();

    let mut organism = champion_data.to_organism();

    pyo3::prepare_freethreaded_python();

    Python::with_gil(|py| {
        let gym = py
            .import_bound("gymnasium")
            .expect("Failed to import gymnasium");

        if record_mode {
            // Record mode: run multiple attempts, save best as GIF
            println!("Recording mode: running 20 attempts...\n");
            let mut best_reward = f64::NEG_INFINITY;
            let mut best_frames: Vec<PyObject> = Vec::new();

            for i in 1..=20 {
                let (reward, frames) = run_episode(&mut organism, py, &gym, "rgb_array");
                println!("  Attempt {}: reward = {:.2}", i, reward);
                if reward > best_reward {
                    best_reward = reward;
                    best_frames = frames;
                }
            }

            println!(
                "\nBest reward: {:.2} ({} frames)",
                best_reward,
                best_frames.len()
            );
            println!("Saving GIF...");

            let pil = py.import_bound("PIL.Image").expect("Failed to import PIL");
            let mut images: Vec<PyObject> = Vec::new();
            for frame in &best_frames {
                let img = pil.call_method1("fromarray", (frame.bind(py),)).unwrap();
                images.push(img.unbind());
            }

            if !images.is_empty() {
                let first = images[0].bind(py);
                let rest: Vec<&Bound<PyAny>> = images[1..].iter().map(|i| i.bind(py)).collect();
                let save_kwargs = [
                    ("save_all", true.into_py(py).into_bound(py)),
                    (
                        "append_images",
                        pyo3::types::PyList::new_bound(py, &rest).into_any(),
                    ),
                    ("duration", 33i32.into_py(py).into_bound(py)),
                    ("loop", 0i32.into_py(py).into_bound(py)),
                ]
                .into_py_dict_bound(py);

                let output_path = "docs/results/lunar_lander.gif";
                first
                    .call_method("save", (output_path,), Some(&save_kwargs))
                    .unwrap();
                println!("Saved to {}", output_path);
            }
        } else {
            // Normal test mode
            println!("Running 10 test episodes...\n");
            let mut total_reward = 0.0;
            let num_tests = 10;

            for i in 1..=num_tests {
                let (reward, _) = run_episode(&mut organism, py, &gym, "");
                let fitness = reward + 500.0;
                total_reward += fitness;
                println!(
                    "Test {}/{}: Fitness = {:.2} (reward = {:.2})",
                    i, num_tests, fitness, reward
                );
            }

            let avg_fitness = total_reward / num_tests as f64;
            println!("\n=== RESULTS ===");
            println!("Average fitness: {:.2}", avg_fitness);
            println!("Stored fitness: {:.2}", champion_data.fitness);

            // Visual demo
            println!("\nRunning visual demo...");
            let (reward, _) = run_episode(&mut organism, py, &gym, "human");
            println!("Visual demo fitness: {:.2}", reward + 500.0);
        }
    });
}
