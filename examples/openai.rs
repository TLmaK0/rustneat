//! Cart pole (gymnasium `CartPole-v1`) evolved with NEAT + CTRNN. Needs a
//! Python with `gymnasium` (and `pygame` to render the champion):
//!
//! ```bash
//! pip install "gymnasium[classic-control]"
//! cargo run --release --example openai --features openai
//! ```

extern crate ctrlc;
extern crate pyo3;
extern crate rustneat;

use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use rustneat::{Environment, Organism, Population};
use std::process;

mod cart_pole_common;
use cart_pole_common::{run_episode, Step, ENV_ID, INPUTS, OUTPUTS, SOLVED_REWARD};

#[cfg(feature = "telemetry")]
mod telemetry_helper;

struct CartPole {
    /// Environment the population is evaluated in (no rendering).
    env: Py<PyAny>,
}

impl Environment for CartPole {
    fn test(&self, organism: &mut Organism) -> f64 {
        Python::with_gil(|py| episode(self.env.bind(py), organism))
            .expect("the cart pole episode failed in Python")
    }

    /// Every episode runs in the one Python environment.
    fn threads(&self) -> usize {
        1
    }
}

impl CartPole {
    fn new() -> PyResult<CartPole> {
        Python::with_gil(|py| {
            let env = make_env(py, None)?;
            Ok(CartPole { env: env.unbind() })
        })
    }

    /// Run one episode of `organism` in a window and give its reward.
    fn render(&self, organism: &mut Organism) -> PyResult<f64> {
        Python::with_gil(|py| {
            let env = make_env(py, Some("human"))?;
            let reward = episode(&env, organism);
            env.call_method0("close")?;
            reward
        })
    }

    fn close(&self) -> PyResult<()> {
        Python::with_gil(|py| self.env.bind(py).call_method0("close").map(|_| ()))
    }
}

fn make_env<'py>(py: Python<'py>, render_mode: Option<&str>) -> PyResult<Bound<'py, PyAny>> {
    let gym = py.import_bound("gymnasium")?;
    let kwargs = [("render_mode", render_mode)].into_py_dict_bound(py);
    gym.call_method("make", (ENV_ID,), Some(&kwargs))
}

/// One episode with the gymnasium API: `reset()` gives `(observation,
/// info)` and `step(action)` gives `(observation, reward, terminated,
/// truncated, info)`.
fn episode(env: &Bound<PyAny>, organism: &mut Organism) -> PyResult<f64> {
    let (observation, _info): (Vec<f64>, PyObject) = env.call_method0("reset")?.extract()?;
    run_episode(organism, observation, |action| {
        let (observation, reward, terminated, truncated, _info): (
            Vec<f64>,
            f64,
            bool,
            bool,
            PyObject,
        ) = env.call_method1("step", (action,))?.extract()?;
        Ok(Step {
            observation,
            reward,
            terminated,
            truncated,
        })
    })
}

fn main() {
    if let Err(error) = ctrlc::set_handler(move || {
        println!("Exiting...");
        process::exit(130);
    }) {
        eprintln!("Could not set the Ctrl-C handler: {}", error);
    }

    #[cfg(feature = "telemetry")]
    telemetry_helper::enable_telemetry("?max_fitness=500", true);

    let environment = CartPole::new().unwrap_or_else(|error| {
        eprintln!(
            "Could not create {} (is gymnasium installed?): {}",
            ENV_ID, error
        );
        process::exit(1);
    });
    let mut population = Population::create_population_initialized(150, INPUTS, OUTPUTS);
    let mut generation = 0;
    let mut champion: Option<Organism> = None;
    while champion.is_none() {
        population.evolve();
        population
            .evaluate_in(&environment)
            .expect("the environment must return a finite fitness");
        generation += 1;
        let organisms = population.get_organisms();
        let best = organisms
            .iter()
            .map(|organism| organism.fitness)
            .fold(f64::NEG_INFINITY, f64::max);
        champion = organisms
            .into_iter()
            .find(|organism| organism.fitness > SOLVED_REWARD);
        println!(
            "Gen {}: best reward = {:.1} (species: {})",
            generation,
            best,
            population.species.len()
        );
    }
    environment.close().expect("closing the environment failed");

    let mut champion = champion.unwrap();
    println!(
        "Champion found in generation {}, rendering it...",
        generation
    );
    match environment.render(&mut champion) {
        Ok(reward) => println!("Rendered episode reward: {:.1}", reward),
        Err(error) => eprintln!("Could not render the champion: {}", error),
    }
}
