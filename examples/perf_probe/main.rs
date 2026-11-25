//! Performance probe of the compiled phenotype (issue #71).
//!
//! For random networks of several sizes it runs the same sequence of
//! activations (persistent state, random sensors, linear outputs so no
//! difference is hidden by the sigmoid) through `Organism::activate` (the
//! compiled phenotype) and through `DenseOrganism` (the dense phenotype it
//! replaced, `reference.rs`), and prints the time per activation of each,
//! the speed-up, the cost of compiling, and the largest difference between
//! their outputs. The last line checks the criterion of #71: with 100
//! neurons and 500 genes the outputs differ by less than 1e-12 and the
//! compiled phenotype is at least 5 times faster.
//!
//! cargo run --release --example perf_probe

mod reference;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use reference::{random_network, random_sensors, DenseOrganism};
use rustneat::{Integrator, Organism, OutputActivation, Phenotype};
use std::hint::black_box;
use std::time::Instant;

const INPUTS: usize = 4;
const OUTPUTS: usize = 2;
const SIZES: [(usize, usize); 5] = [(6, 6), (10, 30), (30, 120), (100, 500), (300, 1500)];
const REPEATS: usize = 3;

/// Activates a network: `activate(network, sensors, outputs)`.
type Activate<T> = dyn Fn(&mut T, Vec<f64>, &mut Vec<f64>);

/// Run `calls` activations, return the outputs of all of them and the best
/// time per call (µs) of `REPEATS` runs, each from a fresh copy and rest.
fn time_activations<T: Clone>(
    prototype: &T,
    sensors: &[Vec<f64>],
    activate: &Activate<T>,
) -> (Vec<f64>, f64) {
    let mut best = f64::INFINITY;
    let mut all = vec![];
    for _ in 0..REPEATS {
        let mut subject = prototype.clone();
        let mut outputs = vec![0f64; OUTPUTS];
        all = Vec::with_capacity(sensors.len() * OUTPUTS);
        let start = Instant::now();
        for input in sensors {
            activate(&mut subject, black_box(input.clone()), &mut outputs);
            all.extend_from_slice(black_box(&outputs));
        }
        let micros = start.elapsed().as_secs_f64() * 1e6 / sensors.len() as f64;
        best = best.min(micros);
    }
    (all, best)
}

fn main() {
    println!(
        "{:>16} {:>8} {:>14} {:>14} {:>8} {:>12} {:>12}",
        "neurons/genes", "calls", "dense µs", "compiled µs", "factor", "compile µs", "max |diff|"
    );
    let mut criterion = None;
    for &(neurons, genes) in &SIZES {
        let mut rng = ChaCha8Rng::seed_from_u64(neurons as u64);
        let genome = random_network(INPUTS, OUTPUTS, neurons, genes, &mut rng);
        let mut organism = Organism::new(genome.clone());
        organism.output_activation = OutputActivation::Linear;
        let dense = DenseOrganism::from_organism(&organism);
        let calls = (4_000_000 / (neurons * neurons)).clamp(100, 20_000);
        let sensors: Vec<Vec<f64>> = (0..calls)
            .map(|_| random_sensors(INPUTS, &mut rng))
            .collect();

        let (dense_out, dense_us) = time_activations(&dense, &sensors, &|d, s, o| d.activate(s, o));
        let (compiled_out, compiled_us) =
            time_activations(&organism, &sensors, &|c, s, o| c.activate(s, o));
        let max_diff = dense_out
            .iter()
            .zip(&compiled_out)
            .map(|(a, b)| (a - b).abs())
            .fold(0f64, f64::max);
        assert!(dense_out.iter().all(|v| v.is_finite()));

        let compilations = 200;
        let start = Instant::now();
        for _ in 0..compilations {
            black_box(Phenotype::compile(
                black_box(&genome),
                INPUTS,
                OUTPUTS,
                organism.tau,
                organism.dt,
                organism.step_time,
                Integrator::ExponentialEuler,
            ));
        }
        let compile_us = start.elapsed().as_secs_f64() * 1e6 / compilations as f64;

        let factor = dense_us / compiled_us;
        println!(
            "{:>16} {:>8} {:>14.2} {:>14.2} {:>7.1}× {:>12.2} {:>12.3e}",
            format!("{}/{}", neurons, genes),
            calls,
            dense_us,
            compiled_us,
            factor,
            compile_us,
            max_diff
        );
        if (neurons, genes) == (100, 500) {
            criterion = Some((factor, max_diff));
        }
    }
    let (factor, max_diff) = criterion.unwrap();
    let pass = factor >= 5.0 && max_diff < 1e-12;
    println!(
        "CRITERION 100 neurons / 500 genes: factor={:.1}× (>= 5) max_diff={:.3e} (< 1e-12) -> {}",
        factor,
        max_diff,
        if pass { "PASS" } else { "FAIL" }
    );
    if !pass {
        std::process::exit(1);
    }
}
