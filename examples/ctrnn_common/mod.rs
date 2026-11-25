//! Demo of `examples/ctrnn.rs` (issue #105): explicit Euler
//! (`Integrator::Euler`) against exponential Euler
//! (`Integrator::ExponentialEuler`, the default), first on a single CTRNN
//! neuron with growing step sizes (after
//! http://www.tinyblueplanet.com/easy/FCSReport.pdf), then on an evolvable
//! `Organism` whose `MutationConfig` chooses the integrator. Tested by
//! `tests/ctrnn_example_tests.rs`. Everything here is deterministic.

use rustneat::{
    Ctrnn, CtrnnNeuralNetwork, Gene, Genome, Integrator, MutationConfig, Organism, OutputActivation,
};
use std::fmt::Write;

/// Step sizes h of the single-neuron demo (τ = 1, so h/τ = h). Explicit
/// Euler is stable for h/τ < 2 and diverges above.
pub const STEP_SIZES: [f64; 8] = [0.1, 0.5, 1.0, 1.5, 2.0, 2.1, 2.5, 3.0];
/// Integration steps of the single-neuron demo for every step size.
pub const STEPS: usize = 15;
/// The single neuron: `τ·dy/dt = -y + w·σ(y + θ) + I` with τ = 1, w = 0,
/// θ = 0, I = 0.5 and y(0) = 1. Its exact solution is
/// `y(t) = 0.5 + 0.5·e^(-t)`, so y tends to the fixed point I = 0.5.
pub const STEP_SIZE_Y0: f64 = 1.0;
/// Fixed point of the single-neuron demo (y = I).
pub const STEP_SIZE_FIXED_POINT: f64 = 0.5;
/// Step size whose first values the report lists.
pub const TRAJECTORY_STEP_SIZE: f64 = 2.5;

/// The values y(0), y(h), …, y(STEPS·h) of the single neuron integrated
/// with `integrator` in steps of `h`, one `Ctrnn::activate_nn` call per step.
pub fn step_size_trajectory(integrator: Integrator, h: f64) -> Vec<f64> {
    let ctrnn = Ctrnn::with_integrator(integrator);
    let mut trajectory = vec![STEP_SIZE_Y0];
    for _ in 0..STEPS {
        let y = *trajectory.last().unwrap();
        let next = ctrnn.activate_nn(
            h,
            h,
            &CtrnnNeuralNetwork {
                y: &[y],
                tau: &[1.0],
                wji: &[0.0],
                theta: &[0.0],
                i: &[STEP_SIZE_FIXED_POINT],
            },
        );
        trajectory.push(next[0]);
    }
    trajectory
}

/// τ of the organism demo: with the default dt = 0.01, dt/τ = 2.5.
pub const ORGANISM_TAU: f64 = 0.004;
/// Calls to `Organism::activate` in the organism demo (10 steps each).
pub const ORGANISM_ACTIVATIONS: usize = 4;
/// The organism's input is 1 and its single connection has weight 1, so its
/// output (read linearly) tends to w·x = 1.
pub const ORGANISM_TARGET: f64 = 1.0;

/// The `MutationConfig` of the organism demo with `integrator`: a
/// population created with it would evolve organisms integrated that way.
pub fn organism_config(integrator: Integrator) -> MutationConfig {
    MutationConfig::builder()
        .tau(ORGANISM_TAU)
        .output_activation(OutputActivation::Linear)
        .integrator(integrator)
        .build()
}

/// Outputs of `ORGANISM_ACTIVATIONS` successive activations, from rest, of
/// the minimal genome input → output (weight 1) with input 1 and the CTRNN
/// settings of `organism_config(integrator)`.
pub fn organism_outputs(integrator: Integrator) -> Vec<f64> {
    let genome = Genome::from_genes_with_io(vec![Gene::new(0, 1, 1.0, true)], 1, 1, 1);
    let mut organism = Organism::with_config(genome, &organism_config(integrator));
    let mut output = vec![0f64];
    (0..ORGANISM_ACTIVATIONS)
        .map(|_| {
            organism.activate(vec![1.0], &mut output);
            output[0]
        })
        .collect()
}

/// The text the example prints.
pub fn report() -> String {
    let mut out = String::new();
    writeln!(
        out,
        "1. One CTRNN neuron (Ctrnn::with_integrator): tau = 1, w = 0, theta = 0, I = {}, y(0) = {}.",
        STEP_SIZE_FIXED_POINT, STEP_SIZE_Y0
    )
    .unwrap();
    writeln!(
        out,
        "   Exact solution y(t) = 0.5 + 0.5 e^(-t) -> 0.5. Explicit Euler multiplies the error"
    )
    .unwrap();
    writeln!(
        out,
        "   by (1 - h/tau) each step: stable only for h/tau < 2. Exponential Euler is exact here."
    )
    .unwrap();
    writeln!(
        out,
        "   Error |y - 0.5| after {} steps of size h (initial error {}):",
        STEPS,
        STEP_SIZE_Y0 - STEP_SIZE_FIXED_POINT
    )
    .unwrap();
    writeln!(
        out,
        "   {:>5}  {:>14}  {:>17}  {:>14}",
        "h/tau", "explicit Euler", "exponential Euler", "exact"
    )
    .unwrap();
    let initial_error = STEP_SIZE_Y0 - STEP_SIZE_FIXED_POINT;
    for &h in STEP_SIZES.iter() {
        let error =
            |integrator| (step_size_trajectory(integrator, h)[STEPS] - STEP_SIZE_FIXED_POINT).abs();
        let euler = error(Integrator::Euler);
        let verdict = if euler > initial_error {
            "  <- explicit Euler diverges"
        } else if euler == initial_error {
            "  <- explicit Euler oscillates forever"
        } else {
            ""
        };
        writeln!(
            out,
            "   {:>5.1}  {:>14.3e}  {:>17.3e}  {:>14.3e}{}",
            h,
            euler,
            error(Integrator::ExponentialEuler),
            initial_error * (-(STEPS as f64) * h).exp(),
            verdict
        )
        .unwrap();
    }
    let h = TRAJECTORY_STEP_SIZE;
    writeln!(out, "   First values of y with h/tau = {}:", h).unwrap();
    for (name, integrator) in [
        ("explicit Euler", Integrator::Euler),
        ("exponential Euler", Integrator::ExponentialEuler),
    ] {
        let values: Vec<String> = step_size_trajectory(integrator, h)
            .iter()
            .take(8)
            .map(|y| format!("{:.3}", y))
            .collect();
        writeln!(out, "     {:>17}: {} ...", name, values.join(", ")).unwrap();
    }
    writeln!(out).unwrap();
    writeln!(
        out,
        "2. Organism with MutationConfig::integrator (what a Population evolves with): genome\n   \
         input -> output (weight 1), input 1, tau = {}, dt = {} (dt/tau = {}), linear output, \
         target {}.",
        ORGANISM_TAU,
        MutationConfig::default().dt,
        MutationConfig::default().dt / ORGANISM_TAU,
        ORGANISM_TARGET
    )
    .unwrap();
    writeln!(
        out,
        "   {:>10}  {:>14}  {:>17}",
        "activation", "explicit Euler", "exponential Euler"
    )
    .unwrap();
    let euler = organism_outputs(Integrator::Euler);
    let exponential = organism_outputs(Integrator::ExponentialEuler);
    for (n, (euler, exponential)) in euler.iter().zip(exponential.iter()).enumerate() {
        writeln!(
            out,
            "   {:>10}  {:>14.3e}  {:>17.3e}",
            n + 1,
            euler,
            exponential
        )
        .unwrap();
    }
    out
}
