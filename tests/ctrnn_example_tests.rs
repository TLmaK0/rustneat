//! Issue #105: `examples/ctrnn.rs` shows that explicit Euler is unstable
//! for h/τ > 2 and that exponential Euler is not, with the demo parameters
//! of `examples/ctrnn_common`.

#[path = "../examples/ctrnn_common/mod.rs"]
#[allow(dead_code)]
mod ctrnn_common;

use ctrnn_common::{
    organism_outputs, step_size_trajectory, ORGANISM_ACTIVATIONS, ORGANISM_TARGET, STEPS,
    STEP_SIZES, STEP_SIZE_FIXED_POINT,
};
use rustneat::Integrator;

fn distances(trajectory: &[f64]) -> Vec<f64> {
    trajectory
        .iter()
        .map(|y| (y - STEP_SIZE_FIXED_POINT).abs())
        .collect()
}

#[test]
fn the_demo_covers_stable_and_unstable_step_sizes() {
    assert!(STEP_SIZES.iter().any(|&h| h < 2.0));
    assert!(STEP_SIZES.iter().any(|&h| h > 2.0));
}

#[test]
fn explicit_euler_diverges_only_above_h_over_tau_two() {
    for &h in STEP_SIZES.iter() {
        let distance = distances(&step_size_trajectory(Integrator::Euler, h));
        assert_eq!(distance.len(), STEPS + 1);
        let (first, last) = (distance[0], distance[STEPS]);
        if h < 2.0 {
            assert!(last < first, "h = {}: {:?}", h, distance);
        } else if h > 2.0 {
            // |1 - h/τ| > 1: the distance grows every step.
            assert!(
                distance.windows(2).all(|pair| pair[1] > pair[0]),
                "h = {}: {:?}",
                h,
                distance
            );
            assert!(last > 2.0 * first, "h = {}: {:?}", h, distance);
        }
    }
    let worst = distances(&step_size_trajectory(Integrator::Euler, 3.0));
    assert!(worst[STEPS] > 1e4, "{:?}", worst);
}

#[test]
fn exponential_euler_stays_bounded_for_every_step_size() {
    for &h in STEP_SIZES.iter() {
        let distance = distances(&step_size_trajectory(Integrator::ExponentialEuler, h));
        assert_eq!(distance.len(), STEPS + 1);
        // Never overshoots: the distance shrinks monotonically.
        assert!(
            distance.windows(2).all(|pair| pair[1] <= pair[0]),
            "h = {}: {:?}",
            h,
            distance
        );
    }
}

#[test]
fn organism_with_explicit_euler_diverges_and_exponential_converges() {
    let euler = organism_outputs(Integrator::Euler);
    let exponential = organism_outputs(Integrator::ExponentialEuler);
    assert_eq!(euler.len(), ORGANISM_ACTIVATIONS);
    assert_eq!(exponential.len(), ORGANISM_ACTIVATIONS);
    let euler_distance: Vec<f64> = euler.iter().map(|o| (o - ORGANISM_TARGET).abs()).collect();
    assert!(
        euler_distance.windows(2).all(|pair| pair[1] > pair[0]),
        "{:?}",
        euler
    );
    assert!(
        euler_distance[ORGANISM_ACTIVATIONS - 1] > 1e6,
        "{:?}",
        euler
    );
    for output in exponential {
        assert!((output - ORGANISM_TARGET).abs() < 1e-3, "{:?}", output);
    }
}

#[test]
fn the_report_is_deterministic() {
    assert_eq!(ctrnn_common::report(), ctrnn_common::report());
}
