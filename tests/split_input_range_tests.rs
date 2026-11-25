//! Issue #107: a split of a connection from an input is nearly neutral for
//! the input range given by `MutationConfig::split_input_range` (default
//! [0, 1], the rule of #68 unchanged).
extern crate rand;
extern crate rand_chacha;
extern crate rustneat;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rustneat::{Gene, Genome, InnovationRegistry, MutationConfig, Organism};

#[test]
fn default_input_range_is_the_unit_interval() {
    assert_eq!(MutationConfig::default().split_input_range, (0.0, 1.0));
    let config = MutationConfig::builder()
        .split_input_range(-1.0, 1.0)
        .build();
    assert_eq!(config.split_input_range, (-1.0, 1.0));
}

#[test]
#[should_panic(expected = "split_input_range")]
fn an_empty_input_range_is_rejected() {
    MutationConfig::builder().split_input_range(1.0, 1.0);
}

/// Inputs 0, 1 and output 2 with weights 3 and -2 and output bias 0.5.
fn network() -> Genome {
    let mut genome = Genome::from_genes_with_io(
        vec![Gene::new(0, 2, 3.0, true), Gene::new(1, 2, -2.0, true)],
        2,
        2,
        1,
    );
    genome.set_bias(2, 0.5);
    genome
}

fn split_only(range: (f64, f64)) -> MutationConfig {
    MutationConfig::builder()
        .add_connection_rate(0.0)
        .add_neuron_rate(1.0)
        .weight_mutation_rate(0.0)
        .bias_mutation_rate(0.0)
        .toggle_expression_rate(0.0)
        .split_input_range(range.0, range.1)
        .build()
}

/// Split one random connection of `genome`; returns the split (from, to)
/// and the new neuron.
fn split(genome: &mut Genome, config: &MutationConfig, seed: u64) -> ((usize, usize), usize) {
    let before: Vec<(usize, usize)> = genome
        .get_genes()
        .iter()
        .filter(|g| g.enabled())
        .map(|g| (g.in_neuron_id(), g.out_neuron_id()))
        .collect();
    let mut registry = InnovationRegistry::new(genome.len());
    genome.mutate_with_innovations(config, &mut registry, &mut ChaCha8Rng::seed_from_u64(seed));
    let new = *genome.neurons().last().unwrap();
    let old = before
        .into_iter()
        .find(|&(from, to)| {
            !genome
                .get_genes()
                .iter()
                .any(|g| g.enabled() && g.in_neuron_id() == from && g.out_neuron_id() == to)
        })
        .expect("no connection split");
    (old, new)
}

fn weight(genome: &Genome, from: usize, to: usize) -> f64 {
    genome
        .get_genes()
        .iter()
        .find(|g| g.in_neuron_id() == from && g.out_neuron_id() == to)
        .unwrap()
        .weight()
}

#[test]
fn a_split_of_an_input_connection_is_rescaled_to_the_input_range() {
    let config = split_only((-1.0, 1.0));
    for seed in 0..8 {
        let mut genome = network();
        let ((from, to), new) = split(&mut genome, &config, seed);
        let w = weight(&network(), from, to);
        // s in [-1, 1] -> u = (s + 1) / 2 in [0, 1]:
        // s ≈ -1 + 2·σ(5·u - 2.5) = -1 + 2·σ(2.5·s).
        assert_eq!(weight(&genome, from, new), 2.5);
        assert_eq!(genome.bias(new), 0.0);
        assert_eq!(weight(&genome, new, to), 2.0 * w);
        assert_eq!(genome.bias(to), 0.5 - w);
    }
}

#[test]
fn a_split_of_a_neuron_connection_ignores_the_input_range() {
    // 0 -> 3 -> 2: the source of 3 -> 2 is a CTRNN neuron, whose signal is
    // σ(y + θ) in (0, 1) whatever the inputs.
    let mut genome = Genome::from_genes_with_io(
        vec![Gene::new(0, 3, 1.0, false), Gene::new(3, 2, 4.0, true)],
        3,
        2,
        1,
    );
    genome.set_bias(2, 0.5);
    let config = split_only((-1.0, 1.0));
    let ((from, to), new) = split(&mut genome, &config, 1);
    assert_eq!((from, to), (3, 2));
    assert_eq!(weight(&genome, 3, new), config.split_input_weight);
    assert_eq!(genome.bias(new), config.split_bias);
    assert_eq!(weight(&genome, new, 2), 4.0);
    assert_eq!(genome.bias(2), 0.5);
}

#[test]
fn rescaled_weights_and_biases_stay_within_their_limits() {
    let config = MutationConfig {
        weight_max: 5.0,
        bias_max: 2.0,
        ..split_only((-10.0, 10.0))
    };
    let mut genome = network();
    let ((from, to), new) = split(&mut genome, &config, 3);
    assert!(weight(&genome, new, to).abs() <= 5.0);
    assert!(genome.bias(to).abs() <= 2.0);
    assert!(from < 2);
}

fn output(genome: &Genome, a: f64, b: f64) -> f64 {
    let mut organism = Organism::new(genome.clone());
    let mut out = vec![0f64];
    organism.activate(vec![a, b], &mut out);
    out[0]
}

/// Mean output change over a grid of [lo, hi]^2 when splitting an input
/// connection with `config`.
fn mean_change(lo: f64, hi: f64, config: &MutationConfig) -> f64 {
    let mut sum = 0f64;
    for seed in 0..8 {
        let mut child = network();
        split(&mut child, config, seed);
        for i in 0..=10 {
            for j in 0..=10 {
                let a = lo + (hi - lo) * i as f64 / 10.0;
                let b = lo + (hi - lo) * j as f64 / 10.0;
                sum += (output(&network(), a, b) - output(&child, a, b)).abs();
            }
        }
    }
    sum / (8.0 * 121.0)
}

#[test]
fn splitting_an_input_connection_barely_changes_the_output_for_signed_inputs() {
    for &(lo, hi) in &[(0.0, 1.0), (-1.0, 1.0), (-3.0, 3.0)] {
        let fit = mean_change(lo, hi, &split_only((lo, hi)));
        let unit = mean_change(lo, hi, &split_only((0.0, 1.0)));
        assert!(fit < 0.03, "[{}, {}]: {}", lo, hi, fit);
        if lo < 0.0 {
            // The [0, 1] rule is not neutral for signed inputs.
            assert!(unit > 3.0 * fit, "[{}, {}]: {} vs {}", lo, hi, unit, fit);
        }
    }
}
