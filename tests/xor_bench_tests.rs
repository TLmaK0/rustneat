//! Issue #106: the champion checks of `examples/xor_bench.rs` say in their
//! names whether they reset the CTRNN state between patterns, and the bench
//! has a key for every `MutationConfig` field that tuning (#108) needs.

#[path = "../examples/xor_bench_common/mod.rs"]
#[allow(dead_code)]
mod xor_bench_common;

use rustneat::{Gene, Genome, Integrator, MutationConfig, Organism, OutputActivation};
use std::collections::HashSet;
use xor_bench_common::{
    all_orders, champion_checks, parse_args, summary, RunResult, Xor, SUMMARY_CHECKS, USAGE,
};

fn args(keys: &[&str]) -> Vec<String> {
    ["xor_bench", "1", "10", "init"]
        .iter()
        .chain(keys.iter())
        .map(|s| s.to_string())
        .collect()
}

fn config(keys: &[&str]) -> MutationConfig {
    parse_args(&args(keys)).expect("valid arguments").cfg
}

/// An organism with memory: the output feeds itself back and has a slow τ,
/// so what it answers depends on the patterns it saw before unless its
/// state is reset.
fn stateful_organism() -> Organism {
    let genes = vec![
        Gene::new(0, 2, 3.0, true),
        Gene::new(1, 2, 3.0, true),
        Gene::new(2, 2, 4.0, true),
    ];
    let genome = Genome::from_genes_with_io(genes, 2, 2, 1);
    Organism::with_config(genome, &MutationConfig::builder().tau(0.5).build())
}

#[test]
fn without_keys_the_bench_uses_the_library_defaults() {
    assert_eq!(
        format!("{:?}", config(&[])),
        format!("{:?}", MutationConfig::default())
    );
}

#[test]
fn every_tuning_key_sets_its_config_field() {
    let c = config(&[
        "xavg=0.1",
        "surv=0.3",
        "champ=6",
        "splitw=1",
        "splitb=-0.5",
        "integ=1",
        "step=0.2",
        "out=1",
        "stag=20",
        "elit=1",
        "reeval=1",
        "mutp=0.25",
        "tmax=6",
        "assign=1",
    ]);
    assert_eq!(c.crossover_weight_averaging_rate, 0.1);
    assert_eq!(c.survival_threshold, 0.3);
    assert_eq!(c.champion_copy_min_offspring, 6);
    assert_eq!(c.split_input_weight, 1.0);
    assert_eq!(c.split_bias, -0.5);
    assert_eq!(c.integrator, Integrator::Euler);
    assert_eq!(c.step_time, 0.2);
    assert_eq!(c.output_activation, OutputActivation::Linear);
    assert_eq!(c.species_stagnation_generations, 20);
    assert_eq!(c.species_elitism, 1);
    assert!(c.reevaluate_elites);
    assert_eq!(c.mutation_probability, 0.25);
    assert_eq!(c.compatibility_threshold_max, 6.0);
    assert_eq!(c.species_assignment, rustneat::SpeciesAssignment::Closest);
}

#[test]
fn split_keys_are_independent() {
    let d = MutationConfig::default();
    let c = config(&["splitb=0"]);
    assert_eq!(c.split_input_weight, d.split_input_weight);
    assert_eq!(c.split_bias, 0.0);
    let c = config(&["stag=30"]);
    assert_eq!(c.species_elitism, d.species_elitism);
}

#[test]
fn split_range_keys_set_the_input_range() {
    assert_eq!(config(&["splitlo=-1"]).split_input_range, (-1.0, 1.0));
    assert_eq!(
        config(&["splitlo=-3", "splithi=3"]).split_input_range,
        (-3.0, 3.0)
    );
    // An empty range is an error, not a panic.
    assert!(parse_args(&args(&["splitlo=2"])).is_err());
}

#[test]
fn input_range_keys_present_xor_with_signed_inputs() {
    let bench = parse_args(&args(&[])).unwrap();
    assert_eq!(bench.env.input_range, (0.0, 1.0));
    let bench = parse_args(&args(&["inlo=-1"])).unwrap();
    assert_eq!(bench.env.input_range, (-1.0, 1.0));
    // A linear organism: the output is the first input.
    let genome = Genome::from_genes_with_io(vec![Gene::new(0, 2, 1.0, true)], 2, 2, 1);
    let mut o = Organism::with_config(
        genome,
        &MutationConfig::builder()
            .output_activation(OutputActivation::Linear)
            .build(),
    );
    let out = bench.env.outputs(&mut o, &[0, 1, 2, 3], true);
    for (k, expected) in [-1.0, -1.0, 1.0, 1.0].iter().enumerate() {
        assert!((out[k] - expected).abs() < 1e-3, "{:?}", out);
    }
    assert!(parse_args(&args(&["inlo=1"])).is_err());
}

#[test]
fn champion_elitism_can_be_turned_off() {
    assert_eq!(
        config(&["champ=inf"]).champion_copy_min_offspring,
        usize::MAX
    );
}

#[test]
fn enumerated_keys_reject_unknown_values() {
    for key in ["integ=2", "out=2", "reeval=2", "assign=2"] {
        assert!(parse_args(&args(&[key])).is_err(), "{} accepted", key);
    }
}

#[test]
fn every_key_is_documented_in_the_usage() {
    for key in [
        "xavg", "surv", "champ", "splitw", "splitb", "splitlo", "splithi", "inlo", "inhi", "integ",
        "step", "out", "stag", "elit", "reeval",
    ] {
        assert!(
            USAGE.contains(&format!("`{}`", key)),
            "{} undocumented",
            key
        );
    }
}

#[test]
fn all_orders_are_the_24_permutations() {
    let orders = all_orders();
    assert_eq!(orders.len(), 24);
    let distinct: HashSet<_> = orders.iter().collect();
    assert_eq!(distinct.len(), 24);
    for order in &orders {
        let mut sorted = *order;
        sorted.sort_unstable();
        assert_eq!(sorted, [0, 1, 2, 3]);
    }
}

#[test]
fn checks_with_reset_do_not_depend_on_the_order() {
    let env = Xor {
        bias_input: false,
        reset: true,
        input_range: (0.0, 1.0),
    };
    let mut o = stateful_organism();
    let c = champion_checks(&env, &mut o);
    assert_eq!(c.reversed_reset_each, c.reset_each);
    assert_eq!(c.all_orders_reset_each, c.reset_each);
    // Trained with reset: as trained = reset each.
    assert_eq!(c.as_trained, c.reset_each);
}

#[test]
fn the_reversed_check_without_reset_sees_the_state() {
    let env = Xor {
        bias_input: false,
        reset: true,
        input_range: (0.0, 1.0),
    };
    let mut o = stateful_organism();
    let c = champion_checks(&env, &mut o);
    assert_ne!(c.reversed_no_reset, c.reversed_reset_each);
}

#[test]
fn trained_without_reset_the_reference_check_keeps_the_state() {
    let env = Xor {
        bias_input: false,
        reset: false,
        input_range: (0.0, 1.0),
    };
    let mut o = stateful_organism();
    let c = champion_checks(&env, &mut o);
    assert_ne!(c.as_trained, c.reset_each);
}

#[test]
fn the_summary_names_say_whether_the_state_is_reset() {
    let bench = parse_args(&args(&["reset=1"])).unwrap();
    let solved = RunResult {
        solved_gen: Some(3),
        ..Default::default()
    };
    let line = summary(&bench, &[solved]);
    assert!(!line.contains("still_solved_reversed_order"));
    assert!(!line.contains("reproduced_same_order"));
    for name in SUMMARY_CHECKS {
        assert!(
            line.contains(&format!(" {}=0/1", name)),
            "{} missing in {}",
            name,
            line
        );
        assert!(USAGE.contains(name), "{} undocumented", name);
        assert!(
            name.ends_with("_no_reset")
                || name.ends_with("_reset_each")
                || name.ends_with("_as_trained"),
            "{} does not say whether it resets",
            name
        );
    }
}
