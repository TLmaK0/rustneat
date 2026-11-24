extern crate rustneat;

use rustneat::{Environment, MutationConfig, Organism, Population};

/// XOR fitness, evaluated with several threads so the test also covers the
/// parallel evaluator: reproducibility must not depend on thread scheduling.
struct Xor {
    threads: usize,
}

impl Environment for Xor {
    fn test(&self, organism: &mut Organism) -> f64 {
        let mut distance = 0f64;
        for &(a, b, expected) in &[(0., 0., 0.), (0., 1., 1.), (1., 0., 1.), (1., 1., 0.)] {
            let mut output = vec![0f64];
            organism.activate(vec![a, b], &mut output);
            distance += (expected - output[0]).powi(2);
        }
        16f64 / (1f64 + distance)
    }

    fn threads(&self) -> usize {
        self.threads
    }
}

/// Full snapshot of a population: every genome (ids, weights, enabled, bias)
/// and every fitness, in order, plus the champion.
fn snapshot(population: &Population) -> String {
    let organisms: Vec<String> = population
        .get_organisms()
        .iter()
        .map(|o| format!("{:?} fitness={:?}", o.genome, o.fitness))
        .collect();
    let champion = population
        .champion
        .as_ref()
        .map(|c| format!("{:?} {:?}", c.genome, c.fitness));
    format!("{:?}\nchampion={:?}", organisms, champion)
}

fn run(make: &dyn Fn() -> Population, generations: usize, threads: usize) -> String {
    let environment = Xor { threads };
    let mut population = make();
    for _ in 0..generations {
        population.evolve();
        population.evaluate_in(&environment).unwrap();
    }
    snapshot(&population)
}

fn config() -> MutationConfig {
    // Higher structural rates than default so 50 generations exercise
    // add-neuron, add-connection, toggles and crossover.
    MutationConfig::new()
        .add_connection_rate(0.2)
        .add_neuron_rate(0.1)
        .toggle_expression_rate(0.05)
        .build()
}

#[test]
fn same_seed_gives_identical_genomes_after_50_generations() {
    let make =
        || Population::create_population_initialized_with_config(150, 2, 1, config()).with_seed(42);
    assert_eq!(run(&make, 50, 1), run(&make, 50, 1));
}

#[test]
fn same_seed_is_reproducible_with_parallel_evaluation() {
    let make =
        || Population::create_population_initialized_with_config(150, 2, 1, config()).with_seed(7);
    let single = run(&make, 50, 1);
    assert_eq!(single, run(&make, 50, 4));
    assert_eq!(single, run(&make, 50, 8));
}

#[test]
#[allow(deprecated)]
fn same_seed_is_reproducible_for_every_constructor() {
    let unconnected =
        || Population::create_population_unconnected_with_config(50, 2, 1, config()).with_seed(3);
    assert_eq!(run(&unconnected, 20, 2), run(&unconnected, 20, 3));

    let default = || Population::create_population(50).with_seed(3);
    assert_eq!(run(&default, 20, 2), run(&default, 20, 3));

    let initialized = || Population::create_population_initialized(50, 2, 1).with_seed(3);
    assert_eq!(run(&initialized, 20, 2), run(&initialized, 20, 3));
}

#[test]
fn different_seeds_give_different_genomes() {
    let a =
        || Population::create_population_initialized_with_config(150, 2, 1, config()).with_seed(1);
    let b =
        || Population::create_population_initialized_with_config(150, 2, 1, config()).with_seed(2);
    assert_ne!(run(&a, 10, 1), run(&b, 10, 1));
}

#[test]
fn with_seed_keeps_the_minimal_initial_topology() {
    let population =
        Population::create_population_unconnected_with_config(10, 2, 1, config()).with_seed(5);
    assert_eq!(population.size(), 10);
    for organism in population.get_organisms() {
        assert_eq!(organism.genome.total_genes(), 0);
        assert_eq!(organism.genome.len(), 3);
    }

    let population =
        Population::create_population_initialized_with_config(10, 2, 1, config()).with_seed(5);
    assert_eq!(population.size(), 10);
    for organism in population.get_organisms() {
        // inputs -> outputs only, no hidden neurons
        assert_eq!(organism.genome.total_genes(), 2);
        assert_eq!(organism.genome.len(), 3);
    }
}

#[test]
fn unseeded_constructors_still_work_and_are_not_fixed() {
    let make = || Population::create_population_initialized_with_config(50, 2, 1, config());
    assert_ne!(run(&make, 5, 1), run(&make, 5, 1));
}
