//! Long-run profile (issue #101): the scenario of the regression test of #73
//! (`tests/empty_species_regression_tests.rs`: XOR as minus the squared
//! error, target 5 species, stagnation 3 generations with 1 protected
//! species, seed 19, 30 organisms, 1000 generations).
//!
//! It times `evolve` and `evaluate_in` of every generation, prints a table
//! every `every` generations (time per generation of each phase, species,
//! genes and neurons per genome) and ends with one `SUMMARY` line.
//!
//! The `hash` is a digest of every generation (each organism's genes,
//! biases, τ genes, fitness and species id, and the champion fitness): two
//! builds with the same arguments and the same hash ran exactly the same
//! evolution, so a change that only makes it faster must keep it.
//!
//! usage: `long_run_profile [key=value ...]`, keys `seed` (19), `pop` (30),
//! `gens` (1000), `every` (100), `repeat` (1: runs the whole scenario this
//! many times and reports the fastest).
//!
//! cargo run --release --example long_run_profile
extern crate rustneat;

use rustneat::{Environment, MutationConfig, Organism, Population};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};

/// The environment of the #73 regression test.
struct Xor;

impl Environment for Xor {
    fn test(&self, organism: &mut Organism) -> f64 {
        let mut distance = 0f64;
        for &(a, b, expected) in &[(0., 0., 0.), (0., 1., 1.), (1., 0., 1.), (1., 1., 0.)] {
            organism.reset_state();
            let mut output = vec![0f64];
            organism.activate(vec![a, b], &mut output);
            distance += (expected - output[0]).powi(2);
        }
        -distance
    }
}

struct Args {
    seed: u64,
    pop: usize,
    gens: usize,
    every: usize,
    repeat: usize,
}

fn parse_args() -> Args {
    let mut args = Args {
        seed: 19,
        pop: 30,
        gens: 1000,
        every: 100,
        repeat: 1,
    };
    for arg in std::env::args().skip(1) {
        let (key, value) = arg
            .split_once('=')
            .unwrap_or_else(|| panic!("expected key=value, got {}", arg));
        let number: u64 = value
            .parse()
            .unwrap_or_else(|_| panic!("{} is not a number", value));
        match key {
            "seed" => args.seed = number,
            "pop" => args.pop = number as usize,
            "gens" => args.gens = number as usize,
            "every" => args.every = (number as usize).max(1),
            "repeat" => args.repeat = (number as usize).max(1),
            _ => panic!("unknown key {}", key),
        }
    }
    args
}

/// Adds the whole population (and its champion fitness) to `hasher`.
fn hash_population(population: &Population, hasher: &mut DefaultHasher) {
    for specie in &population.species {
        specie.id.hash(hasher);
        for organism in &specie.organisms {
            organism.fitness.to_bits().hash(hasher);
            let genome = &organism.genome;
            for gene in genome.get_genes() {
                gene.in_neuron_id().hash(hasher);
                gene.out_neuron_id().hash(hasher);
                gene.weight().to_bits().hash(hasher);
                gene.enabled().hash(hasher);
            }
            for id in genome.neurons() {
                id.hash(hasher);
                genome.bias(id).to_bits().hash(hasher);
                genome.tau(id).map(f64::to_bits).hash(hasher);
            }
        }
    }
    population
        .champion
        .as_ref()
        .map(|champion| champion.fitness.to_bits())
        .hash(hasher);
}

/// Genes and neurons per genome: (mean genes, max genes, mean neurons).
fn genome_sizes(population: &Population) -> (f64, usize, f64) {
    let organisms: Vec<&Organism> = population
        .species
        .iter()
        .flat_map(|specie| specie.organisms.iter())
        .collect();
    let count = organisms.len().max(1) as f64;
    let genes: Vec<usize> = organisms
        .iter()
        .map(|o| o.genome.get_genes().len())
        .collect();
    let neurons: usize = organisms.iter().map(|o| o.genome.neurons().len()).sum();
    (
        genes.iter().sum::<usize>() as f64 / count,
        genes.iter().copied().max().unwrap_or(0),
        neurons as f64 / count,
    )
}

struct Run {
    evolve: Duration,
    evaluate: Duration,
    hash: u64,
    champion: f64,
    rows: Vec<String>,
}

fn run(args: &Args) -> Run {
    let config = MutationConfig::new()
        .target_species(5)
        .species_stagnation(3, 1)
        .build();
    let mut population =
        Population::create_population_initialized_with_config(args.pop, 2, 1, config)
            .with_seed(args.seed);
    let mut hasher = DefaultHasher::new();
    let mut rows = vec![];
    let (mut evolve_total, mut evaluate_total) = (Duration::ZERO, Duration::ZERO);
    let (mut evolve_window, mut evaluate_window) = (Duration::ZERO, Duration::ZERO);

    population.evaluate_in(&Xor).unwrap();
    hash_population(&population, &mut hasher);
    for generation in 1..=args.gens {
        let start = Instant::now();
        population.evolve();
        let evolved = Instant::now();
        population.evaluate_in(&Xor).unwrap();
        let evaluated = Instant::now();
        evolve_window += evolved - start;
        evaluate_window += evaluated - evolved;
        hash_population(&population, &mut hasher);

        if generation % args.every == 0 || generation == args.gens {
            let window = (generation - 1) % args.every + 1;
            let (genes, max_genes, neurons) = genome_sizes(&population);
            rows.push(format!(
                "{:>5} {:>11.1} {:>11.1} {:>7} {:>8.1} {:>8} {:>8.1} {:>12.4}",
                generation,
                evolve_window.as_secs_f64() * 1e6 / window as f64,
                evaluate_window.as_secs_f64() * 1e6 / window as f64,
                population.species.len(),
                genes,
                max_genes,
                neurons,
                population.champion.as_ref().map_or(f64::NAN, |c| c.fitness),
            ));
            evolve_total += evolve_window;
            evaluate_total += evaluate_window;
            evolve_window = Duration::ZERO;
            evaluate_window = Duration::ZERO;
        }
    }
    Run {
        evolve: evolve_total,
        evaluate: evaluate_total,
        hash: hasher.finish(),
        champion: population.champion.as_ref().map_or(f64::NAN, |c| c.fitness),
        rows,
    }
}

fn main() {
    let args = parse_args();
    let mut best: Option<Run> = None;
    for _ in 0..args.repeat {
        let current = run(&args);
        if let Some(previous) = &best {
            assert_eq!(previous.hash, current.hash, "same seed, different run");
        }
        let faster = best
            .as_ref()
            .is_none_or(|b| current.evolve + current.evaluate < b.evolve + b.evaluate);
        if faster {
            best = Some(current);
        }
    }
    let best = best.unwrap();
    println!(
        "{:>5} {:>11} {:>11} {:>7} {:>8} {:>8} {:>8} {:>12}",
        "gen", "evolve_us", "evaluate_us", "species", "genes", "max_gen", "neurons", "champion"
    );
    for row in &best.rows {
        println!("{}", row);
    }
    let total = best.evolve + best.evaluate;
    println!(
        "SUMMARY seed={} pop={} gens={} total_s={:.3} evolve_s={:.3} evaluate_s={:.3} per_gen_ms={:.3} champion={:.6} hash={:016x}",
        args.seed,
        args.pop,
        args.gens,
        total.as_secs_f64(),
        best.evolve.as_secs_f64(),
        best.evaluate.as_secs_f64(),
        total.as_secs_f64() * 1e3 / args.gens as f64,
        best.champion,
        best.hash,
    );
}
