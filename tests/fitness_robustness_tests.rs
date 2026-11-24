extern crate rustneat;

use rustneat::{Environment, Organism, Population};
use std::sync::atomic::{AtomicUsize, Ordering};

/// XOR score (0..16) moved into [-100, -90]: every fitness is negative.
struct NegativeXor;

fn xor_score(organism: &mut Organism) -> f64 {
    let mut distance = 0f64;
    for &(a, b, expected) in &[(0., 0., 0.), (0., 1., 1.), (1., 0., 1.), (1., 1., 0.)] {
        let mut output = vec![0f64];
        organism.activate(vec![a, b], &mut output);
        distance += (expected - output[0]).powi(2);
    }
    16f64 / (1f64 + distance)
}

impl Environment for NegativeXor {
    fn test(&self, organism: &mut Organism) -> f64 {
        -100f64 + xor_score(organism) * 10f64 / 16f64
    }
    fn threads(&self) -> usize {
        1
    }
}

/// With fitness in [-100, -90] there is a champion from the first
/// evaluation, it never gets worse and it improves over the generations.
#[test]
fn negative_fitness_has_a_champion_that_improves() {
    let mut population = Population::create_population_initialized(150, 2, 1).with_seed(1);
    population.evolve();
    population.evaluate_in(&NegativeXor).unwrap();
    let first = population
        .champion
        .as_ref()
        .expect("negative fitness must still give a champion")
        .fitness;
    assert!(
        (-100f64..=-90f64).contains(&first),
        "first champion {}",
        first
    );
    let mut previous = first;
    for generation in 0..100 {
        population.evolve();
        assert_eq!(population.size(), 150, "generation {}", generation);
        population.evaluate_in(&NegativeXor).unwrap();
        let current = population.champion.as_ref().unwrap().fitness;
        assert!(
            current >= previous,
            "champion got worse in generation {}: {} -> {}",
            generation,
            previous,
            current
        );
        previous = current;
    }
    assert!(
        previous > first,
        "the champion never improved: {} -> {}",
        first,
        previous
    );
    let best_in_population = population
        .get_organisms()
        .iter()
        .map(|organism| organism.fitness)
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        best_in_population <= -90f64,
        "an organism has a fitness the environment never gives: {}",
        best_in_population
    );
}

/// Returns NaN for one organism out of `every` (all of them with 1).
struct SomeNan {
    every: usize,
    calls: AtomicUsize,
}

impl Environment for SomeNan {
    fn test(&self, organism: &mut Organism) -> f64 {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        if call.is_multiple_of(self.every) {
            f64::NAN
        } else {
            xor_score(organism)
        }
    }
    fn threads(&self) -> usize {
        1
    }
}

fn nan_keeps_population_size(every: usize) {
    let environment = SomeNan {
        every,
        calls: AtomicUsize::new(0),
    };
    let mut population = Population::create_population_initialized(20, 2, 1).with_seed(1);
    for generation in 0..30 {
        population.evolve();
        assert_eq!(population.size(), 20, "generation {}", generation);
        let error = population
            .evaluate_in(&environment)
            .expect_err("a NaN fitness must be reported");
        let message = error.to_string();
        assert!(message.contains("NaN"), "unclear message: {}", message);
        assert_eq!(population.size(), 20, "generation {}", generation);
    }
}

#[test]
fn some_nan_fitness_returns_an_error_and_keeps_population_size() {
    // 8 of every 20 organisms (more or less) get NaN.
    nan_keeps_population_size(3);
}

#[test]
fn all_nan_fitness_returns_an_error_and_keeps_population_size() {
    nan_keeps_population_size(1);
}

/// Deterministic and never 0, so an organism that was never evaluated
/// (fitness 0) is told apart from one that was.
struct Shifted;

impl Environment for Shifted {
    fn test(&self, organism: &mut Organism) -> f64 {
        -50f64 + xor_score(organism)
    }
    fn threads(&self) -> usize {
        1
    }
}

#[test]
fn no_preserved_organism_has_an_unevaluated_fitness() {
    let mut population = Population::create_population_initialized(150, 2, 1).with_seed(1);
    for generation in 0..50 {
        population.evolve();
        for organism in population.get_organisms() {
            if organism.preserve_fitness {
                let mut copy = organism.clone();
                copy.reset_state();
                let expected = Shifted.test(&mut copy);
                assert!(
                    organism.is_evaluated(),
                    "generation {}: preserved organism never evaluated (fitness {})",
                    generation,
                    organism.fitness
                );
                assert_eq!(
                    organism.fitness, expected,
                    "generation {}: preserved fitness is not the evaluated one",
                    generation
                );
            }
        }
        population.evaluate_in(&Shifted).unwrap();
    }
}

/// Gives every organism the same fitness, set by the test before each
/// evaluation, as a noisy environment could.
struct Noisy {
    fitness: AtomicUsize,
}

impl Environment for Noisy {
    fn test(&self, _organism: &mut Organism) -> f64 {
        self.fitness.load(Ordering::SeqCst) as f64
    }
    /// A noisy environment re-evaluates everybody, elites included.
    fn test_batch(&self, organisms: &mut [Organism]) {
        for organism in organisms.iter_mut() {
            organism.fitness = self.test(organism);
        }
    }
    fn threads(&self) -> usize {
        1
    }
}

/// The champion is never replaced by a worse organism, and the epochs
/// without improvement count against the historical best.
#[test]
fn champion_is_the_historical_best() {
    let environment = Noisy {
        fitness: AtomicUsize::new(10),
    };
    let mut population = Population::create_population_initialized(30, 2, 1).with_seed(1);
    population.evolve();
    population.evaluate_in(&environment).unwrap();
    assert_eq!(population.champion.as_ref().unwrap().fitness, 10f64);
    assert_eq!(population.epochs_without_improvements(), 0);
    for (generation, fitness) in [1usize, 5, 3, 9].iter().enumerate() {
        environment.fitness.store(*fitness, Ordering::SeqCst);
        population.evolve();
        population.evaluate_in(&environment).unwrap();
        assert_eq!(
            population.champion.as_ref().unwrap().fitness,
            10f64,
            "champion replaced by a worse one (fitness {})",
            fitness
        );
        assert_eq!(population.epochs_without_improvements(), generation + 1);
    }
    environment.fitness.store(11, Ordering::SeqCst);
    population.evolve();
    population.evaluate_in(&environment).unwrap();
    assert_eq!(population.champion.as_ref().unwrap().fitness, 11f64);
    assert_eq!(population.epochs_without_improvements(), 0);
}

struct NegativeXorParallel;

impl Environment for NegativeXorParallel {
    fn test(&self, organism: &mut Organism) -> f64 {
        NegativeXor.test(organism)
    }
    fn threads(&self) -> usize {
        4
    }
}

/// Parallel evaluation finds the same negative champion: the best organism.
#[test]
fn parallel_evaluation_finds_a_negative_champion() {
    let mut population = Population::create_population_initialized(150, 2, 1).with_seed(1);
    population.evolve();
    population.evaluate_in(&NegativeXorParallel).unwrap();
    let best = population
        .get_organisms()
        .iter()
        .map(|organism| organism.fitness)
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(best < 0f64);
    assert_eq!(population.champion.as_ref().unwrap().fitness, best);
}
