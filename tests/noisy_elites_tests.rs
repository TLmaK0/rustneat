extern crate rand;
extern crate rand_chacha;
extern crate rustneat;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rustneat::{Environment, MutationConfig, Organism, Population};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

/// Every organism is equally good: its fitness is 1 plus seeded noise in
/// [0, 1). The very first evaluation is lucky and gets `LUCKY_BONUS` more,
/// so one organism starts with a score it does not deserve.
struct NoisyWithOneLuckyScore {
    noise: Mutex<ChaCha8Rng>,
    lucky_pending: AtomicBool,
}

const LUCKY_BONUS: f64 = 20.0;

impl NoisyWithOneLuckyScore {
    fn new(seed: u64) -> NoisyWithOneLuckyScore {
        NoisyWithOneLuckyScore {
            noise: Mutex::new(ChaCha8Rng::seed_from_u64(seed)),
            lucky_pending: AtomicBool::new(true),
        }
    }
}

impl Environment for NoisyWithOneLuckyScore {
    fn test(&self, _organism: &mut Organism) -> f64 {
        let noise: f64 = self.noise.lock().unwrap().gen();
        let bonus = if self.lucky_pending.swap(false, Ordering::SeqCst) {
            LUCKY_BONUS
        } else {
            0.0
        };
        1.0 + noise + bonus
    }
    fn threads(&self) -> usize {
        1
    }
}

fn run(reevaluate_elites: bool, generations: usize) -> Population {
    let config = MutationConfig {
        reevaluate_elites,
        ..MutationConfig::default()
    };
    let environment = NoisyWithOneLuckyScore::new(7);
    let mut population =
        Population::create_population_initialized_with_config(30, 2, 1, config).with_seed(1);
    for _ in 0..generations {
        population.evolve();
        population.evaluate_in(&environment).unwrap();
    }
    population
}

/// Without re-evaluation the lucky organism is kept as elite with its
/// lucky score and stays champion for ever (the behaviour #99 is about).
#[test]
fn without_reevaluation_a_lucky_score_stays_champion() {
    let population = run(false, 60);
    let champion = population.champion.as_ref().unwrap();
    assert!(
        champion.fitness > LUCKY_BONUS,
        "champion fitness {}",
        champion.fitness
    );
    assert_eq!(champion.evaluations(), 1);
}

/// With re-evaluation the lucky organism is evaluated again as an elite;
/// its fitness becomes the mean of its evaluations and it loses the
/// champion place to organisms whose score is confirmed.
#[test]
fn with_reevaluation_a_lucky_score_does_not_stay_champion() {
    let population = run(true, 60);
    let champion = population.champion.as_ref().unwrap();
    assert!(
        champion.fitness < 3.0,
        "champion fitness {} still carries the lucky score",
        champion.fitness
    );
    assert!(
        champion.evaluations() >= 2,
        "the champion must be confirmed by a re-evaluation (evaluations: {})",
        champion.evaluations()
    );
    // Every organism is worth 1.5 on average: a confirmed mean is close.
    assert!(champion.fitness > 1.0);
}

/// Re-evaluated elites keep the mean of all their evaluations.
#[test]
fn reevaluated_elites_keep_the_mean_of_their_evaluations() {
    let mut population = run(true, 1);
    let lucky = population.champion.clone().unwrap();
    assert!(lucky.fitness > LUCKY_BONUS);
    assert_eq!(lucky.evaluations(), 1);
    population.evolve();
    let elite = population
        .get_organisms()
        .into_iter()
        .find(|organism| organism.fitness == lucky.fitness)
        .expect("the lucky organism is the elite of its species");
    assert_eq!(elite.evaluations(), 1);
    assert!(!elite.preserve_fitness, "elites must be evaluated again");
    assert_eq!(elite.fitness, lucky.fitness);

    let environment = NoisyWithOneLuckyScore::new(8);
    environment.lucky_pending.store(false, Ordering::SeqCst);
    population.evaluate_in(&environment).unwrap();
    // The other elites were evaluated twice too, but their mean is < 2.
    let elite = population
        .get_organisms()
        .into_iter()
        .filter(|organism| organism.evaluations() == 2)
        .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
        .expect("the elite was evaluated twice");
    let second = elite.fitness * 2.0 - lucky.fitness;
    assert!(
        (1.0..2.0).contains(&second),
        "mean {} is not the mean of {} and a score in [1, 2)",
        elite.fitness,
        lucky.fitness
    );
}

/// Off by default: elites keep their fitness and are not evaluated again.
#[test]
fn reevaluation_is_off_by_default() {
    assert!(!MutationConfig::default().reevaluate_elites);
    let mut population = run(false, 1);
    population.evolve();
    assert!(population
        .get_organisms()
        .iter()
        .any(|organism| organism.preserve_fitness));
}
