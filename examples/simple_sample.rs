extern crate rand;
extern crate rustneat;

use rustneat::Environment;
use rustneat::Organism;
use rustneat::Population;

#[cfg(feature = "telemetry")]
mod telemetry_helper;

const PATTERNS: [([f64; 2], f64); 4] = [
    ([0f64, 0f64], 0f64),
    ([0f64, 1f64], 1f64),
    ([1f64, 0f64], 1f64),
    ([1f64, 1f64], 0f64),
];

/// XOR as four independent cases: `activate` keeps the CTRNN state between
/// calls, so it is reset before each pattern.
struct XORClassification;

impl Environment for XORClassification {
    fn test(&self, organism: &mut Organism) -> f64 {
        let mut output = vec![0f64];
        let mut distance = 0f64;
        for (inputs, expected) in PATTERNS.iter() {
            organism.reset_state();
            organism.activate(inputs.to_vec(), &mut output);
            distance += (expected - output[0]).powi(2);
        }
        16f64 / (1f64 + distance)
    }
}

fn main() {
    #[cfg(feature = "telemetry")]
    telemetry_helper::enable_telemetry("?max_fitness=18", true);

    #[cfg(feature = "telemetry")]
    std::thread::sleep(std::time::Duration::from_millis(2000));

    let mut population = Population::create_population_initialized(150, 2, 1);
    let environment = XORClassification;
    let mut champion: Option<Organism> = None;
    while champion.is_none() {
        population.evolve();
        population
            .evaluate_in(&environment)
            .expect("the environment must return a finite fitness");
        for organism in &population.get_organisms() {
            if organism.fitness > 15.5f64 {
                champion = Some(organism.clone());
            }
        }
    }
    let mut champion = champion.unwrap();
    println!("{:?}", champion.genome);
    for (inputs, expected) in PATTERNS.iter() {
        let mut output = vec![0f64];
        champion.reset_state();
        champion.activate(inputs.to_vec(), &mut output);
        println!("{:?} -> {:.3} (expected {})", inputs, output[0], expected);
    }
}
