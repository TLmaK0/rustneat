extern crate rand;
extern crate rustneat;

use rustneat::{Environment, Organism, Population, NeuralNetwork};

#[cfg(feature = "telemetry")]
mod telemetry_helper;

struct XORClassification;

impl Environment<NeuralNetwork> for XORClassification {
    fn test(&self, organism: &mut NeuralNetwork) -> f64 {
        let mut output = vec![0f64];
        let mut distance: f64;
        organism.activate(vec![0f64, 0f64], &mut output);
        distance = (0f64 - output[0]).powi(2);
        organism.activate(vec![0f64, 1f64], &mut output);
        distance += (1f64 - output[0]).powi(2);
        organism.activate(vec![1f64, 0f64], &mut output);
        distance += (1f64 - output[0]).powi(2);
        organism.activate(vec![1f64, 1f64], &mut output);
        distance += (0f64 - output[0]).powi(2);

        let fitness = 16.0 / (1.0 + distance);

        fitness
    }
}

fn main() {
    #[cfg(feature = "telemetry")]
    telemetry_helper::enable_telemetry("?max_fitness=16", true);

    #[cfg(feature = "telemetry")]
    std::thread::sleep(std::time::Duration::from_millis(2000));

    let mut population = Population::create_population(150);
    let mut environment = XORClassification;
    let mut champion: Option<Organism<NeuralNetwork>> = None;
    let mut i = 0;
    while champion.is_none() {
        i += 1;
        population.evolve();
        population.evaluate_in(&mut environment);
        let mut best_fitness = 0.0;
        for organism in &population.get_organisms() {
            if organism.fitness > best_fitness {
                best_fitness = organism.fitness;
            }
            if organism.fitness > 15.5 {
                champion = Some(organism.clone());
            }
        }
        if i % 10 == 0 {
            println!("Gen {}: {}", i, best_fitness);
        }
    }
    println!("{:?}", champion.unwrap().genome);
}
