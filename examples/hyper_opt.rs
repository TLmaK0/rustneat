extern crate rand;
extern crate rustneat;
#[macro_use]
extern crate blackbox;

use rustneat::{Environment, Organism, Population, NeuralNetwork, Params};

struct XORClassification;

impl Environment for XORClassification {
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

fn run(p: &Params, n_gen: usize) -> f64 {

    let start_genome = NeuralNetwork::with_neurons(3);
    let mut population = Population::create_population_from(start_genome, 150);
    let mut environment = XORClassification;
    let mut champion: Option<Organism> = None;
    for i in 0..n_gen {
        population.evolve(&mut environment, p);
    }

    let mut best_fitness = 0.0;
    for organism in &population.get_organisms() {
        if organism.fitness > best_fitness {
            best_fitness = organism.fitness;
        }
    }
    best_fitness
}

make_optimizer! {
    Configuration {
        c2: f64 = 0.5 .. 1.0,
        c3: f64 = 0.0 .. 0.3
    }

    const N_GEN: usize = 50; // generations per round
    const N_POPULATIONS: usize = 6; // populations per iteration
    let mut p = Params::default();
    p.c2 = c2;
    p.c3 = c3;
    // Take the average of N rounds
    let score = (0..N_POPULATIONS)
        .map(|_| run(&p, N_GEN))
        .sum::<f64>() / N_POPULATIONS as f64;
    println!("Iteration... Score = {}", score);
    score
    
}
fn main() {
    const N_ITER: usize = 30;
    let config = Configuration::random_search(N_ITER);
    println!("Score: {}", config.evaluate());
    println!("Config: {:?}", config);
}
