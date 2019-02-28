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
        population.evolve(&mut environment);
        let mut best_fitness = 0.0;
        let mut best_organism = None;
        for organism in &population.get_organisms() {
            if organism.fitness > best_fitness {
                best_fitness = organism.fitness;
                best_organism = Some(organism.clone());
            }
            if organism.fitness > 15.5 {
                champion = Some(organism.clone());
            }
        }
        if i % 50 == 0 {
            let best_organism = best_organism.unwrap().genome;
            println!("Gen {}: {}", i, best_fitness);
            // println!(" - Genome: {:?}", best_organism);
            println!(" - {} neurons, {} connections", best_organism.n_neurons(), best_organism.n_connections());

            { // print the test
                let mut organism = best_organism.clone();
                let mut output = vec![0f64];
                let mut distance: f64;
                organism.activate(vec![0f64, 0f64], &mut output);
                distance = (0f64 - output[0]).powi(2);
                println!(" - [0, 0]: {}", output[0]);
                organism.activate(vec![0f64, 1f64], &mut output);
                distance += (1f64 - output[0]).powi(2);
                println!(" - [0, 1]: {}", output[0]);
                organism.activate(vec![1f64, 0f64], &mut output);
                distance += (1f64 - output[0]).powi(2);
                println!(" - [1, 0]: {}", output[0]);
                organism.activate(vec![1f64, 1f64], &mut output);
                distance += (0f64 - output[0]).powi(2);
                println!(" - [1, 1]: {}", output[0]);
            }

            // Print weights, biases ETC
            println!("Weights:");
            print_table(&best_organism.get_weights());
            println!("Biases: {:?}", best_organism.get_bias());
            println!("Enabled: {:?}", best_organism.get_enabled());
        }
    }
    println!("{:?}", champion.unwrap().genome);
}
fn print_table(table: &Vec<f64>) {
    let n = (table.len() as f64).sqrt() as usize;
    for i in 0..n {
        for j in 0..n {
            // Print with precision 3, and ensure width of 6 with spaces
            print!("{: >8.3}", table[i*n + j]);
        }
        println!("");
    }
}
