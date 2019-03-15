extern crate rand;
extern crate rustneat;

use rustneat::{Environment, Organism, Population, NeuralNetwork, Params};

#[cfg(feature = "telemetry")]
mod telemetry_helper;

struct XORClassification;

impl Environment for XORClassification {
    fn test(&self, organism: &mut NeuralNetwork) -> f64 {
        let mut output = vec![0f64];
        let mut distance: f64;
        organism.activate(vec![0f64, 0f64], &mut output);
        distance = (0f64 - output[0]).abs();
        organism.activate(vec![0f64, 1f64], &mut output);
        distance += (1f64 - output[0]).abs();
        organism.activate(vec![1f64, 0f64], &mut output);
        distance += (1f64 - output[0]).abs();
        organism.activate(vec![1f64, 1f64], &mut output);
        distance += (0f64 - output[0]).abs();

        let fitness = 4.0 - distance;
        if fitness < 0.0 {
            0.0
        } else  { 
            fitness.powf(2.0)
        }
    }
}

fn main() {
    #[cfg(feature = "telemetry")]
    telemetry_helper::enable_telemetry("?max_fitness=16", true);

    #[cfg(feature = "telemetry")]
    std::thread::sleep(std::time::Duration::from_millis(2000));

    let mut p = Params::optimized_for_xor();
    p.compatibility_threshold = 1.5;

    const MAX_ITERATIONS: usize = 100;
    let mut start_genome = NeuralNetwork::with_neurons(3);
    start_genome.add_connection(0, 2, 0.0);
    start_genome.add_connection(1, 2, 0.0);
    let mut population = Population::create_population_from(start_genome, 150);
    let mut environment = XORClassification;
    let mut champion: Option<Organism> = None;
    let mut best_organism = None;
    let mut best_fitness = 0.0;
    let mut i = 0;
    while champion.is_none() && i < MAX_ITERATIONS {
        i += 1;
        population.evolve(&mut environment, &p);
        for organism in population.get_organisms() {
            if organism.fitness > best_fitness {
                best_fitness = organism.fitness;
                best_organism = Some(organism.clone());
            }
            if organism.fitness > 15.0 {
                champion = Some(organism.clone());
            }
        }
        if i % 1 == 0 {
            // let best_organism = best_organism.unwrap().genome;
            // println!("= Gen {}: {} =", i, best_fitness);
            let specie_stats = population.species.iter_mut()
                .map(|s| (s.organisms.len(), {s.calculate_champion_fitness(); s.champion_fitness()}))
                .collect::<Vec<_>>();
            println!(" - {} species: {:?}", population.species.len(), specie_stats);
            println!("");
        }
    }
    let best_organism = best_organism.unwrap();
    println!("Result: {}", best_fitness);
    println!(" - {} neurons, {} connections",
             best_organism.genome.n_neurons(),
             best_organism.genome.n_connections());
    { // print the test
        let mut organism = best_organism.clone().genome;
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
    // println!("{:?}", champion.unwrap().genome);
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
