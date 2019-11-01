extern crate rand;
extern crate rustneat;

use rustneat::{Environment, NeatParams, NeuralNetwork, Organism, Population};

#[cfg(feature = "telemetry")]
mod telemetry_helper;

struct XORClassification;

impl Environment for XORClassification {
    fn test(&self, organism: &mut NeuralNetwork) -> f64 {
        let nn = organism.make_network();
        let mut output = vec![0f64];
        let mut distance: f64;
        nn.activate(vec![0f64, 0f64], &mut output);
        distance = (0f64 - output[0]).powi(2);
        nn.activate(vec![0f64, 1f64], &mut output);
        distance += (1f64 - output[0]).powi(2);
        nn.activate(vec![1f64, 0f64], &mut output);
        distance += (1f64 - output[0]).powi(2);
        nn.activate(vec![1f64, 1f64], &mut output);
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

    let p = NeatParams::optimized_for_xor3(2, 1);

    const MAX_ITERATIONS: usize = 100;
    let mut start_genome = NeuralNetwork::with_neurons(3);
    start_genome.add_connection(0, 2, 1.0);
    start_genome.add_connection(1, 2, 1.0);
    let mut population = Population::create_population_from(start_genome, 150);
    let mut environment = XORClassification;
    let mut champion: Option<Organism> = None;
    let mut best_organism = None;
    let mut best_fitness = 0.0;
    let mut i = 0;
    while champion.is_none() && i < MAX_ITERATIONS {
        i += 1;
        population.evolve(&mut environment, &p, true);
        for organism in population.get_organisms() {
            if organism.fitness > best_fitness {
                best_fitness = organism.fitness;
                best_organism = Some(organism.clone());
            }
            if organism.fitness > 15.0 {
                champion = Some(organism.clone());
            }
        }
    }
    let best_organism = best_organism.unwrap();
    println!("Result: {}", best_fitness);
    println!(
        " - {} neurons, {} connections",
        best_organism.genome.n_neurons(),
        best_organism.genome.n_connections()
    );
}
