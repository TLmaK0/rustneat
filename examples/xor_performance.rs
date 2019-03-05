extern crate rand;
extern crate rustneat;

use rustneat::{Environment, Organism, Population, NeuralNetwork, Params};

// This example measure average XOR performance, and should be useful to check that changes in the
// algorithm doesn't break the algorithm

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

fn main() {
    const N_EXP: usize = 22;
    const N_GEN: usize = 150;
    let p = Params::default();

    let mut scores = Vec::new();
    let mut neurons = Vec::new();
    let mut connections = Vec::new();
    for exp in 0..N_EXP {
        println!("Experiment {}/{}", exp+1, N_EXP);
        let start_genome = NeuralNetwork::with_neurons(3);
        let mut population = Population::create_population_from(start_genome, 150);
        let mut environment = XORClassification;

        for i in 0..N_GEN {
            population.evolve(&mut environment, &p);

            let best_organism = population.get_champion();


            if i == N_GEN-1 {
                scores.push(best_organism.fitness);
                neurons.push(best_organism.genome.n_neurons());
                connections.push(best_organism.genome.n_connections());
            }
        }
    }

    {
        println!("= BEST FITNESS LAST GENERATION =");
        println!("- values: {:?}", scores);
        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        let var = scores.iter().map(|x| (x - mean).powf(2.0)).sum::<f64>() / scores.len() as f64;
        println!("- mean {}", mean);
        println!("- var {}", var);
    }



    {
        println!("= N NEURONS LAST GENERATION");
        let mean = neurons.iter().sum::<usize>() as f64 / neurons.len() as f64;
        let var = neurons.iter().map(|x| (*x as f64 - mean).powf(2.0)).sum::<f64>() / neurons.len() as f64;
        println!("- mean {}", mean);
        println!("- var {}", var);
    }

    {
        println!("= N CONNECTIONS LAST GENERATION");
        let mean = connections.iter().sum::<usize>() as f64 / connections.len() as f64;
        let var = connections.iter().map(|x| (*x as f64 - mean).powf(2.0)).sum::<f64>() / connections.len() as f64;
        println!("- mean {}", mean);
        println!("- var {}", var);
    }


}
