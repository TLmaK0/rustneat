extern crate rand;
extern crate rustneat;

use rustneat::{Environment, Population, NeuralNetwork, Params, Organism};
use std::io::Write;

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
    let p = Params {
        prune_after_n_generations: 37,
        n_to_prune: 3,
        mutation_pr: 0.74,
        interspecie_mate_pr: 0.001,
        cull_fraction: 0.122,

        c2: 0.8,
        c3: 0.16,
        mutate_conn_weight_pr: 0.39,
        mutate_conn_weight_perturbed_pr: 0.9,
        n_conn_to_mutate: 0,
        mutate_add_conn_pr: 0.00354,
        mutate_add_neuron_pr: 0.001,
        mutate_toggle_expr_pr: 0.00171,
        mutate_bias_pr: 0.0222,
        include_weak_disjoint_gene: 0.183,
        compatibility_threshold: 3.1725,
        ..Default::default()
    };

    solve_time_perf(&p);
    // fixed_generations_perf(&p);

}

/// See how fast, on average, rustneat can solve XOR
fn solve_time_perf(p: &Params) {
    const N_EXP: usize = 40;
    const MAX_GEN: usize = 800;
    let mut solve_gens = Vec::new();
    let mut neurons = Vec::new();
    let mut connections = Vec::new();
    let mut could_not_solve = 0;
    for exp in 0..N_EXP {
        print!("Experiment {}/{}\r", exp+1, N_EXP);
        std::io::stdout().flush().unwrap();
        let start_genome = NeuralNetwork::with_neurons(3);
        let mut population = Population::create_population_from(start_genome, 150);
        let mut environment = XORClassification;

        let mut champion: Option<Organism> = None;
        let mut i = 0;
        while champion.is_none() && i < MAX_GEN {
            population.evolve(&mut environment, &p);
            for organism in population.get_organisms() {
                if organism.fitness > 15.7 {
                    champion = Some(organism.clone());
                }
            }
            i += 1;
        }
        if let Some(champion) = champion {
            solve_gens.push(i);
            neurons.push(champion.genome.n_neurons());
            connections.push(champion.genome.n_connections());
        } else {
            could_not_solve += 1;
        }
    }
    println!("");
    println!("Could not solve {} times", could_not_solve);
    {
        let mean = solve_gens.iter().sum::<usize>() as f64 / solve_gens.len() as f64;
        println!("{:?}", solve_gens);
        println!("Mean solve time: {}", mean);
    }

    {
        let mean = neurons.iter().sum::<usize>() as f64 / neurons.len() as f64;
        println!("Neurons:      {}", mean);
    }

    {
        let mean = connections.iter().sum::<usize>() as f64 / connections.len() as f64;
        println!("Connections:  {}", mean);
    }
}

/// See the best fitness, on average, after a fixed amount of generations
fn fixed_generations_perf(p: &Params) {
    const N_EXP: usize = 40;
    const N_GEN: usize = 300;
    let mut scores = Vec::new();
    let mut neurons = Vec::new();
    let mut connections = Vec::new();
    for exp in 0..N_EXP {
        print!("Experiment {}/{}\r", exp+1, N_EXP);
        std::io::stdout().flush().unwrap();
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

    println!("");
    {
        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        println!("{:?}", scores);
        println!("Mean:         {}", mean);
    }

    {
        let mean = neurons.iter().sum::<usize>() as f64 / neurons.len() as f64;
        println!("Neurons:      {}", mean);
    }

    {
        let mean = connections.iter().sum::<usize>() as f64 / connections.len() as f64;
        println!("Connections:  {}", mean);
    }


}
