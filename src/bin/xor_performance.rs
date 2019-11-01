extern crate rand;
extern crate rustneat;

use rustneat::{Environment, NeatParams, NeuralNetwork, Organism, Population};
use std::io::Write;

// This example measure average XOR performance, and should be useful to check
// that changes in the algorithm doesn't break the algorithm

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

         16.0 / (1.0 + distance)
    }
}

fn main() {
    // let p = NeatParams {
    // n_inputs: 2,
    // n_outputs: 1,

    // remove_after_n_generations: 18,
    // species_elite: 2,

    // mutation_pr: 1.0,
    // interspecie_mate_pr: 0.01,
    // cull_fraction: 0.2,

    // mutate_add_conn_pr: 0.5,
    // mutate_del_conn_pr: 0.5,
    // mutate_add_neuron_pr: 0.1,
    // mutate_del_neuron_pr: 0.1,

    // weight_init_mean: 0.0,
    // weight_init_var: 1.0,
    // weight_mutate_var: 0.5,
    // weight_mutate_pr: 0.8,
    // weight_replace_pr: 0.1,

    // bias_init_mean: 0.0,
    // bias_init_var: 1.0,
    // bias_mutate_var: 0.5,
    // bias_mutate_pr: 0.7,
    // bias_replace_pr: 0.1,

    // include_weak_disjoint_gene: 0.0,

    // // other
    // compatibility_threshold: 3.3,
    // distance_weight_coef: 0.13,
    // distance_disjoint_coef: 0.6,
    // };
    let p = NeatParams::optimized_for_xor3(2, 1);

    use chrono::{Timelike, Utc};
    let now = Utc::now();
    println!(
        "Start: {:02}:{:02}:{:02}",
        now.hour(),
        now.minute(),
        now.second()
    );

    solve_time_perf(&p, 40, 1000, 150);
    // fixed_generations_perf(&p);

    println!("\nExecution time: {}", (Utc::now() - now).num_seconds());
}

/// See how fast, on average, rustneat can solve XOR
fn solve_time_perf(p: &NeatParams, n_exp: usize, n_gen: usize, population_size: usize) {
    let mut solve_gens = Vec::new();
    let mut neurons = Vec::new();
    let mut connections = Vec::new();
    let mut could_not_solve = 0;
    for exp in 0..n_exp {
        print!("Experiment {}/{}\r", exp + 1, n_exp);
        std::io::stdout().flush().unwrap();
        let start_genome = NeuralNetwork::with_neurons(3);
        let mut population = Population::create_population_from(start_genome, population_size);
        let mut environment = XORClassification;

        let mut champion: Option<Organism> = None;
        let mut i = 0;
        while champion.is_none() && i < n_gen {
            population.evolve(&mut environment, &p, true);
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
    println!();
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
fn fixed_generations_perf(p: &NeatParams) {
    const N_EXP: usize = 40;
    const N_GEN: usize = 200;
    let mut scores = Vec::new();
    let mut neurons = Vec::new();
    let mut connections = Vec::new();
    for exp in 0..N_EXP {
        print!("Experiment {}/{}\r", exp + 1, N_EXP);
        std::io::stdout().flush().unwrap();
        let start_genome = NeuralNetwork::with_neurons(3);
        let mut population = Population::create_population_from(start_genome, 150);
        let mut environment = XORClassification;

        for i in 0..N_GEN {
            population.evolve(&mut environment, &p, true);

            let best_organism = population.get_champion();

            if i == N_GEN - 1 {
                scores.push(best_organism.fitness);
                neurons.push(best_organism.genome.n_neurons());
                connections.push(best_organism.genome.n_connections());
            }
        }
    }

    println!();
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
