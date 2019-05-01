#[macro_use]
extern crate blackbox_derive;
#[macro_use]
extern crate slog;

use blackbox_derive::make_optimizer;
use blackbox::BlackboxInput;
use slog::Logger;

use rustneat::{Environment, Organism, Population, NeuralNetwork, NeatParams};
use chrono::{Timelike, Utc};

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

fn run(p: &NeatParams, n_gen: usize) -> f64 {

    let mut start_genome = NeuralNetwork::with_neurons(3);
    // start_genome.add_connection(0, 2, 0.0);
    // start_genome.add_connection(1, 2, 0.0);
    let mut population = Population::create_population_from(start_genome, 150);
    let mut environment = XORClassification;
    let mut champion: Option<Organism> = None;
    for i in 0..n_gen {
        population.evolve(&mut environment, p);
    }

    let mut best_fitness = 0.0;
    for organism in population.get_organisms() {
        if organism.fitness > best_fitness {
            best_fitness = organism.fitness;
        }
    }
    best_fitness
}

make_optimizer! {
    Configuration {
        remove_after_n_generations: usize = 5 .. 40,
        species_elite: usize = 1 .. 5,

        mutation_pr: f64 = 0.2 .. 1.0,
        interspecie_mate_pr: f64 = 0.0 .. 0.002,
        cull_fraction: f64 = 0.05 .. 0.3,

        // n_conn_to_mutate: 0,
        mutate_add_conn_pr: f64 = 0.1..0.5,
        mutate_del_conn_pr: f64 = 0.1..0.5
        mutate_add_neuron_pr: f64 = 0.01..0.04,
        mutate_del_neuron_pr: f64 = 0.01..0.04,

        // weight_init_mean: 0.0, 
        weight_init_var: f64 = 0.5 .. 2.0, 
        weight_mutate_var: f64 = 0.2 .. 2.0,
        weight_mutate_pr: f64 = 0.2 .. 0.8,
        weight_replace_pr: f64 = 0.01 .. 0.2,

        // bias_init_mean: 0.0, 
        bias_init_var: f64 = 0.5 .. 2.0, 
        bias_mutate_var: f64 = 0.2 .. 2.0,
        bias_mutate_pr: f64 = 0.2 .. 0.8,
        bias_replace_pr: f64 = 0.01 .. 0.2,

        include_weak_disjoint_gene: f64 = 0.1 .. 0.3,

        compatibility_threshold: f64 = 2.0 .. 4.0,
        distance_weight_coef: f64 = 0.0 .. 0.5,
        distance_disjoint_coef: f64 = 0.5 .. 1.0,
    }

    const N_GEN: usize = 100; // generations per round
    const N_POPULATIONS: usize = 26; // populations per iteration
    let p = NeatParams {
        n_inputs: 2,
        n_outputs: 1,
        remove_after_n_generations,
        species_elite,

        mutation_pr,
        interspecie_mate_pr,
        cull_fraction,

        mutate_add_conn_pr,
        mutate_del_conn_pr,
        mutate_add_neuron_pr,
        mutate_del_neuron_pr,
        include_weak_disjoint_gene,

        weight_init_mean: 0.0,
        weight_init_var, 
        weight_mutate_var,
        weight_mutate_pr,
        weight_replace_pr,

        bias_init_mean: 0.0,
        bias_init_var, 
        bias_mutate_var,
        bias_mutate_pr,
        bias_replace_pr,

        compatibility_threshold,
        distance_weight_coef,
        distance_disjoint_coef,
    };
    // Take the average of N rounds
    let score = (0..N_POPULATIONS)
        .map(|_| run(&p, N_GEN))
        .sum::<f64>() / N_POPULATIONS as f64;
    println!("Iteration... Score = {}", score);
    score
    
}

fn main() {
    let log = slog::Logger::root(slog::Discard, o!());
    let now = Utc::now();
    println!("Start: {:02}:{:02}:{:02}", now.hour(), now.minute(), now.second());

    const N_ITER: usize = 250;
    let config = Configuration::bayesian_search(12, N_ITER, log.clone());
    println!("Score: {}", config.evaluate(log));
    println!("Config: {:?}", config);

    println!("\nExecution time: {}", (Utc::now() - now).num_seconds());
}
