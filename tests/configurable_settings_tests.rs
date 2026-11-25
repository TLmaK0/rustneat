//! Issue #100: the NEAT settings that used to be hardcoded (crossover weight
//! averaging, survival threshold, champion copy, split of a connection) are
//! `MutationConfig` fields, with the old values as defaults.
extern crate rand;
extern crate rand_chacha;
extern crate rustneat;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rustneat::{Gene, Genome, InnovationRegistry, MutationConfig, Organism, Specie};

#[test]
fn defaults_are_the_values_that_were_hardcoded() {
    let config = MutationConfig::default();
    assert_eq!(config.crossover_weight_averaging_rate, 0.4);
    // Was 0.5 until #108 (measured on the XOR bench).
    assert_eq!(config.survival_threshold, 0.2);
    assert_eq!(config.champion_copy_min_offspring, 2);
    assert_eq!(config.split_input_weight, 5.0);
    assert_eq!(config.split_bias, -2.5);
}

#[test]
fn builders_set_the_new_fields() {
    let config = MutationConfig::builder()
        .crossover_weight_averaging_rate(0.1)
        .survival_threshold(0.2)
        .champion_copy_min_offspring(6)
        .split_neuron(1.0, 0.0)
        .build();
    assert_eq!(config.crossover_weight_averaging_rate, 0.1);
    assert_eq!(config.survival_threshold, 0.2);
    assert_eq!(config.champion_copy_min_offspring, 6);
    assert_eq!(config.split_input_weight, 1.0);
    assert_eq!(config.split_bias, 0.0);
}

/// Two genomes with the same genes and different weights.
fn parents() -> (Genome, Genome) {
    let genes = |w: f64| {
        vec![
            Gene::new(0, 2, w, true),
            Gene::new(1, 2, w + 1.0, true),
            Gene::new(0, 3, w + 2.0, true),
            Gene::new(1, 3, w + 3.0, true),
        ]
    };
    (
        Genome::from_genes_with_io(genes(1.0), 3, 2, 2),
        Genome::from_genes_with_io(genes(5.0), 3, 2, 2),
    )
}

fn weights(genome: &Genome) -> Vec<f64> {
    genome.get_genes().iter().map(|g| g.weight()).collect()
}

#[test]
fn crossover_always_averages_matching_weights_with_rate_one() {
    let (a, b) = parents();
    let config = MutationConfig::builder()
        .crossover_weight_averaging_rate(1.0)
        .build();
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    for _ in 0..50 {
        let child = a.mate_with_config_and_rng(&b, true, &config, &mut rng);
        assert_eq!(weights(&child), vec![3.0, 5.0, 4.0, 6.0]);
    }
}

#[test]
fn crossover_never_averages_matching_weights_with_rate_zero() {
    let (a, b) = parents();
    let config = MutationConfig::builder()
        .crossover_weight_averaging_rate(0.0)
        .build();
    let (wa, wb) = (weights(&a), weights(&b));
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    for _ in 0..50 {
        let child = a.mate_with_config_and_rng(&b, true, &config, &mut rng);
        for (i, w) in weights(&child).iter().enumerate() {
            assert!(*w == wa[i] || *w == wb[i], "weight {} averaged", w);
        }
    }
}

#[test]
fn a_split_uses_the_configured_weight_and_bias() {
    let config = MutationConfig::builder()
        .add_connection_rate(0.0)
        .add_neuron_rate(1.0)
        .weight_mutation_rate(0.0)
        .bias_mutation_rate(0.0)
        .toggle_expression_rate(0.0)
        .split_neuron(1.0, 0.25)
        .build();
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let mut genome = Genome::new_initialized_with_rng(2, 1, &mut rng);
    genome.mutate_with_innovations(
        &config,
        &mut InnovationRegistry::new(genome.len()),
        &mut rng,
    );
    let new = genome
        .neurons()
        .into_iter()
        .find(|&id| id > 2)
        .expect("no new neuron");
    assert_eq!(genome.bias(new), 0.25);
    let into_new = genome
        .get_genes()
        .iter()
        .find(|g| g.out_neuron_id() == new)
        .unwrap();
    assert_eq!(into_new.weight(), 1.0);
}

fn species_of(fitnesses: &[f64]) -> Specie {
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let genome = Genome::new_initialized_with_rng(2, 1, &mut rng);
    let mut specie = Specie::new(genome.clone());
    for &fitness in fitnesses {
        let mut organism = Organism::new(genome.clone());
        organism.fitness = fitness;
        specie.add(organism);
    }
    specie
}

#[test]
fn survival_threshold_sets_how_many_organisms_reproduce() {
    let specie = species_of(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    let count = |threshold: f64| {
        let config = MutationConfig::builder()
            .survival_threshold(threshold)
            .build();
        specie.parents_with_config(&config).count()
    };
    assert_eq!(count(0.5), 5);
    assert_eq!(count(0.2), 2);
    assert_eq!(count(1.0), 10);
    // At least one organism always reproduces.
    assert_eq!(count(0.0), 1);
    // The default (#108) is the fittest fifth, as in Stanley's NEAT.
    assert_eq!(
        specie
            .parents_with_config(&MutationConfig::default())
            .count(),
        2
    );
    let mut best: Vec<f64> = specie
        .parents_with_config(&MutationConfig::builder().survival_threshold(0.3).build())
        .map(|o| o.fitness)
        .collect();
    best.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(best, vec![8.0, 9.0, 10.0]);
}

#[test]
fn survival_threshold_limits_the_parents_of_the_offspring() {
    // Each organism has its own weight: with mutation only and no weight
    // changes, every child keeps the weights of its parent.
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let mut specie = Specie::new(Genome::new_initialized_with_rng(2, 1, &mut rng));
    for fitness in 1..=10 {
        let genome = Genome::from_genes_with_io(
            vec![
                Gene::new(0, 2, fitness as f64, true),
                Gene::new(1, 2, 0.0, true),
            ],
            2,
            2,
            1,
        );
        let mut organism = Organism::new(genome);
        organism.fitness = fitness as f64;
        specie.add(organism);
    }
    let config = MutationConfig::builder()
        .mutation_probability(1.0)
        .add_connection_rate(0.0)
        .add_neuron_rate(0.0)
        .weight_mutation_rate(0.0)
        .bias_mutation_rate(0.0)
        .toggle_expression_rate(0.0)
        .survival_threshold(0.2)
        .build();
    specie.generate_offspring_with_innovations(
        40,
        &[],
        &config,
        &mut InnovationRegistry::default(),
        &mut rng,
    );
    assert_eq!(specie.organisms.len(), 40);
    for organism in &specie.organisms {
        let w = organism.genome.get_genes()[0].weight();
        assert!(
            w == 9.0 || w == 10.0,
            "child of a parent outside the top 20%: {}",
            w
        );
    }
}

#[test]
fn the_champion_is_copied_only_from_the_configured_offspring_size() {
    let elites = |offspring: usize, min: usize| {
        let mut specie = species_of(&[1.0, 2.0, 3.0, 4.0]);
        // The elite copy keeps the champion's fitness (4.0); every other
        // child is a new organism with fitness 0.
        let config = MutationConfig::builder()
            .champion_copy_min_offspring(min)
            .build();
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        specie.generate_offspring_with_innovations(
            offspring,
            &[],
            &config,
            &mut InnovationRegistry::default(),
            &mut rng,
        );
        assert_eq!(specie.organisms.len(), offspring);
        specie
            .organisms
            .iter()
            .filter(|organism| organism.fitness == 4.0)
            .count()
    };
    // Default: a species with 2 or more offspring keeps its champion.
    assert_eq!(elites(2, 2), 1);
    assert_eq!(elites(1, 2), 0);
    assert_eq!(elites(5, 6), 0);
    assert_eq!(elites(6, 6), 1);
    // 1: even a single child is the champion itself.
    assert_eq!(elites(1, 1), 1);
    // usize::MAX: no elitism at all.
    assert_eq!(elites(50, usize::MAX), 0);
    // 0 behaves like 1 (there must be a child to copy into).
    assert_eq!(elites(0, 0), 0);
}
