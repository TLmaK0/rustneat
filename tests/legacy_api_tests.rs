//! Issue #103: the legacy methods that bypass the population's random
//! generator, `MutationConfig` or `InnovationRegistry` are deprecated in
//! favour of the variant the population uses, and behave exactly like that
//! variant called with the defaults they imply. Only the tests that check
//! that equivalence use deprecated items (`#[allow(deprecated)]`).
extern crate rand;
extern crate rand_chacha;
extern crate rustneat;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rustneat::{
    Gene, Genome, InnovationRegistry, MutationConfig, Organism, OutputActivation, Specie,
};

fn rng(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

/// A genome with a hidden neuron, so mutation and crossover have something
/// to work with.
fn genome(seed: u64) -> Genome {
    let mut genome = Genome::new_initialized_with_rng(2, 1, &mut rng(seed));
    let config = MutationConfig::builder().add_neuron_rate(1f64).build();
    genome.mutate_with_innovations(&config, &mut InnovationRegistry::new(3), &mut rng(seed));
    genome
}

fn same(a: &Genome, b: &Genome) -> bool {
    format!("{:?}", a) == format!("{:?}", b)
}

#[test]
fn builder_starts_from_the_defaults() {
    let built = MutationConfig::builder().build();
    assert_eq!(
        format!("{:?}", built),
        format!("{:?}", MutationConfig::default())
    );
    let config = MutationConfig::builder().add_neuron_rate(0.7).build();
    assert_eq!(config.add_neuron_rate, 0.7);
}

#[test]
#[allow(deprecated)]
fn new_is_the_builder() {
    assert_eq!(
        format!("{:?}", MutationConfig::new().tau(0.3).build()),
        format!("{:?}", MutationConfig::builder().tau(0.3).build())
    );
}

#[test]
fn organism_with_config_takes_the_ctrnn_settings_of_the_config() {
    let config = MutationConfig::builder()
        .tau(0.2)
        .dt(0.005)
        .step_time(0.3)
        .output_activation(OutputActivation::Linear)
        .build();
    let organism = Organism::with_config(Genome::new_unconnected(2, 1), &config);
    assert_eq!(organism.tau, 0.2);
    assert_eq!(organism.dt, 0.005);
    assert_eq!(organism.step_time, 0.3);
    assert_eq!(organism.output_activation, OutputActivation::Linear);
    assert_eq!(organism.fitness, 0f64);
    assert!(!organism.is_evaluated());
}

#[test]
fn organism_new_uses_the_default_config() {
    let config = MutationConfig::default();
    let organism = Organism::new(Genome::new_unconnected(2, 1));
    let expected = Organism::with_config(Genome::new_unconnected(2, 1), &config);
    assert_eq!(organism.tau, expected.tau);
    assert_eq!(organism.dt, expected.dt);
    assert_eq!(organism.step_time, expected.step_time);
    assert_eq!(organism.output_activation, expected.output_activation);
}

#[test]
#[allow(deprecated)]
fn genome_mutate_without_registry_numbers_like_a_fresh_registry() {
    let config = MutationConfig::builder()
        .add_neuron_rate(0.5)
        .add_connection_rate(0.5)
        .build();
    for seed in 0..50 {
        let mut legacy = genome(seed);
        let mut current = legacy.clone();
        legacy.mutate_with_config_and_rng(&config, &mut rng(seed + 1000));
        let mut registry = InnovationRegistry::new(current.len());
        current.mutate_with_innovations(&config, &mut registry, &mut rng(seed + 1000));
        assert!(same(&legacy, &current), "seed {}", seed);
    }
}

#[test]
#[allow(deprecated)]
fn organism_mutate_without_registry_numbers_like_a_fresh_registry() {
    let config = MutationConfig::builder().add_neuron_rate(0.5).build();
    for seed in 0..20 {
        let parent = Organism::new(genome(seed));
        let legacy = parent.mutate_with_config_and_rng(&config, &mut rng(seed));
        let mut registry = InnovationRegistry::new(parent.genome.len());
        let current = parent.mutate_with_innovations(&config, &mut registry, &mut rng(seed));
        assert!(same(&legacy.genome, &current.genome), "seed {}", seed);
    }
}

#[test]
#[allow(deprecated)]
fn genome_mate_without_config_uses_the_default_config() {
    let config = MutationConfig::default();
    for seed in 0..20 {
        let (a, b) = (genome(seed), genome(seed + 100));
        for fitter in [true, false] {
            let legacy = a.mate_with_rng(&b, fitter, &mut rng(seed));
            let current = a.mate_with_config_and_rng(&b, fitter, &config, &mut rng(seed));
            assert!(same(&legacy, &current));
        }
        let legacy = a.mate_equal_with_rng(&b, &mut rng(seed));
        let current = a.mate_equal_with_config_and_rng(&b, &config, &mut rng(seed));
        assert!(same(&legacy, &current));
    }
}

#[test]
#[allow(deprecated)]
fn organism_mate_without_config_uses_the_default_config() {
    let config = MutationConfig::default();
    for seed in 0..20 {
        let mut a = Organism::new(genome(seed));
        let mut b = Organism::new(genome(seed + 100));
        a.fitness = (seed % 3) as f64;
        b.fitness = 1f64;
        let legacy = a.mate_with_rng(&b, &mut rng(seed));
        let current = a.mate_with_config_and_rng(&b, &config, &mut rng(seed));
        assert!(same(&legacy.genome, &current.genome));
    }
}

#[test]
#[allow(deprecated)]
fn species_checks_without_config_use_the_default_config() {
    let config = MutationConfig::default();
    for seed in 0..20 {
        let (a, b) = (genome(seed), genome(seed + 100));
        assert_eq!(
            a.compatibility_distance(&b),
            a.compatibility_distance_with_config(&b, &config)
        );
        assert_eq!(
            a.is_same_specie(&b),
            a.is_same_specie_with_config(&b, &config)
        );
        let threshold = a.compatibility_distance_with_config(&b, &config);
        for t in [threshold * 0.5, threshold * 2.0] {
            let custom = MutationConfig {
                compatibility_threshold: t,
                ..MutationConfig::default()
            };
            assert_eq!(
                a.is_same_specie_with_threshold(&b, t),
                a.is_same_specie_with_config(&b, &custom)
            );
            let specie = Specie::new(a.clone());
            let organism = Organism::new(b.clone());
            assert_eq!(
                specie.match_genome_with_threshold(&organism, t),
                specie.match_genome_with_config(&organism, &custom)
            );
        }
        let specie = Specie::new(a.clone());
        let organism = Organism::new(b.clone());
        assert_eq!(
            specie.match_genome(&organism),
            specie.match_genome_with_config(&organism, &config)
        );
    }
}

#[test]
#[allow(deprecated)]
fn specie_parents_without_config_use_the_default_config() {
    let mut specie = Specie::new(genome(0));
    for seed in 0..7 {
        let mut organism = Organism::new(genome(seed));
        organism.fitness = seed as f64;
        specie.add(organism);
    }
    let legacy: Vec<f64> = specie.parents().map(|o| o.fitness).collect();
    let current: Vec<f64> = specie
        .parents_with_config(&MutationConfig::default())
        .map(|o| o.fitness)
        .collect();
    assert_eq!(legacy, current);
}

/// Without registry the offspring is drawn exactly like with one: only the
/// ids of new neurons differ, so with no add-neuron mutation both agree.
#[test]
#[allow(deprecated)]
fn specie_offspring_without_registry_differs_only_in_neuron_ids() {
    let config = MutationConfig::builder()
        .add_neuron_rate(0f64)
        .add_connection_rate(0.5)
        .build();
    let specie = || {
        let mut specie = Specie::new(genome(0));
        for seed in 0..6 {
            let mut organism = Organism::new(genome(seed));
            organism.fitness = seed as f64 + 1.0;
            organism.adjusted_fitness = organism.fitness;
            specie.add(organism);
        }
        specie
    };
    let (mut legacy, mut current) = (specie(), specie());
    legacy.generate_offspring_with_config_and_rng(8, &[], &config, &mut rng(5));
    current.generate_offspring_with_innovations(
        8,
        &[],
        &config,
        &mut InnovationRegistry::new(3),
        &mut rng(5),
    );
    assert_eq!(legacy.organisms.len(), 8);
    for (l, c) in legacy.organisms.iter().zip(&current.organisms) {
        assert!(same(&l.genome, &c.genome));
    }
}

/// `add_gene` is a construction primitive: adding a connection the genome
/// already has re-enables it and keeps its weight (the add-connection
/// mutation never picks an existing connection, so it never does this).
#[test]
fn add_gene_re_enables_an_existing_connection_and_keeps_its_weight() {
    let mut genome = Genome::new_unconnected(1, 1);
    genome.add_gene(Gene::new(0, 1, 0.5, false));
    genome.add_gene(Gene::new(0, 1, 3.0, false));
    assert_eq!(genome.get_genes().len(), 1);
    assert!(genome.get_genes()[0].enabled());
    assert_eq!(genome.get_genes()[0].weight(), 0.5);
}
