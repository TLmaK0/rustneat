//! Innovation registry (issue #67): splitting the same connection gives the
//! same neuron id in every lineage of a population, different splits give
//! different ids.

extern crate rand;
extern crate rand_chacha;
extern crate rustneat;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rustneat::{
    Environment, Gene, Genome, InnovationRegistry, MutationConfig, Organism, Population,
};

/// Only the add-neuron mutation, always.
fn only_add_neuron() -> MutationConfig {
    MutationConfig::builder()
        .add_connection_rate(0f64)
        .add_neuron_rate(1f64)
        .weight_mutation_rate(0f64)
        .toggle_expression_rate(0f64)
        .bias_mutation_rate(0f64)
        .build()
}

/// 2 inputs (0, 1), 1 output (2) and the single connection `from -> 2`, so
/// the add-neuron mutation can only split that connection.
fn ancestor(from: usize) -> Genome {
    let mut genome = Genome::new_unconnected(2, 1);
    genome.add_gene(Gene::new(from, 2, 1f64, true));
    genome
}

fn connections(genome: &Genome) -> Vec<(usize, usize)> {
    genome
        .get_genes()
        .iter()
        .map(|g| (g.in_neuron_id(), g.out_neuron_id()))
        .collect()
}

/// Mutate `genome` once (it splits its only connection) and return the new neuron.
fn split(genome: &mut Genome, registry: &mut InnovationRegistry, seed: u64) -> usize {
    let before = genome.neurons();
    genome.mutate_with_innovations(
        &only_add_neuron(),
        registry,
        &mut ChaCha8Rng::seed_from_u64(seed),
    );
    let new: Vec<usize> = genome
        .neurons()
        .into_iter()
        .filter(|id| !before.contains(id))
        .collect();
    assert_eq!(new.len(), 1, "one new neuron expected, got {:?}", new);
    new[0]
}

#[test]
fn the_same_split_in_two_lineages_gets_the_same_neuron_id() {
    let mut registry = InnovationRegistry::new(3);
    let mut a = ancestor(0);
    let mut b = ancestor(0);
    let id_a = split(&mut a, &mut registry, 1);
    let id_b = split(&mut b, &mut registry, 2);
    assert_eq!(id_a, id_b);
    assert_eq!(connections(&a), connections(&b));
    // Same genes, same weights: nothing is disjoint or excess.
    assert_eq!(
        a.compatibility_distance_with_config(&b, &MutationConfig::default()),
        0f64
    );
}

#[test]
fn different_splits_get_different_neuron_ids() {
    let mut registry = InnovationRegistry::new(3);
    let mut a = ancestor(0);
    let mut b = ancestor(1);
    let id_a = split(&mut a, &mut registry, 1);
    let id_b = split(&mut b, &mut registry, 1);
    assert!(id_a != id_b);
    assert!(id_a >= 3 && id_b >= 3, "never an input or output id");
}

#[test]
fn a_later_lineage_repeating_an_innovation_aligns_with_the_first_one() {
    // The case of the audit (#55, H5): A splits 0->2, B splits 1->2, and C
    // = A + split 1->2. C's genes for that split must match B's.
    let mut registry = InnovationRegistry::new(3);
    let mut a = Genome::new_unconnected(2, 1);
    a.add_gene(Gene::new(0, 2, 1f64, true));
    let id_a = split(&mut a, &mut registry, 1);
    let mut b = ancestor(1);
    let id_b = split(&mut b, &mut registry, 1);

    // C: A plus the connection 1 -> 2, then split it.
    let mut c = Genome::new_unconnected(2, 1);
    c.add_gene(Gene::new(1, 2, 1f64, true));
    let id_c = split(&mut c, &mut registry, 3);
    assert_eq!(id_c, id_b);
    assert!(id_a != id_b);
    for gene in [(1, id_b), (id_b, 2)] {
        assert!(connections(&b).contains(&gene));
        assert!(connections(&c).contains(&gene));
    }
}

#[test]
fn bias_crossover_mixes_only_the_same_innovation() {
    let mut registry = InnovationRegistry::new(3);
    let mut a = ancestor(0);
    let mut b = ancestor(1);
    let mut c = ancestor(0);
    let id_a = split(&mut a, &mut registry, 1);
    let id_b = split(&mut b, &mut registry, 1);
    let id_c = split(&mut c, &mut registry, 1);
    a.set_bias(id_a, 1f64);
    b.set_bias(id_b, -1f64);
    c.set_bias(id_c, 3f64);
    let mut rng = ChaCha8Rng::seed_from_u64(7);

    // Different splits: the hidden neuron of the structure donor keeps its
    // own bias, never the other's.
    for _ in 0..200 {
        let child = a.mate_with_config_and_rng(&b, true, &MutationConfig::default(), &mut rng);
        assert_eq!(child.bias(id_a), 1f64);
    }
    // The same split: the bias comes from either parent (or their average).
    let mut seen = std::collections::BTreeSet::new();
    for _ in 0..200 {
        let child = a.mate_with_config_and_rng(&c, true, &MutationConfig::default(), &mut rng);
        seen.insert(child.bias(id_a).to_string());
    }
    assert!(seen.contains("1") && seen.contains("3"), "{:?}", seen);
}

/// The deprecated mutation without registry keeps numbering new neurons by
/// the genome alone (issue #103).
#[test]
#[allow(deprecated)]
fn genomes_without_registry_still_get_new_ids_of_their_own() {
    let mut genome = ancestor(0);
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    for _ in 0..5 {
        genome.mutate_with_config_and_rng(&only_add_neuron(), &mut rng);
    }
    assert_eq!(genome.neurons(), (0..8).collect::<Vec<_>>());
    let mut organism = Organism::new(genome);
    let mut output = vec![0f64];
    organism.activate(vec![1f64, 1f64], &mut output);
    assert!(output[0].is_finite());
}

/// Fitness = number of genes, so the population keeps growing.
struct Growth;

impl Environment for Growth {
    fn test(&self, organism: &mut Organism) -> f64 {
        organism.genome.total_genes() as f64
    }
}

#[test]
fn population_hidden_neurons_come_from_its_registry() {
    let config = MutationConfig::builder().add_neuron_rate(0.5f64).build();
    let mut population =
        Population::create_population_initialized_with_config(30, 2, 1, config).with_seed(5);
    for _ in 0..15 {
        population.evaluate_in(&Growth).unwrap();
        population.evolve();
    }
    let mut hidden = std::collections::BTreeMap::new();
    for organism in population.get_organisms() {
        for id in organism.genome.neurons().into_iter().filter(|&id| id >= 3) {
            let split = population
                .innovations()
                .split_of(id)
                .unwrap_or_else(|| panic!("neuron {} not in the registry", id));
            // Every genome that has the neuron got it from the same split.
            assert_eq!(*hidden.entry(id).or_insert(split), split);
        }
    }
    assert!(
        hidden.len() > 3,
        "the population did not grow: {:?}",
        hidden
    );
    // Some hidden neuron is shared by several lineages.
    let shared = hidden.keys().any(|id| {
        population
            .get_organisms()
            .iter()
            .filter(|o| o.genome.neurons().contains(id))
            .count()
            > 1
    });
    assert!(shared);
}
