extern crate rustneat;

use rustneat::{Environment, Organism, Population};

/// Every organism gets the same fitness: the population never improves, so
/// any population-level "rescue" and the species stagnation kick in.
struct Constant;

impl Environment for Constant {
    fn test(&self, _organism: &mut Organism) -> f64 {
        1.0
    }
}

fn genomes(population: &Population) -> Vec<String> {
    let mut genomes: Vec<String> = population
        .get_organisms()
        .iter()
        .map(|organism| format!("{:?}", organism.genome))
        .collect();
    genomes.sort();
    genomes
}

/// `evolve` always produces a new generation: no generation is lost
/// leaving the population identical, bit for bit.
#[test]
fn with_constant_fitness_every_evolve_changes_the_population() {
    let mut population = Population::create_population_initialized(150, 2, 1).with_seed(1);
    population.evaluate_in(&Constant);
    for generation in 1..=200 {
        let before = genomes(&population);
        population.evolve();
        assert_eq!(population.size(), 150, "generation {}", generation);
        assert_ne!(
            genomes(&population),
            before,
            "evolve left the population identical in generation {}",
            generation
        );
        population.evaluate_in(&Constant);
    }
}
