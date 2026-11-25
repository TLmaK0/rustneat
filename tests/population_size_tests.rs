extern crate rustneat;

use rustneat::{Environment, MutationConfig, Organism, Population};

struct Xor;

impl Environment for Xor {
    fn test(&self, organism: &mut Organism) -> f64 {
        let mut distance = 0f64;
        for &(a, b, expected) in &[(0., 0., 0.), (0., 1., 1.), (1., 0., 1.), (1., 1., 0.)] {
            let mut output = vec![0f64];
            organism.activate(vec![a, b], &mut output);
            distance += (expected - output[0]).powi(2);
        }
        16f64 / (1f64 + distance)
    }
}

/// With a low compatibility threshold the population splits into many
/// species; the offspring shares must still add up to the target size in
/// every generation, also when stagnant species are removed.
#[test]
fn population_size_is_exact_for_500_generations_with_many_species() {
    let config = MutationConfig::builder()
        .compatibility_threshold(0.6)
        .build();
    let mut population =
        Population::create_population_initialized_with_config(150, 2, 1, config).with_seed(1);
    assert_eq!(population.target_size(), 150);
    let mut max_species = 0;
    for generation in 0..500 {
        population.evolve();
        assert_eq!(
            population.size(),
            150,
            "after evolve, generation {}",
            generation
        );
        population.evaluate_in(&Xor).unwrap();
        assert_eq!(
            population.size(),
            150,
            "after evaluate, generation {}",
            generation
        );
        max_species = max_species.max(population.species.len());
    }
    // The test is only meaningful if speciation actually split the population.
    assert!(max_species > 1, "max species = {}", max_species);
}
