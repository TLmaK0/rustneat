extern crate rustneat;

use rustneat::{Environment, MutationConfig, Organism, Population};

/// XOR fitness as minus the squared error (always <= 0): different organisms
/// get different fitness, and with negative fitness the species with the
/// lowest average gets no offspring at all, so it is left empty after
/// `evolve`.
struct Xor;

impl Environment for Xor {
    fn test(&self, organism: &mut Organism) -> f64 {
        let mut distance = 0f64;
        for &(a, b, expected) in &[(0., 0., 0.), (0., 1., 1.), (1., 0., 1.), (1., 1., 0.)] {
            organism.reset_state();
            let mut output = vec![0f64];
            organism.activate(vec![a, b], &mut output);
            distance += (expected - output[0]).powi(2);
        }
        -distance
    }
}

/// Regression test for #19 (panic `Range::new called with low >= high` when
/// a species had no organisms) and #45 (remove empty species).
///
/// 1000 generations with a fixed seed, several species (dynamic threshold
/// aiming at 5 species) and a short stagnation limit (3 generations, only
/// the best species protected), so species are removed again and again and
/// some species are left without offspring (empty) after `evolve`. The
/// population must never panic and must keep its size.
#[test]
fn thousand_generations_with_many_species_and_empty_ones_do_not_panic() {
    const SIZE: usize = 30;
    const GENERATIONS: usize = 1000;
    let config = MutationConfig::builder()
        .target_species(5)
        .species_stagnation(3, 1)
        .build();
    let mut population =
        Population::create_population_initialized_with_config(SIZE, 2, 1, config).with_seed(19);

    let mut max_species = 0;
    let mut removed_species = 0;
    let mut generations_with_empty_species = 0;
    population.evaluate_in(&Xor).unwrap();
    for generation in 1..=GENERATIONS {
        let ids_before: Vec<i64> = population.species.iter().map(|s| s.id).collect();
        population.evolve();
        assert_eq!(
            population.size(),
            SIZE,
            "after evolve, generation {}",
            generation
        );
        removed_species += ids_before
            .iter()
            .filter(|id| population.species.iter().all(|s| s.id != **id))
            .count();
        if population.species.iter().any(|s| s.organisms.is_empty()) {
            generations_with_empty_species += 1;
        }

        population.evaluate_in(&Xor).unwrap();
        assert_eq!(
            population.size(),
            SIZE,
            "after evaluate, generation {}",
            generation
        );
        max_species = max_species.max(
            population
                .species
                .iter()
                .filter(|s| !s.organisms.is_empty())
                .count(),
        );
    }

    // The test is only meaningful if the scenario of #19 really happened:
    // several species, species removed and species left empty.
    assert!(max_species >= 3, "max species = {}", max_species);
    assert!(removed_species > 0, "no species was removed");
    assert!(
        generations_with_empty_species > 0,
        "no species was left empty after evolve"
    );
    println!(
        "max species {}, removed species {}, generations with empty species {}",
        max_species, removed_species, generations_with_empty_species
    );
}
