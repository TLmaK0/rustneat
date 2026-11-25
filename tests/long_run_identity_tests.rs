extern crate rustneat;

use rustneat::{Environment, MutationConfig, Organism, Population};

/// The environment of the #73 regression test
/// (`empty_species_regression_tests.rs`).
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

/// FNV-1a: a hash that, unlike `DefaultHasher`, is the same in every Rust
/// release.
fn fnv1a(hash: &mut u64, bytes: &[u8]) {
    for &byte in bytes {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

/// Digest of a whole run of the #73 scenario: after every generation, the
/// species ids, every genome (genes, biases, τ genes), every fitness and
/// the champion fitness.
fn run_digest(generations: usize) -> u64 {
    let config = MutationConfig::new()
        .target_species(5)
        .species_stagnation(3, 1)
        .build();
    let mut population =
        Population::create_population_initialized_with_config(30, 2, 1, config).with_seed(19);
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    population.evaluate_in(&Xor).unwrap();
    for _ in 0..generations {
        population.evolve();
        population.evaluate_in(&Xor).unwrap();
        for specie in &population.species {
            fnv1a(&mut hash, &specie.id.to_le_bytes());
            for organism in &specie.organisms {
                fnv1a(&mut hash, format!("{:?}", organism.genome).as_bytes());
                fnv1a(&mut hash, &organism.fitness.to_bits().to_le_bytes());
            }
        }
        let champion = population.champion.as_ref().map(|c| c.fitness.to_bits());
        fnv1a(&mut hash, &champion.unwrap_or(0).to_le_bytes());
    }
    hash
}

/// The speed-ups of #101 must not change a single result: the digest of
/// 300 generations of the #73 scenario (seed 19, 30 organisms) is the one
/// the code gave before them. Fitness goes through `exp`, whose last bit
/// may differ between math libraries, so the recorded digest is only
/// checked on Linux x86_64 (where it was recorded); elsewhere the run must
/// at least be reproducible.
#[test]
fn long_run_gives_the_recorded_results() {
    let digest = run_digest(300);
    assert_eq!(digest, run_digest(300), "same seed, different run");
    if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        assert_eq!(digest, 0xb085_f2fe_3578_d4a4, "digest {:#018x}", digest);
    }
}
