//! Issue #105: the CTRNN integrator of an organism is a `MutationConfig`
//! setting (`integrator`, default exponential Euler) instead of being
//! hardcoded, so explicit Euler can be chosen for a whole population.
extern crate rand;
extern crate rand_chacha;
extern crate rustneat;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rustneat::{Gene, Genome, Integrator, MutationConfig, Organism, OutputActivation};

/// One input connected to one output with weight 1, τ = 0.004 and the
/// default dt = 0.01 (h/τ = 2.5), output read linearly.
fn organism(integrator: Integrator) -> Organism {
    let genome = Genome::from_genes_with_io(vec![Gene::new(0, 1, 1.0, true)], 1, 1, 1);
    let config = MutationConfig::builder()
        .tau(0.004)
        .output_activation(OutputActivation::Linear)
        .integrator(integrator)
        .build();
    Organism::with_config(genome, &config)
}

fn outputs(organism: &mut Organism, activations: usize) -> Vec<f64> {
    let mut output = vec![0f64];
    (0..activations)
        .map(|_| {
            organism.activate(vec![1.0], &mut output);
            output[0]
        })
        .collect()
}

#[test]
fn default_integrator_is_exponential_euler() {
    assert_eq!(
        MutationConfig::default().integrator,
        Integrator::ExponentialEuler
    );
    let organism = Organism::new(Genome::new_unconnected(1, 1));
    assert_eq!(organism.integrator, Integrator::ExponentialEuler);
}

#[test]
fn with_config_takes_the_integrator() {
    assert_eq!(organism(Integrator::Euler).integrator, Integrator::Euler);
}

#[test]
fn explicit_euler_organism_diverges_and_exponential_converges() {
    // The target of the output is w·x = 1. Explicit Euler multiplies the
    // distance to it by 1 - h/τ = -1.5 each of the 10 steps of an
    // activation; exponential Euler by e^(-2.5).
    let euler = outputs(&mut organism(Integrator::Euler), 3);
    let expected = 1.0 - 1.5f64.powi(30);
    assert!(
        (euler[2] - expected).abs() < 1e-6 * expected.abs(),
        "{:?}",
        euler
    );

    let exponential = outputs(&mut organism(Integrator::ExponentialEuler), 3);
    for output in exponential {
        assert!((output - 1.0).abs() < 1e-9, "{}", output);
    }
}

#[test]
fn changing_the_integrator_recompiles_the_phenotype() {
    let mut organism = organism(Integrator::ExponentialEuler);
    outputs(&mut organism, 1);
    organism.reset_state();
    organism.integrator = Integrator::Euler;
    let first = outputs(&mut organism, 1)[0];
    assert!((first - (1.0 - 1.5f64.powi(10))).abs() < 1e-9, "{}", first);
}

#[test]
fn offspring_keep_the_integrator() {
    let config = MutationConfig::builder()
        .integrator(Integrator::Euler)
        .build();
    let parent = organism(Integrator::Euler);
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let mut registry = rustneat::InnovationRegistry::new(2);
    let child = parent.mutate_with_innovations(&config, &mut registry, &mut rng);
    assert_eq!(child.integrator, Integrator::Euler);
    let child = parent.mate_with_config_and_rng(&parent.clone(), &config, &mut rng);
    assert_eq!(child.integrator, Integrator::Euler);
}
