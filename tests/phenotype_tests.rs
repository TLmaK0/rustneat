//! The compiled phenotype (issue #71) gives the same outputs as the dense
//! one it replaces (`examples/perf_probe/reference.rs`), and is recompiled
//! whenever the genome or the CTRNN settings change.

#[path = "../examples/perf_probe/reference.rs"]
mod reference;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use reference::{random_network, random_sensors, DenseOrganism};
use rustneat::{
    Gene, Genome, InnovationRegistry, Integrator, MutationConfig, Organism, OutputActivation,
    Phenotype,
};

const TOLERANCE: f64 = 1e-12;

fn assert_close(compiled: &[f64], dense: &[f64], context: &str) {
    assert_eq!(compiled.len(), dense.len(), "{}", context);
    for (k, (a, b)) in compiled.iter().zip(dense).enumerate() {
        assert!(
            (a - b).abs() < TOLERANCE,
            "{}: output {} is {} (compiled) vs {} (dense)",
            context,
            k,
            a,
            b
        );
    }
}

/// Activate both with the same sensors and compare the outputs.
fn activate_both(
    organism: &mut Organism,
    dense: &mut DenseOrganism,
    sensors: Vec<f64>,
    outputs: usize,
    context: &str,
) {
    let mut compiled_out = vec![0f64; outputs];
    let mut dense_out = vec![0f64; outputs];
    organism.activate(sensors.clone(), &mut compiled_out);
    dense.activate(sensors, &mut dense_out);
    assert_close(&compiled_out, &dense_out, context);
}

const SIZES: [(usize, usize, usize, usize); 6] = [
    // inputs, outputs, neurons, genes
    (1, 1, 2, 1),
    (2, 1, 4, 6),
    (3, 2, 10, 30),
    (4, 2, 30, 120),
    (4, 2, 100, 500),
    (2, 3, 5, 0),
];

#[test]
fn organism_matches_the_dense_phenotype_on_random_networks() {
    let mut rng = ChaCha8Rng::seed_from_u64(71);
    for &(inputs, outputs, neurons, genes) in &SIZES {
        for activation in [OutputActivation::Sigmoid, OutputActivation::Linear] {
            for _ in 0..3 {
                let mut organism =
                    Organism::new(random_network(inputs, outputs, neurons, genes, &mut rng));
                organism.output_activation = activation;
                organism.tau = rng.gen_range(0.005..0.5);
                let mut dense = DenseOrganism::from_organism(&organism);
                for call in 0..40 {
                    // The state persists between calls, except after a reset.
                    if call % 13 == 12 {
                        organism.reset_state();
                        dense.reset_state();
                    }
                    let context = format!(
                        "{} neurons / {} genes, {:?}, call {}",
                        neurons, genes, activation, call
                    );
                    activate_both(
                        &mut organism,
                        &mut dense,
                        random_sensors(inputs, &mut rng),
                        outputs,
                        &context,
                    );
                }
            }
        }
    }
}

#[test]
fn compiled_phenotype_matches_the_dense_ctrnn_with_both_integrators() {
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    for integrator in [Integrator::ExponentialEuler, Integrator::Euler] {
        for &(inputs, outputs, neurons, genes) in &SIZES {
            let genome = random_network(inputs, outputs, neurons, genes, &mut rng);
            let mut organism = Organism::new(genome.clone());
            // Small dt so explicit Euler stays stable (h/τ ≤ 0.2).
            organism.dt = 0.001;
            organism.step_time = 0.05;
            let mut dense = DenseOrganism::from_organism(&organism);
            dense.integrator = integrator;
            let mut phenotype = Phenotype::compile(
                &genome,
                inputs,
                outputs,
                organism.tau,
                organism.dt,
                organism.step_time,
                integrator,
            );
            let mut state = vec![0f64; phenotype.integrated_neurons()];
            for call in 0..20 {
                let sensors = random_sensors(inputs, &mut rng);
                let mut dense_out = vec![0f64; outputs];
                dense.activate(sensors.clone(), &mut dense_out);
                phenotype.activate(&mut state, &sensors);
                let context = format!(
                    "{:?}, {} neurons / {} genes, call {}",
                    integrator, neurons, genes, call
                );
                assert_close(&state, &dense.ctrnn_state, &context);
            }
        }
    }
}

#[test]
fn phenotype_is_recompiled_when_the_genome_or_the_settings_change() {
    let mut rng = ChaCha8Rng::seed_from_u64(3);
    let mut organism = Organism::new(random_network(2, 1, 8, 20, &mut rng));
    let mut dense = DenseOrganism::from_organism(&organism);
    fn step(organism: &mut Organism, dense: &mut DenseOrganism, rng: &mut ChaCha8Rng, what: &str) {
        for _ in 0..3 {
            let sensors = random_sensors(2, rng);
            activate_both(organism, dense, sensors, 1, what);
        }
    }
    step(&mut organism, &mut dense, &mut rng, "initial");

    let both = |organism: &mut Organism, dense: &mut DenseOrganism, f: &dyn Fn(&mut Genome)| {
        f(&mut organism.genome);
        f(&mut dense.genome);
    };
    both(&mut organism, &mut dense, &|g| {
        g.add_gene(Gene::new(0, 2, 3.5, true))
    });
    step(
        &mut organism,
        &mut dense,
        &mut rng,
        "add_gene (new connection)",
    );
    both(&mut organism, &mut dense, &|g| {
        g.add_gene(Gene::new(1, 30, -2.0, true))
    });
    step(&mut organism, &mut dense, &mut rng, "add_gene (new neuron)");
    both(&mut organism, &mut dense, &|g| g.set_bias(2, 1.75));
    step(&mut organism, &mut dense, &mut rng, "set_bias");
    both(&mut organism, &mut dense, &|g| g.set_tau(2, 0.3));
    step(&mut organism, &mut dense, &mut rng, "set_tau");
    both(&mut organism, &mut dense, &|g| {
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        g.mutate_with_config_and_rng(&MutationConfig::default(), &mut rng)
    });
    step(
        &mut organism,
        &mut dense,
        &mut rng,
        "mutate_with_config_and_rng",
    );
    both(&mut organism, &mut dense, &|g| {
        let mut rng = ChaCha8Rng::seed_from_u64(12);
        let mut registry = InnovationRegistry::new(3);
        let config = MutationConfig {
            add_neuron_rate: 1.0,
            ..MutationConfig::default()
        };
        g.mutate_with_innovations(&config, &mut registry, &mut rng)
    });
    step(
        &mut organism,
        &mut dense,
        &mut rng,
        "mutate_with_innovations",
    );

    organism.tau = 0.2;
    dense.tau = 0.2;
    step(&mut organism, &mut dense, &mut rng, "default tau");
    organism.dt = 0.02;
    dense.dt = 0.02;
    step(&mut organism, &mut dense, &mut rng, "dt");
    organism.step_time = 0.3;
    dense.step_time = 0.3;
    step(&mut organism, &mut dense, &mut rng, "step_time");
    organism.output_activation = OutputActivation::Linear;
    dense.output_activation = OutputActivation::Linear;
    step(&mut organism, &mut dense, &mut rng, "output activation");

    // A whole new genome, even one of the same size.
    let other = random_network(2, 1, 8, 20, &mut rng);
    organism.genome = other.clone();
    dense.genome = other;
    step(&mut organism, &mut dense, &mut rng, "replaced genome");

    // More sensors and outputs than the genome knows.
    for (sensors, outputs) in [(3, 1), (3, 2), (2, 1)] {
        let mut a = vec![0f64; outputs];
        let mut b = vec![0f64; outputs];
        let input = random_sensors(sensors, &mut rng);
        organism.activate(input.clone(), &mut a);
        dense.activate(input, &mut b);
        assert_close(&a, &b, &format!("{} sensors, {} outputs", sensors, outputs));
    }
}

#[test]
fn offspring_never_use_the_phenotype_of_their_parents() {
    let mut rng = ChaCha8Rng::seed_from_u64(5);
    let config = MutationConfig {
        add_connection_rate: 1.0,
        add_neuron_rate: 1.0,
        weight_mutation_rate: 1.0,
        bias_mutation_rate: 1.0,
        ..MutationConfig::default()
    };
    let mut parent = Organism::new(random_network(2, 1, 6, 12, &mut rng));
    let mut other = Organism::new(random_network(2, 1, 6, 12, &mut rng));
    other.fitness = 1.0;
    let mut out = vec![0f64];
    parent.activate(vec![0.5, -0.5], &mut out);
    other.activate(vec![0.5, -0.5], &mut out);

    let mut registry = InnovationRegistry::new(3);
    let mut children = vec![
        parent.mutate_with_config_and_rng(&config, &mut rng),
        parent.mutate_with_innovations(&config, &mut registry, &mut rng),
        parent.mate_with_rng(&other, &mut rng),
        other.mate_with_rng(&parent, &mut rng),
    ];
    // A clone shares the phenotype until its genome changes.
    let mut clone = parent.clone();
    clone.genome.add_gene(Gene::new(0, 2, -6.0, true));
    clone.reset_state();
    children.push(clone);

    for (k, child) in children.iter_mut().enumerate() {
        let mut dense = DenseOrganism::from_organism(child);
        for _ in 0..5 {
            let sensors = random_sensors(2, &mut rng);
            activate_both(child, &mut dense, sensors, 1, &format!("child {}", k));
        }
    }
}
