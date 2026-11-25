//! Reference for the compiled phenotype (issue #71), shared by
//! `examples/perf_probe` and `tests/phenotype_tests.rs`:
//!
//! - `DenseOrganism`: the dense phenotype `Organism::activate` used before
//!   #71, kept verbatim as the reference the compiled one must match. Every
//!   call rebuilds the N×N weight matrix, the bias and τ vectors and runs
//!   `Ctrnn::activate_nn` (rulinalg).
//! - `random_network`: a random genome with a given number of neurons and
//!   genes (hidden ids with gaps, self-loops, cycles, disabled genes,
//!   connections into inputs, biases and τ genes).
#![allow(dead_code)]

use rand::Rng;
use rustneat::{Ctrnn, CtrnnNeuralNetwork, Gene, Genome, Integrator, Organism, OutputActivation};

/// The phenotype of an `Organism` before issue #71 (dense, rebuilt on every
/// activation), with its own CTRNN state.
#[derive(Clone)]
pub struct DenseOrganism {
    pub genome: Genome,
    pub tau: f64,
    pub dt: f64,
    pub step_time: f64,
    pub output_activation: OutputActivation,
    pub integrator: Integrator,
    pub ctrnn_state: Vec<f64>,
}

impl DenseOrganism {
    /// Same genome and CTRNN settings as `organism`, from rest.
    pub fn from_organism(organism: &Organism) -> DenseOrganism {
        DenseOrganism {
            genome: organism.genome.clone(),
            tau: organism.tau,
            dt: organism.dt,
            step_time: organism.step_time,
            output_activation: organism.output_activation,
            integrator: Integrator::ExponentialEuler,
            ctrnn_state: vec![],
        }
    }

    pub fn reset_state(&mut self) {
        self.ctrnn_state = vec![];
    }

    /// `Organism::activate` as it was before issue #71.
    pub fn activate(&mut self, sensors: Vec<f64>, outputs: &mut [f64]) {
        let inputs_len = sensors.len();
        let fixed_len = inputs_len + outputs.len();
        // Neuron ids of the network, in id order: inputs, outputs, hidden.
        let ids: Vec<usize> = (0..fixed_len)
            .chain(
                self.genome
                    .neurons()
                    .into_iter()
                    .filter(|&id| id >= fixed_len),
            )
            .collect();
        let neurons_len = ids.len();
        // Integrated neurons: outputs and hidden, in neuron id order.
        let integrated_len = neurons_len - inputs_len;

        let weights = self.get_weights(&ids);
        let mut wji = vec![0.0; integrated_len * integrated_len];
        let mut i = vec![0.0; integrated_len];
        for row in 0..integrated_len {
            let to = (inputs_len + row) * neurons_len;
            for (from, sensor) in sensors.iter().enumerate() {
                i[row] += weights[to + from] * sensor;
            }
            wji[row * integrated_len..(row + 1) * integrated_len]
                .copy_from_slice(&weights[to + inputs_len..to + neurons_len]);
        }
        let theta: Vec<f64> = ids[inputs_len..]
            .iter()
            .map(|&id| self.genome.bias(id))
            .collect();
        let tau: Vec<f64> = ids[inputs_len..]
            .iter()
            .map(|&id| self.genome.tau(id).unwrap_or(self.tau))
            .collect();

        if self.ctrnn_state.len() != integrated_len {
            self.ctrnn_state = vec![0.0; integrated_len];
        }
        if integrated_len == 0 {
            return;
        }

        let activations = Ctrnn::with_integrator(self.integrator).activate_nn(
            self.step_time,
            self.dt,
            &CtrnnNeuralNetwork {
                y: &self.ctrnn_state,
                tau: &tau,
                wji: &wji,
                theta: &theta,
                i: &i,
            },
        );

        for (n, output) in outputs.iter_mut().enumerate() {
            let value = activations[n] + theta[n];
            *output = match self.output_activation {
                OutputActivation::Sigmoid => Ctrnn::sigmoid(value),
                OutputActivation::Linear => value,
            };
        }
        self.ctrnn_state = activations;
    }

    /// Connection weights as a square matrix over the neurons `ids`
    /// (row = destination, column = source, by position in `ids`).
    fn get_weights(&self, ids: &[usize]) -> Vec<f64> {
        let neurons_len = ids.len();
        let position = |id: usize| ids.binary_search(&id).ok();
        let mut matrix = vec![0.0; neurons_len * neurons_len];
        for gene in self.genome.get_genes() {
            if gene.enabled() {
                if let (Some(from), Some(to)) = (
                    position(gene.in_neuron_id()),
                    position(gene.out_neuron_id()),
                ) {
                    matrix[to * neurons_len + from] = gene.weight()
                }
            }
        }
        matrix
    }
}

/// A random genome with `inputs` inputs, `outputs` outputs and `neurons`
/// neurons in total (hidden ids are spread with gaps, as with an innovation
/// registry), and `genes` distinct connections: every hidden neuron gets
/// one incoming connection, the rest join random pairs (self-loops, cycles
/// and connections into inputs included). About 10 % of the genes are
/// disabled; every neuron gets a random bias and about half a τ gene.
pub fn random_network<R: Rng>(
    inputs: usize,
    outputs: usize,
    neurons: usize,
    genes: usize,
    rng: &mut R,
) -> Genome {
    let fixed = inputs + outputs;
    assert!(neurons >= fixed);
    let hidden: Vec<usize> = (0..neurons - fixed).map(|k| fixed + 3 * k + 1).collect();
    let ids: Vec<usize> = (0..fixed).chain(hidden.iter().copied()).collect();
    let mut genome = Genome::new_unconnected(inputs, outputs);
    let mut pairs = std::collections::HashSet::new();
    let mut added = 0;
    let mut add = |genome: &mut Genome, from: usize, to: usize, rng: &mut R| {
        if pairs.insert((from, to)) {
            let enabled = rng.gen::<f64>() >= 0.1;
            genome.add_gene(Gene::new(from, to, rng.gen_range(-4f64..4f64), enabled));
            1
        } else {
            0
        }
    };
    for (k, &id) in hidden.iter().enumerate() {
        // From an input, an output or an earlier hidden neuron.
        let from = ids[rng.gen_range(0..fixed + k)];
        added += add(&mut genome, from, id, rng);
    }
    let max_pairs = neurons * neurons;
    while added < genes.min(max_pairs) {
        let from = ids[rng.gen_range(0..ids.len())];
        let to = ids[rng.gen_range(0..ids.len())];
        added += add(&mut genome, from, to, rng);
    }
    for &id in &ids {
        genome.set_bias(id, rng.gen_range(-2f64..2f64));
        if rng.gen::<bool>() {
            genome.set_tau(id, rng.gen_range(0.005f64..0.5f64));
        }
    }
    genome
}

/// Random sensor values in [-2, 2].
pub fn random_sensors<R: Rng>(inputs: usize, rng: &mut R) -> Vec<f64> {
    (0..inputs).map(|_| rng.gen_range(-2f64..2f64)).collect()
}
