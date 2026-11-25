use crate::ctrnn::{Ctrnn, Integrator};
use crate::genome::{Genome, Revision};
use crate::organism::Organism;
use std::fmt;

/// The CTRNN of a genome, compiled once and activated many times.
///
/// `Organism::activate` compiles it on the first call and keeps it until the
/// genome or the CTRNN settings change. It is the same network that
/// `Ctrnn::activate_nn` integrates with a dense matrix (see
/// `Organism::activate` for the neurons it has), stored sparsely: only the
/// enabled connections, with the neurons renumbered compactly, and with
/// the buffers of an activation allocated once.
///
/// Integrated neurons (outputs, then hidden neurons by id) are numbered
/// `0..integrated_neurons()`; the state `y` of a network is one value per
/// integrated neuron. Inputs are sources: each activation adds `w · x` of
/// every input connection to the external input `I` of its target.
#[derive(Clone)]
pub struct Phenotype {
    revision: Revision,
    inputs: usize,
    outputs: usize,
    default_tau: f64,
    dt: f64,
    step_time: f64,
    integrator: Integrator,
    /// Connections from an input, `(target, input, weight)`, by target and
    /// then input.
    input_connections: Vec<(usize, usize, f64)>,
    /// Connections between integrated neurons in compressed rows: those
    /// into neuron `t` are `sources[rows[t]..rows[t + 1]]` (increasing)
    /// with `weights[rows[t]..rows[t + 1]]`.
    rows: Vec<usize>,
    sources: Vec<usize>,
    weights: Vec<f64>,
    /// Bias θ of each integrated neuron.
    theta: Vec<f64>,
    /// τ of each integrated neuron.
    tau: Vec<f64>,
    /// Exponential Euler: fraction of the way to its target each neuron
    /// moves per step, `1 - e^(-h/τ)`.
    rate: Vec<f64>,
    steps: usize,
    /// Step length, `step_time / steps`.
    h: f64,
    /// Buffers of an activation: external input `I` and `σ(y + θ)`.
    external: Vec<f64>,
    emitted: Vec<f64>,
}

impl Phenotype {
    /// Compile the network of `genome` with `inputs` input neurons and
    /// `outputs` output neurons (the first ids), `default_tau` for the
    /// neurons without τ gene, and an activation that integrates
    /// `step_time` seconds in steps of about `dt` with `integrator`.
    pub fn compile(
        genome: &Genome,
        inputs: usize,
        outputs: usize,
        default_tau: f64,
        dt: f64,
        step_time: f64,
        integrator: Integrator,
    ) -> Phenotype {
        let fixed = inputs + outputs;
        // Neuron ids of the network, in id order: inputs, outputs, hidden.
        let ids: Vec<usize> = (0..fixed)
            .chain(genome.neurons().into_iter().filter(|&id| id >= fixed))
            .collect();
        let integrated = ids.len() - inputs;
        let position = |id: usize| ids.binary_search(&id).ok();

        // (target, source position in `ids`, weight), in gene order.
        let mut connections: Vec<(usize, usize, f64)> = genome
            .get_genes()
            .iter()
            .filter(|gene| gene.enabled())
            .filter_map(|gene| {
                match (
                    position(gene.in_neuron_id()),
                    position(gene.out_neuron_id()),
                ) {
                    // Connections into an input have no effect.
                    (Some(from), Some(to)) if to >= inputs => {
                        Some((to - inputs, from, gene.weight()))
                    }
                    _ => None,
                }
            })
            .collect();
        // Sources in increasing order, as the dense product adds them; of
        // two genes with the same ends the last one counts, as in the dense
        // matrix (stable sort).
        connections.sort_by_key(|&(to, from, _)| (to, from));
        let mut unique: Vec<(usize, usize, f64)> = Vec::with_capacity(connections.len());
        for connection in connections {
            match unique.last_mut() {
                Some(last) if (last.0, last.1) == (connection.0, connection.1) => {
                    *last = connection
                }
                _ => unique.push(connection),
            }
        }

        let mut input_connections = vec![];
        let mut rows = Vec::with_capacity(integrated + 1);
        let mut sources = vec![];
        let mut weights = vec![];
        let mut next = unique.iter().peekable();
        for target in 0..integrated {
            rows.push(sources.len());
            while let Some(&&(to, from, weight)) = next.peek() {
                if to != target {
                    break;
                }
                if from < inputs {
                    input_connections.push((to, from, weight));
                } else {
                    sources.push(from - inputs);
                    weights.push(weight);
                }
                next.next();
            }
        }
        rows.push(sources.len());

        let theta: Vec<f64> = ids[inputs..].iter().map(|&id| genome.bias(id)).collect();
        let tau: Vec<f64> = ids[inputs..]
            .iter()
            .map(|&id| genome.tau(id).unwrap_or(default_tau))
            .collect();
        let steps = Ctrnn::steps(step_time, dt);
        let h = if steps > 0 {
            step_time / steps as f64
        } else {
            0f64
        };
        let rate = tau.iter().map(|&tau| -(-h / tau).exp_m1()).collect();

        Phenotype {
            revision: genome.revision(),
            inputs,
            outputs,
            default_tau,
            dt,
            step_time,
            integrator,
            input_connections,
            rows,
            sources,
            weights,
            theta,
            tau,
            rate,
            steps,
            h,
            external: vec![0f64; integrated],
            emitted: vec![0f64; integrated],
        }
    }

    /// True if this is the phenotype `Organism::activate` compiles for
    /// `organism` (its genome as it is now and its CTRNN settings) with
    /// `inputs` sensors and `outputs` outputs.
    pub(crate) fn is_compiled_for(
        &self,
        organism: &Organism,
        inputs: usize,
        outputs: usize,
    ) -> bool {
        self.revision == organism.genome.revision()
            && self.inputs == inputs
            && self.outputs == outputs
            && self.default_tau.to_bits() == organism.tau.to_bits()
            && self.dt.to_bits() == organism.dt.to_bits()
            && self.step_time.to_bits() == organism.step_time.to_bits()
            && self.integrator == Organism::INTEGRATOR
    }

    /// Number of integrated neurons (outputs and hidden): the length of
    /// the state.
    pub fn integrated_neurons(&self) -> usize {
        self.theta.len()
    }

    /// Address of the buffers: it changes only when compiled again.
    #[cfg(test)]
    pub(crate) fn buffers_address(&self) -> *const f64 {
        self.emitted.as_ptr()
    }

    /// Bias θ of the integrated neuron `neuron`.
    pub(crate) fn bias(&self, neuron: usize) -> f64 {
        self.theta[neuron]
    }

    /// Integrate the network from `state` (one value per integrated
    /// neuron, updated in place) with the input values `sensors` (one per
    /// input neuron), exactly as `Ctrnn::activate_nn` does with the dense
    /// matrix: `round(step_time / dt)` steps of
    /// `y ← y + (1 - e^(-h/τ))·(W·σ(y + θ) - y + I)` (exponential Euler) or
    /// `y ← y + h·((W·σ(y + θ) - y + I) / τ)` (explicit Euler).
    ///
    /// # Panics
    ///
    /// If `state` has not `integrated_neurons()` values or `sensors` has
    /// fewer values than input neurons.
    pub fn activate(&mut self, state: &mut [f64], sensors: &[f64]) {
        assert_eq!(state.len(), self.integrated_neurons(), "state length");
        assert!(sensors.len() >= self.inputs, "fewer sensors than inputs");
        if self.steps == 0 {
            return;
        }
        for value in self.external.iter_mut() {
            *value = 0f64;
        }
        for &(to, from, weight) in &self.input_connections {
            self.external[to] += weight * sensors[from];
        }

        #[cfg(feature = "ctrnn_telemetry")]
        Ctrnn::telemetry_values(state);

        for _ in 0..self.steps {
            for ((emitted, &y), &theta) in
                self.emitted.iter_mut().zip(state.iter()).zip(&self.theta)
            {
                *emitted = Ctrnn::sigmoid(y + theta);
            }
            for (target, y) in state.iter_mut().enumerate() {
                let (start, end) = (self.rows[target], self.rows[target + 1]);
                let mut net = 0f64;
                for (&source, &weight) in self.sources[start..end]
                    .iter()
                    .zip(&self.weights[start..end])
                {
                    net += weight * self.emitted[source];
                }
                let towards_target = net - *y + self.external[target];
                *y = match self.integrator {
                    Integrator::ExponentialEuler => *y + towards_target * self.rate[target],
                    Integrator::Euler => *y + self.h * (towards_target / self.tau[target]),
                };
            }
            #[cfg(feature = "ctrnn_telemetry")]
            Ctrnn::telemetry_values(state);
        }
    }
}

impl fmt::Debug for Phenotype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Phenotype")
            .field("integrated_neurons", &self.integrated_neurons())
            .field("input_connections", &self.input_connections.len())
            .field("connections", &self.sources.len())
            .field("steps", &self.steps)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gene::Gene;

    #[test]
    fn only_enabled_connections_between_network_neurons_are_compiled() {
        // 1 input (0), 1 output (1), hidden 2 and 40.
        let mut genome = Genome::new_unconnected(1, 1);
        genome.add_gene(Gene::new(0, 1, 1.0, true));
        genome.add_gene(Gene::new(0, 40, 2.0, true));
        genome.add_gene(Gene::new(40, 2, 3.0, true));
        genome.add_gene(Gene::new(2, 1, 4.0, false));
        genome.add_gene(Gene::new(2, 2, 5.0, true));
        genome.add_gene(Gene::new(1, 0, 6.0, true));
        let phenotype = Phenotype::compile(&genome, 1, 1, 0.01, 0.01, 0.1, Integrator::default());
        assert_eq!(phenotype.integrated_neurons(), 3);
        // Integrated neurons: 1 -> 0, 2 -> 1, 40 -> 2.
        assert_eq!(phenotype.input_connections, vec![(0, 0, 1.0), (2, 0, 2.0)]);
        assert_eq!(phenotype.rows, vec![0, 0, 2, 2]);
        assert_eq!(phenotype.sources, vec![1, 2]);
        assert_eq!(phenotype.weights, vec![5.0, 3.0]);
    }
}
