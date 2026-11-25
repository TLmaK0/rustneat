use rulinalg::matrix::{BaseMatrix, BaseMatrixMut, Matrix};

#[allow(missing_docs)]
#[derive(Debug)]
pub struct CtrnnNeuralNetwork<'a> {
    pub y: &'a [f64],     //current state of neuron(j)
    pub tau: &'a [f64], //τ - time constant ( t > 0 ). The neuron's speed of response to an external sensory signal. Membrane resistance time.
    pub wji: &'a [f64], //w - weights of the connection from neuron(j) to neuron(i)
    pub theta: &'a [f64], //θ - bias of the neuron(j)
    pub i: &'a [f64],   //I - external input to neuron(i)
}

/// How `Ctrnn::activate_nn` integrates `τ·dy/dt = -y + W·σ(y + θ) + I`.
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Integrator {
    /// Exponential Euler: with the target `x = W·σ(y + θ) + I` frozen during
    /// a step of length `h`, the exact solution of the linear part,
    /// `y ← y + (1 - e^(-h/τ))·(x - y)`. Each neuron moves a fraction in
    /// (0, 1] of the way to its target, so it is stable for any τ > 0; the
    /// fixed points are those of the ODE. τ → 0 tends to `y ← x`. Default.
    #[default]
    ExponentialEuler,
    /// Explicit (forward) Euler: `y ← y + (h/τ)·(x - y)`. Only stable for
    /// `h/τ < 2` (and without overshoot for `h/τ ≤ 1`): with h = 0.01 and
    /// τ = 0.004 the audit network of issue #69 diverges to 1.7e70. Kept to
    /// reproduce older results.
    Euler,
}

#[allow(missing_docs)]
#[derive(Default, Clone, Copy, Debug)]
pub struct Ctrnn {
    /// Integration scheme (default `Integrator::ExponentialEuler`).
    pub integrator: Integrator,
}

impl Ctrnn {
    /// A CTRNN integrated with `integrator`.
    pub fn with_integrator(integrator: Integrator) -> Ctrnn {
        Ctrnn { integrator }
    }

    /// Activate the NN: integrate `time` seconds in steps of about
    /// `step_size` (dt). The number of steps is `round(time / step_size)`
    /// (at least 1 if `time > 0`) and each step lasts exactly
    /// `time / steps`, so the simulated time is always `time`.
    ///
    /// A change travels one connection per step (each step reads `σ(y + θ)`
    /// of the previous one), so after one call a neuron sees what happened
    /// at most `steps` connections upstream.
    pub fn activate_nn(&self, time: f64, step_size: f64, nn: &CtrnnNeuralNetwork) -> Vec<f64> {
        let steps = Ctrnn::steps(time, step_size);
        let mut y = Ctrnn::vector_to_column_matrix(nn.y);
        if steps == 0 {
            return y.into_vec();
        }
        let h = time / steps as f64;
        let theta = Ctrnn::vector_to_column_matrix(nn.theta);
        let wji = Ctrnn::vector_to_matrix(nn.wji);
        let i = Ctrnn::vector_to_column_matrix(nn.i);
        let tau = Ctrnn::vector_to_column_matrix(nn.tau);
        // Exponential Euler: fraction of the way to the target each neuron
        // moves per step, 1 - e^(-h/τ).
        let rate: Vec<f64> = nn.tau.iter().map(|&tau| -(-h / tau).exp_m1()).collect();
        let rate = Ctrnn::vector_to_column_matrix(&rate);

        #[cfg(feature = "ctrnn_telemetry")]
        Ctrnn::telemetry(&y);

        for _ in 0..steps {
            let current_weights = (&y + &theta).apply(&Ctrnn::sigmoid);
            let towards_target = (&wji * current_weights) - &y + &i;
            y = match self.integrator {
                Integrator::ExponentialEuler => &y + towards_target.elemul(&rate),
                Integrator::Euler => &y + towards_target.elediv(&tau).apply(&|j_value| h * j_value),
            };
            #[cfg(feature = "ctrnn_telemetry")]
            Ctrnn::telemetry(&y);
        }
        y.into_vec()
    }

    /// Number of integration steps of `activate_nn(time, step_size, ..)`.
    pub fn steps(time: f64, step_size: f64) -> usize {
        if time > 0f64 {
            ((time / step_size).round() as usize).max(1)
        } else {
            0
        }
    }

    /// Calculates sigmoid of a number
    pub fn sigmoid(x: f64) -> f64 {
        1f64 / (1f64 + (-x).exp())
    }

    fn vector_to_column_matrix(vector: &[f64]) -> Matrix<f64> {
        Matrix::new(vector.len(), 1, vector)
    }

    fn vector_to_matrix(vector: &[f64]) -> Matrix<f64> {
        let width = (vector.len() as f64).sqrt() as usize;
        Matrix::new(width, width, vector)
    }

    #[cfg(feature = "ctrnn_telemetry")]
    pub(crate) fn telemetry_values(y: &[f64]) {
        telemetry!("ctrnn1", 1.0, serde_json::to_string(y).unwrap());
    }

    #[cfg(feature = "ctrnn_telemetry")]
    fn telemetry(y: &Matrix<f64>) {
        let y2 = y.clone();
        telemetry!(
            "ctrnn1",
            1.0,
            serde_json::to_string(&y2.into_vec()).unwrap()
        );
    }
}

#[cfg(test)]
// `!(a <= d)` on purpose: a NaN difference must fail the assertion.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
mod tests {
    use super::*;
    macro_rules! assert_delta_vector {
        ($x:expr, $y:expr, $d:expr) => {
            for pos in 0..$x.len() {
                if !(($x[pos] - $y[pos]).abs() <= $d) {
                    panic!(
                        "Element at position {:?} -> {:?} \
                         is not equal to {:?}",
                        pos, $x[pos], $y[pos]
                    );
                }
            }
        };
    }

    /// Network of the audit (issue #69): neuron 0 is driven by I = 1,
    /// neuron 1 gets 0 -> 1 (w = 1) and itself (w = 0.5). Fixed point of
    /// neuron 1: y = σ(1) + 0.5·σ(y) = 1.1068.
    fn audit_network_output(tau: f64, activations: usize) -> f64 {
        let wji = vec![0.0, 0.0, 1.0, 0.5];
        let theta = vec![0.0, 0.0];
        let i = vec![1.0, 0.0];
        let taus = vec![tau, tau];
        let mut y = vec![0.0, 0.0];
        for _ in 0..activations {
            y = Ctrnn::default().activate_nn(
                0.1,
                0.01,
                &CtrnnNeuralNetwork {
                    y: &y,
                    tau: &taus,
                    wji: &wji,
                    theta: &theta,
                    i: &i,
                },
            );
        }
        y[1]
    }

    #[test]
    fn small_tau_stays_finite_and_reaches_the_same_fixed_point() {
        // With explicit Euler dt/τ = 2.5 diverged to 1.7e70 (issue #69).
        let reference = audit_network_output(0.02, 40);
        assert!((reference - 1.1068).abs() < 1e-3, "{}", reference);
        for tau in [0.004, 0.001, 1e-6] {
            for activations in [1, 5, 40] {
                let output = audit_network_output(tau, activations);
                assert!(output.is_finite(), "tau {}: {}", tau, output);
            }
            let output = audit_network_output(tau, 40);
            assert!(
                (output - reference).abs() < 1e-3,
                "tau {}: {} vs {}",
                tau,
                output,
                reference
            );
        }
    }

    #[test]
    fn neural_network_activation_should_return_correct_values() {
        let gamma = vec![0.0, 0.0, 0.0];
        let tau = vec![61.694, 10.149, 16.851];
        let wji = vec![
            -2.94737, 2.70665, -0.57046, -3.27553, 3.67193, 1.83218, 2.32476, 0.24739, 0.58587,
        ];
        let theta = vec![-0.695126, -0.677891, -0.072129];
        let i = vec![0.98856, 0.31540, 0.0];

        let nn = CtrnnNeuralNetwork {
            y: &gamma,
            tau: &tau,
            wji: &wji,
            theta: &theta,
            i: &i,
        };

        // Reference values computed with explicit Euler.
        let ctrnn = Ctrnn::with_integrator(Integrator::Euler);

        assert_delta_vector!(
            ctrnn.activate_nn(1.0, 0.1, &nn),
            vec![
                0.010829986965909134,
                0.1324987329841768,
                0.06644643156742948
            ],
            0.00000000000000000001
        );

        assert_delta_vector!(
            ctrnn.activate_nn(2.0, 0.1, &nn),
            vec![
                0.02255533337532507,
                0.26518982989312406,
                0.13038140193371967
            ],
            0.00000000000000000001
        );

        assert_delta_vector!(
            ctrnn.activate_nn(10.0, 0.1, &nn),
            vec![0.14934191797049204, 1.3345894864370869, 0.5691613026150651],
            0.00000000000000000001
        );

        assert_delta_vector!(
            ctrnn.activate_nn(30.0, 0.1, &nn),
            vec![0.5583616282859531, 3.149231725259237, 1.3050168324825089],
            0.00000000000000000001
        );

        // converges
        assert_delta_vector!(
            ctrnn.activate_nn(100.0, 0.1, &nn),
            vec![1.1121375647080136, 3.43423133661062, 2.0992832630144376],
            0.00000000000000000001
        );
    }
}
