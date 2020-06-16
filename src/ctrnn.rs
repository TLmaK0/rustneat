use rulinalg::matrix::{BaseMatrix, BaseMatrixMut, Matrix};

#[cfg(feature = "ctrnn_telemetry")]
use rusty_dashed;

#[cfg(feature = "ctrnn_telemetry")]
use serde_json;

#[allow(missing_docs)]
#[derive(Debug)]
pub struct CtrnnNeuralNetwork<'a> {
    pub y: &'a [f64],     //current state of neuron(j)
    pub tau: &'a [f64], //τ - time constant ( t > 0 ). The neuron's speed of response to an external sensory signal. Membrane resistance time.
    pub wji: &'a [f64], //w - weights of the connection from neuron(j) to neuron(i)
    pub theta: &'a [f64], //θ - bias of the neuron(j)
    pub i: &'a [f64],   //I - external input to neuron(i)
}

#[allow(missing_docs)]
#[derive(Default, Clone, Copy, Debug)]
pub struct Ctrnn {}

impl Ctrnn {
    /// Activate the NN
    pub fn activate_nn(&self, time: f64, step_size: f64, nn: &CtrnnNeuralNetwork) -> Vec<f64> {
        let steps = (time / step_size) as usize;
        let mut y = Ctrnn::vector_to_column_matrix(nn.y);
        let theta = Ctrnn::vector_to_column_matrix(nn.theta);
        let wji = Ctrnn::vector_to_matrix(nn.wji);
        let i = Ctrnn::vector_to_column_matrix(nn.i);
        let tau = Ctrnn::vector_to_column_matrix(nn.tau);

        #[cfg(feature = "ctrnn_telemetry")]
        Ctrnn::telemetry(&y);

        for _ in 0..steps {
            let current_weights = (&y + &theta).apply(&Ctrnn::sigmoid);
            y = &y
                + ((&wji * current_weights) - &y + &i)
                    .elediv(&tau)
                    .apply(&|j_value| step_size * j_value);
            #[cfg(feature = "ctrnn_telemetry")]
            Ctrnn::telemetry(&y);
        }
        y.into_vec()
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

        let ctrnn = Ctrnn::default();

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
