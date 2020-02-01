use rulinalg::matrix::{BaseMatrix, BaseMatrixMut, Matrix};

/// Continuous Time Recurrent Neural Network implementation, which the
/// `NeuralNetwork` genome encodes for.
#[allow(missing_docs)]
#[derive(Debug, Clone)]
pub struct Ctrnn {
    theta: Matrix<f64>, // bias
    delta_t_tau: Matrix<f64>,
    wij: Matrix<f64>, // weights
    steps: usize,
}

impl Ctrnn {
    /// Create a new CTRNN
    pub fn new(theta: Vec<f64>, tau: Vec<f64>, wij: Vec<f64>, delta_t: f64, steps: usize) -> Ctrnn {
        let tau = Ctrnn::vector_to_column_matrix(tau);
        Ctrnn {
            theta: Ctrnn::vector_to_column_matrix(theta),
            wij: Ctrnn::vector_to_matrix(wij),
            delta_t_tau: tau.apply(&(|x| 1.0 / x)) * delta_t,
            steps,
        }
    }
    /// Activate the neural network. The output is written to `output`, the
    /// amount depending on the length of `output`.
    pub fn activate(&self, mut input: Vec<f64>, output: &mut [f64]) {
        let n_inputs = input.len();
        let n_neurons = self.theta.rows();
        if n_neurons < n_inputs {
            input.truncate(n_neurons);
        } else {
            input = [input, vec![0.0; n_neurons - n_inputs]].concat();
        }

        let input = Ctrnn::vector_to_column_matrix(input);
        let mut y = input.clone(); // TODO: correct? Or zero-vector?
        for _ in 0..self.steps {
            let activations = (&y + &self.theta).apply(&Ctrnn::sigmoid);
            y = &y
                + self
                    .delta_t_tau
                    .elemul(&((&self.wij * activations) - &y + &input));
        }
        let y = y.into_vec();

        if n_inputs < n_neurons {
            let outputs_activations = y.split_at(n_inputs).1.to_vec();

            let len = std::cmp::min(outputs_activations.len(), output.len());

            output[..len].clone_from_slice(&outputs_activations[..len])
        }
    }

    fn sigmoid(y: f64) -> f64 {
        // Inspired from neat-python
        let y = y * 5.0;
        let y = if y < -60.0 {
            -60.0
        } else if y > 60.0 {
            60.0
        } else {
            y
        };
        1.0 / (1.0 + (-y).exp())
    }

    fn vector_to_column_matrix(vector: Vec<f64>) -> Matrix<f64> {
        Matrix::new(vector.len(), 1, vector)
    }

    fn vector_to_matrix(vector: Vec<f64>) -> Matrix<f64> {
        let width = (vector.len() as f64).sqrt() as usize;
        Matrix::new(width, width, vector)
    }
}
