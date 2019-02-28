use rulinalg::matrix::{BaseMatrix, BaseMatrixMut, Matrix};

#[allow(missing_docs)]
#[derive(Debug)]
pub struct Ctrnn<'a> {
    pub y: &'a [f64],
    pub delta_t: f64,
    pub tau: &'a [f64], //time constant
    pub wij: &'a [f64], //weights
    pub theta: &'a [f64], //bias
    pub i: &'a [f64], //sensors
}


#[allow(missing_docs)]
impl<'a> Ctrnn<'a> {
    pub fn activate_nn(&self, steps: usize) -> Vec<f64> {
        let mut y = Ctrnn::vector_to_column_matrix(self.y);
        let theta = Ctrnn::vector_to_column_matrix(self.theta);
        let wij = Ctrnn::vector_to_matrix(self.wij);
        let i = Ctrnn::vector_to_column_matrix(self.i);
        let tau = Ctrnn::vector_to_column_matrix(self.tau);
        let delta_t_tau = tau.apply(&(|x| 1.0 / x)) * self.delta_t;

        for _ in 0..steps {
            let activations = (&y + &theta).apply(&Ctrnn::sigmoid);
            y = &y + delta_t_tau.elemul(
                &((&wij * activations) - &y + &i)
            );
        };
        y.into_vec()
    }

    fn sigmoid(y: f64) -> f64 {
        1.0 / (1.0 + (-y).exp())
    }

    fn vector_to_column_matrix(vector: &[f64]) -> Matrix<f64> {
        Matrix::new(vector.len(), 1, vector)
    }

    fn vector_to_matrix(vector: &[f64]) -> Matrix<f64> {
        let width = (vector.len() as f64).sqrt() as usize;
        Matrix::new(width, width, vector)
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
    fn neural_network_activation_stability() {
        // TODO
        // This test should just ensure that a stable neural network implementation doesn't change
    }
}
