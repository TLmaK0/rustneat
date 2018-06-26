use rulinalg::matrix::{BaseMatrix, BaseMatrixMut, Matrix};
#[allow(missing_docs)]
#[derive(Debug)]
pub struct CtrnnNeuralNetwork<'a> {
    pub gamma: &'a [f64],
    pub delta_t: f64,
    pub tau: &'a [f64],
    pub wij: &'a (usize, usize, &'a [f64]),
    pub theta: &'a [f64],
    pub wik: &'a (usize, usize, &'a [f64]),
    pub i: &'a [f64],
}

#[allow(missing_docs)]
#[derive(Default, Clone, Copy, Debug)]
pub struct Ctrnn {}

impl Ctrnn {
    /// Activate the NN
    // TODO Not sure steps are required here?
    pub fn activate_nn(&self, steps: usize, nn: &CtrnnNeuralNetwork) -> Vec<f64> {
        let mut state = Ctrnn::matrix_from_vector(nn.gamma);
        let theta = Ctrnn::matrix_from_vector(nn.theta);
        let wij = Ctrnn::matrix_from_matrix(nn.wij);
        let wik = Ctrnn::matrix_from_matrix(nn.wik);
        let i = Ctrnn::matrix_from_vector(nn.i);
        let tau = Ctrnn::matrix_from_vector(nn.tau);
        let delta_t_tau = tau.apply(&(|x| 1.0 / x)) * nn.delta_t;
        for _ in 0..steps {
            state = &state
                + delta_t_tau.elemul(
                    &((&wij * (&state - &theta).apply(&Ctrnn::sigmoid)) - &state + (&wik * &i)),
                );
        }
        state
            .apply(&(|x| (x - 3.0) * 2.0))
            .apply(&Ctrnn::sigmoid)
            .into_vec()
    }

    #[allow(missing_docs)]
    #[deprecated(since = "0.1.7", note = "please use `activate_nn` instead")]
    pub fn activate(
        &self,
        steps: usize,
        gamma: &[f64],
        delta_t: f64,
        tau: &[f64],
        wij: &(usize, usize, Vec<f64>),
        theta: &[f64],
        wik: &(usize, usize, Vec<f64>),
        i: &[f64],
    ) -> Vec<f64> {
        self.activate_nn(
            steps,
            &CtrnnNeuralNetwork {
                gamma: gamma,
                delta_t: delta_t,
                tau: tau,
                wij: &(wij.0, wij.1, wij.2.as_slice()),
                theta: theta,
                wik: &(wik.0, wik.1, wik.2.as_slice()),
                i: i,
            },
        )
    }

    fn sigmoid(y: f64) -> f64 {
        1f64 / (1f64 + (-y).exp())
    }

    fn matrix_from_vector(vector: &[f64]) -> Matrix<f64> {
        Matrix::new(vector.len(), 1, vector)
    }

    fn matrix_from_matrix(matrix: &(usize, usize, &[f64])) -> Matrix<f64> {
        Matrix::new(matrix.0, matrix.1, matrix.2)
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
        let delta_t = 13.436;
        let tau = vec![61.694, 10.149, 16.851];
        let wij = (
            3,
            3,
            vec![
                -2.94737, 2.70665, -0.57046, -3.27553, 3.67193, 1.83218, 2.32476, 0.24739, 0.58587,
            ],
        );
        let theta = vec![-0.695126, -0.677891, -0.072129];
        let wik = (
            3,
            2,
            vec![-0.10097, -3.04457, -4.86594, 1.79273, -3.45899, -1.27388],
        );
        let i = vec![0.98856, 0.31540];

        let nn = CtrnnNeuralNetwork {
            gamma: gamma.as_slice(),
            delta_t: delta_t,
            tau: tau.as_slice(),
            wij: &(wij.0, wij.1, wij.2.as_slice()),
            theta: theta.as_slice(),
            wik: &(wik.0, wik.1, wik.2.as_slice()),
            i: i.as_slice(),
        };

        let ctrnn = Ctrnn::default();

        assert_delta_vector!(
            ctrnn.activate_nn(1, &nn),
            vec![
                0.0012732326259646935,
                0.0000007804325967431104,
                0.00013984620250072583,
            ],
            0.00000000000000000001
        );

        assert_delta_vector!(
            ctrnn.activate_nn(2, &nn),
            vec![
                0.00043073019717790323,
                0.000000009937039489593933,
                0.000034080215678448577,
            ],
            0.00000000000000000001
        );

        assert_delta_vector!(
            ctrnn.activate_nn(10, &nn),
            vec![
                0.00007325263764065628,
                0.00000012140174814281648,
                0.000004220860839220797,
            ],
            0.00000000000000000001
        );

        assert_delta_vector!(
            ctrnn.activate_nn(30, &nn),
            vec![
                0.00006952721528466206,
                0.00000012669416324530944,
                0.000004043510745829741,
            ],
            0.00000000000000000001
        );

        // converges
        assert_delta_vector!(
            ctrnn.activate_nn(100, &nn),
            vec![
                0.00006952654167069687,
                0.0000001266951605597891,
                0.000004043479141786699,
            ],
            0.00000000000000000001
        );
    }
}
