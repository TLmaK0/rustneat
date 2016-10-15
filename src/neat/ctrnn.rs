extern crate rulinalg;

use self::rulinalg::matrix::Matrix;
use self::rulinalg::matrix::BaseMatrix;
use self::rulinalg::matrix::BaseMatrixMut;

pub struct Ctrnn {
}

impl Ctrnn {
    pub fn new() -> Ctrnn{
        Ctrnn {}
    }

    pub fn activate(&self, steps: usize, gamma_v: &Vec<f64>, delta_t: f64, tau_v: &Vec<f64>, wij_v: &(usize, usize, Vec<f64>), theta_v: &Vec<f64>, wik_v: &(usize, usize, Vec<f64>), i_v: &Vec<f64>) -> Vec<f64> {
        let mut state = Ctrnn::matrix_from_vector(gamma_v); 
        let theta = Ctrnn::matrix_from_vector(theta_v); 
        let wij = Ctrnn::matrix_from_matrix(wij_v); 
        let wik = Ctrnn::matrix_from_matrix(wik_v); 
        let i = Ctrnn::matrix_from_vector(i_v); 
        let tau = Ctrnn::matrix_from_vector(tau_v);
        let delta_t_tau = tau.apply( &(|x| 1.0/x) ) * delta_t;
        for _ in 0..steps { 
            state = &state + delta_t_tau.elemul(&( (&wij * ( &state - &theta ).apply(&Ctrnn::sigmoid)) - &state + (&wik * &i)));
        }
        return state.apply(&(|x| (x - 3.0) * 2.0)).apply(&Ctrnn::sigmoid).into_vec();
    }

    fn sigmoid(y: f64) -> f64 {
        1f64 / (1f64 + (-y).exp())
    }

    fn matrix_from_vector(vector: &Vec<f64>) -> Matrix<f64> {
        Matrix::new(vector.len(), 1, vector.as_slice())
    }

    fn matrix_from_matrix(matrix: &(usize, usize, Vec<f64>)) -> Matrix<f64> {
        Matrix::new(matrix.0, matrix.1, matrix.2.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use neat::*;

    #[test]
    fn neural_network_activation_should_return_correct_values(){
        let gamma =         vec![0.0, 0.0, 0.0];
        let delta_t =       13.436;
        let tau =           vec![61.694, 10.149, 16.851];
        let wij =           (3, 3, vec![-2.94737, 2.70665, -0.57046,
                            -3.27553,   3.67193, 1.83218,
                            2.32476,   0.24739, 0.58587]);
        let theta =         vec![-0.695126, -0.677891, -0.072129];
        let wik =           (3, 2, vec![-0.10097, -3.04457,
                            -4.86594, 1.79273,
                            -3.45899, -1.27388]);
        let i =             vec![0.98856, 0.31540];

        let ctrnn = Ctrnn::new();

        assert_eq!( ctrnn.activate(1, &gamma, delta_t, &tau, &wij, &theta, &wik, &i),
            vec![0.0012732326259646935, 0.0000007804325967431104, 0.00013984620250072583]);

        assert_eq!( ctrnn.activate(2, &gamma, delta_t, &tau, &wij, &theta, &wik, &i),
            vec![0.00043073019717790323, 0.000000009937039489593933, 0.000034080215678448577]);

        assert_eq!( ctrnn.activate(10, &gamma, delta_t, &tau, &wij, &theta, &wik, &i),
            vec![0.00007325263764065628, 0.00000012140174814281648, 0.000004220860839220797]);

        assert_eq!( ctrnn.activate(30, &gamma, delta_t, &tau, &wij, &theta, &wik, &i),
            vec![0.00006952721528466206, 0.00000012669416324530944, 0.000004043510745829741]);

        //converges
        assert_eq!( ctrnn.activate(100, &gamma, delta_t, &tau, &wij, &theta, &wik, &i),
            vec![0.00006952654167069687, 0.0000001266951605597891, 0.000004043479141786699]);
    }
}
