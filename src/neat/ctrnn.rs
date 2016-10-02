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

    pub fn activate(&self, steps: usize, gamma_v: &Vec<f64>, delta_t: f64, tau_v: &Vec<f64>, wij_v: &Vec<Vec<f64>>, theta_v: &Vec<f64>, wik_v: &Vec<Vec<f64>>, i_v: &Vec<f64>) -> Vec<f64> {
        let mut gamma = Ctrnn::matrix_from_vector(gamma_v); 
        let theta = Ctrnn::matrix_from_vector(theta_v); 
        let wij = Ctrnn::matrix_from_vectors(wij_v); 
        let wik = Ctrnn::matrix_from_vectors(wik_v); 
        let i = Ctrnn::matrix_from_vector(i_v); 
        let tau = Ctrnn::matrix_from_vector(tau_v);
        let delta_t_tau = tau.apply( &(|x| 1.0/x) ) * delta_t;
        for _ in 0..steps { 
            gamma = &gamma + delta_t_tau.elemul(&( (&wij * ( &gamma - &theta ).apply(&Ctrnn::sigmoid)) - &gamma + (&wik * &i)));
        }
        return gamma.into_vec();
    }

    fn sigmoid(y: f64) -> f64 {
        1f64 / (1f64 + (-y).exp())
    }

    fn matrix_from_vector(vector: &Vec<f64>) -> Matrix<f64> {
        Matrix::new(vector.len(), 1, vector.as_slice())
    }

    fn matrix_from_vectors(vectors: &Vec<Vec<f64>>) -> Matrix<f64> {
        Matrix::new(vectors.len(), vectors[1].len(), vectors.iter().flat_map(|s| s.iter().cloned()).collect::<Vec<f64>>())
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
        let wij =           vec![vec![-2.94737, 2.70665, -0.57046],
                            vec![-3.27553,   3.67193, 1.83218],
                            vec![2.32476,   0.24739, 0.58587]];
        let theta =         vec![-0.695126, -0.677891, -0.072129];
        let wik =           vec![vec![-0.10097, -3.04457],
                            vec![-4.86594, 1.79273],
                            vec![-3.45899, -1.27388]];
        let i =             vec![0.98856, 0.31540];

        let ctrnn = Ctrnn::new();

        assert_eq!( ctrnn.activate(1, &gamma, delta_t, &tau, &wij, &theta, &wik, &i),
            vec![-0.3324610969203632, -4.031708339680356, -1.4374137183957871]);

        assert_eq!( ctrnn.activate(2, &gamma, delta_t, &tau, &wij, &theta, &wik, &i),
            vec![-0.8747989167571583, -6.213498344386237, -2.143379723316894]);
        assert_eq!( ctrnn.activate(10, &gamma, delta_t, &tau, &wij, &theta, &wik, &i),
            vec![-1.7607615233242453, -4.96208021858224, -3.1877336197565627]);
        assert_eq!( ctrnn.activate(30, &gamma, delta_t, &tau, &wij, &theta, &wik, &i),
            vec![-1.7868613827907782, -4.940744845752775, -3.2091966016693183]);

        //converges
        assert_eq!( ctrnn.activate(100, &gamma, delta_t, &tau, &wij, &theta, &wik, &i),
            vec![-1.7868662273977671, -4.940740909854486, -3.2092005096958243]);
    }
}
