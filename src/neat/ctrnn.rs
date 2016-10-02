extern crate rulinalg;

use self::rulinalg::vector::Vector;
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
        let mut theta = Ctrnn::matrix_from_vector(theta_v); 
        let mut wij = Ctrnn::matrix_from_vectors(wij_v); 
        let mut wik = Ctrnn::matrix_from_vectors(wik_v); 
        let mut i = Ctrnn::matrix_from_vector(i_v); 
        let mut tau = Ctrnn::matrix_from_vector(tau_v);
        let mut result = &gamma + (tau.apply( &(|x| 1.0/x) ) * delta_t).elemul(&( (&wij * ( &gamma - &theta ).apply(&Ctrnn::sigmoid)) - &gamma + (&wik * &i)));
        println!("{:?}", result); 
        return vec![0.0, 0.0];
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
            vec![-0.33246, -4.03171, -1.43741]);

        assert_eq!( ctrnn.activate(2, &gamma, delta_t, &tau, &wij, &theta, &wik, &i),
            vec![-0.86998, -6.11057, -2.14179]);
        assert_eq!( ctrnn.activate(10, &gamma, delta_t, &tau, &wij, &theta, &wik, &i),
            vec![-1.7698, -4.9550, -3.1953]);
        assert_eq!( ctrnn.activate(30, &gamma, delta_t, &tau, &wij, &theta, &wik, &i),
            vec![-1.7869, -4.9408, -3.2092]);

        //converges
        assert_eq!( ctrnn.activate(30, &gamma, delta_t, &tau, &wij, &theta, &wik, &i),
            ctrnn.activate(100, &gamma, delta_t, &tau, &wij, &theta, &wik, &i));
    }
}
