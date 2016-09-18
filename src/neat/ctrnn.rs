extern crate collenchyma;
extern crate collenchyma_nn;

use self::collenchyma::framework::IFramework;
use self::collenchyma::backend::{Backend, BackendConfig};
use self::collenchyma::frameworks::Native;
use self::collenchyma::hardware::IHardware;
use self::collenchyma::device::IDevice;
use self::collenchyma::binary::IBinary;
use self::collenchyma::memory::IMemory;
use self::collenchyma::Error;
use self::collenchyma::tensor::SharedTensor;
use self::collenchyma_nn::*;

pub struct Ctrnn {
    pub backend: Backend<Native>
}

impl Ctrnn {
    pub fn new() -> Ctrnn{
        let framework = Native::new();
        let hardwares = &framework.hardwares().to_vec();
        let backend_config = BackendConfig::new(framework, hardwares);
        let backend = Backend::new(backend_config);
        Ctrnn {
            backend: backend.unwrap()
        }
    }

    pub fn activate(&self, steps: usize, gamma: &Vec<f64>, delta_t: f64, tau: &Vec<f64>, wij: &Vec<Vec<f64>>, theta: &Vec<f64>, wik: &Vec<Vec<f64>>, i: &Vec<f64>) -> Vec<f64> {
        return vec![0.0, 0.0];
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
            vec![-0.86998, -6.11057, -2.14179]);
        assert_eq!( ctrnn.activate(10, &gamma, delta_t, &tau, &wij, &theta, &wik, &i),
            vec![-1.7698, -4.9550, -3.1953]);
        assert_eq!( ctrnn.activate(30, &gamma, delta_t, &tau, &wij, &theta, &wik, &i),
            vec![-1.7869, -4.9408, -3.2092]);

        assert_eq!( ctrnn.activate(30, &gamma, delta_t, &tau, &wij, &theta, &wik, &i),
            ctrnn.activate(100, &gamma, delta_t, &tau, &wij, &theta, &wik, &i));
    }
}
