extern crate collenchyma;
extern crate collenchyma_nn;

use self::collenchyma::framework::IFramework;
use self::collenchyma::backend::{Backend, BackendConfig};
use self::collenchyma::frameworks::Native;
use self::collenchyma::Error;
use self::collenchyma::tensor::SharedTensor;
use self::collenchyma::DeviceType;
use self::collenchyma::MemoryType;
use self::collenchyma_nn::*;
use self::collenchyma::tensor::IntoTensorDesc;

pub struct Ctrnn {
    pub backend: Backend<Native>, //shuld replace with general backend
    pub cpu: DeviceType
}

impl Ctrnn {
    pub fn new() -> Ctrnn{
        let framework = Native::new();
        let hardwares = &framework.hardwares().to_vec();
        let backend_config = BackendConfig::new(framework, hardwares);
        let backend = Backend::new(backend_config);
        let native = Native::new();
        Ctrnn {
            backend: backend.unwrap(),
            cpu: native.new_device(native.hardwares()).unwrap() 
        }
    }

    pub fn activate(&self, steps: usize, gamma_v: &Vec<f64>, delta_t: f64, tau_v: &Vec<f64>, wij_v: &Vec<Vec<f64>>, theta_v: &Vec<f64>, wik_v: &Vec<Vec<f64>>, i_v: &Vec<f64>) -> Vec<f64> {
        let mut gamma = self.vector_to_tensor(gamma_v);
        let mut tau = self.vector_to_tensor(tau_v);
        let mut wij = self.matrix_to_tensor(wij_v);
        let mut theta = self.vector_to_tensor(theta_v);
        let mut wik = self.matrix_to_tensor(wik_v);
        let mut i = self.vector_to_tensor(i_v);
        let mut result = SharedTensor::<f64>::new(self.backend.device(), wij.desc()).unwrap();
        &self.backend.sigmoid(&mut wij, &mut result).unwrap();
        //result.add_device(&cpu).unwrap();
        result.sync(&self.cpu).unwrap();
        return vec![0.0, 0.0];
    }

    fn matrix_to_tensor(&self, matrix: &Vec<Vec<f64>>) -> SharedTensor<f64> {
        let mut tensor = SharedTensor::<f64>::new(self.backend.device(), &(matrix.len(), matrix[0].len())).unwrap();
        //tensor.add_device(&self.cpu).unwrap();
        tensor.sync(&self.cpu).unwrap();
        self.write_to_memory(tensor.get_mut(&self.cpu).unwrap(), &matrix.iter().flat_map(|s| s.iter()).collect::<Vec<&f64>>());
        tensor
    }

    fn vector_to_tensor(&self, vector: &Vec<f64>) -> SharedTensor<f64> {
        self.generic_to_tensor(&vec!(vector.len()), vector)
    }

    fn generic_to_tensor(&self, desc: &Vec<usize>, data: &[f64]) -> SharedTensor<f64> {
        let mut tensor = SharedTensor::<f64>::new(self.backend.device(), desc).unwrap();
        //tensor.add_device(&self.cpu).unwrap();
        tensor.sync(&self.cpu).unwrap();
        self.write_to_memory(tensor.get_mut(&self.cpu).unwrap(), data);
        tensor
    }

    fn write_to_memory<T: Copy>(&self, mem: &mut MemoryType, data: &[T]) {
        let &mut MemoryType::Native(ref mut mem) = mem;
        let mut mem_buffer = mem.as_mut_slice::<T>();
        for (index, datum) in data.iter().enumerate() {
            mem_buffer[index] = *datum;
        }
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
