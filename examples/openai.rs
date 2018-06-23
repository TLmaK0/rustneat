extern crate rustneat;

extern crate cpython;

extern crate python3_sys as ffi;

use cpython::{Python, ObjectProtocol, NoArgs, PyObject, PyModule};
use ffi::PySys_SetArgv;
use std::ffi::CString;
use rustneat::{Environment, Organism, Population};

struct CartPole {
    gym: PyModule
}

impl Environment for CartPole {
    fn test(&self, organism: &mut Organism) -> f64 {
        let mut total = 0f64;
        for n in 1..100 {
            total += self.cart_pole_test(organism);
        }

println!("{:?}", total / 100f64);
        total / 100f64
    }

}


impl CartPole {
    fn new() -> CartPole {
        let gil = Python::acquire_gil();
        let argv = CString::new("").unwrap().as_ptr();

        unsafe {
          PySys_SetArgv(0, argv as *mut *mut i32);
        }
        let py = gil.python();

        CartPole{ gym: py.import("gym").unwrap()}
    }

    fn cart_pole_test(&self, organism: &mut Organism) -> f64 {
        let gil = Python::acquire_gil();
        let py = gil.python();

        let env = self.gym.call(py, "make", ("CartPole-v0",), None).unwrap();

        env.call_method(py, "reset", NoArgs, None).unwrap();
        let mut total_reward = 0f64;
        let mut next_movement = 0;

        let mut output = vec![0f64, 0f64];
        while {
            //env.call_method(py, "render", NoArgs, None).unwrap();
            let action_space = env.getattr(py, "action_space").unwrap();
            let sample = action_space.call_method(py, "sample", NoArgs, None).unwrap();

            let (observation, reward, done) =  extract_step_result(py, 
                                                                 env.call_method(py, "step", (sample,), None).unwrap());
            total_reward += reward;

            organism.activate(&observation, &mut output);
            !done
        }{}
        env.call_method(py, "close", NoArgs, None).unwrap();
        total_reward
    }
}

fn main() {

    let mut population = Population::create_population(150);
    let mut environment = CartPole::new();
    let mut champion: Option<Organism> = None;
    while champion.is_none() {
        population.evolve();
        population.evaluate_in(&mut environment);
        for organism in &population.get_organisms() {
            if organism.fitness > 195f64 {
                champion = Some(organism.clone());
            }
        }
    }
    println!("{:?}", champion.unwrap().genome);
}

fn extract_step_result(py: Python, object: PyObject) -> (Vec<f64>, f64, bool) {
    (
        extract_observation(py, object.get_item(py, 0).unwrap()),
        object.get_item(py, 1).unwrap().extract::<f64>(py).unwrap(),
        object.get_item(py, 2).unwrap().extract::<bool>(py).unwrap()
    )
}

fn extract_observation(py: Python, object: PyObject) -> Vec<f64> {
    let mut vec: Vec<f64> = Vec::new();
    //TODO: should be a better way to do this
    for n in 0..4 {
        vec.push(object.get_item(py, n).unwrap().extract::<f64>(py).unwrap());
    }
    vec
}
