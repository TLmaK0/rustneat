extern crate rustneat;
extern crate cpython;
extern crate python3_sys as ffi;
extern crate ctrlc;

use cpython::{NoArgs, ObjectProtocol, PyModule, PyObject, Python};
use ffi::PySys_SetArgv;
use rustneat::{Environment, Organism, Population, Ctrnn};
use std::ffi::CString;
use std::{cmp, process};
use std::cmp::Ordering;

#[cfg(feature = "telemetry")]
mod telemetry_helper;

struct LunarLander {
    gym: PyModule,
}

impl Environment for LunarLander {
    fn test(&self, organism: &mut Organism) -> f64 {
        self.lunar_lander_test(organism, false)
    }
}

impl LunarLander {
    fn new() -> LunarLander {
        let gil = Python::acquire_gil();
        let argv = CString::new("").unwrap().as_ptr();

        unsafe {
            PySys_SetArgv(0, argv as *mut *mut i32);
        }
        let py = gil.python();
        let gym = py.import("gym").unwrap();

        gym.get(py, "logger")
            .unwrap()
            .call_method(py, "set_level", (40,), None)
            .unwrap();

        LunarLander { gym: gym }
    }

    pub fn lunar_lander_test(&self, organism: &mut Organism, render: bool) -> f64 {
        let gil = Python::acquire_gil();
        let py = gil.python();

        let env = self.gym.call(py, "make", ("LunarLander-v2",), None).unwrap();

        env.call_method(py, "reset", NoArgs, None).unwrap();
        let mut total_reward = 0f64;

        let mut output = vec![0f64, 0f64, 0f64, 0f64];
        while {
            if render {
                env.call_method(py, "render", NoArgs, None).unwrap();
            }
            let out_exp = output.clone().into_iter().map(|v| v.exp());
            let sum_exp: f64 = out_exp.clone().sum();
            let softmax = out_exp.map(|e| e / sum_exp); 
            let value = softmax
                            .enumerate()
                            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                            .map(|(index, _)| index).unwrap();

            let (observation, reward, done) = extract_step_result(
                py,
                env.call_method(py, "step", (value,), None)
                    .unwrap(),
            );
            total_reward += reward;

            organism.activate(observation, &mut output);
            !done
        } {}
        env.call_method(py, "close", NoArgs, None).unwrap();
        total_reward
    }
}

fn main() {
    #[allow(unused_must_use)] {
        ctrlc::set_handler(move  || {
            println!("Exiting...");
            process::exit(130);
        });
    }

    #[cfg(feature = "telemetry")]
    telemetry_helper::enable_telemetry("?max_fitness=300", true);

    let mut population = Population::create_population_initialized(150, 8, 4);
    let mut environment = LunarLander::new();
    let mut champion: Option<Organism> = None;
    let mut generations = 0;
    while champion.is_none() {
        population.evolve();
        population.evaluate_in(&mut environment);
        let tmp_champion = population.champion.clone().unwrap();
        if generations == 30 {
            environment.lunar_lander_test(&mut tmp_champion.clone(), true);
            generations = 0;
        }

        if tmp_champion.fitness >= 200f64 {
            //check again. At least 2 successfully landing
            if environment.lunar_lander_test(&mut tmp_champion.clone(), true) >= 200f64 {
                champion = Some(tmp_champion);
            }
        }
        generations += 1;
    }
    environment.lunar_lander_test(&mut champion.unwrap(), true);
}

fn extract_step_result(py: Python, object: PyObject) -> (Vec<f64>, f64, bool) {
    (
        extract_observation(py, object.get_item(py, 0).unwrap()),
        object.get_item(py, 1).unwrap().extract::<f64>(py).unwrap(),
        object.get_item(py, 2).unwrap().extract::<bool>(py).unwrap(),
    )
}

fn extract_observation(py: Python, object: PyObject) -> Vec<f64> {
    let mut vec: Vec<f64> = Vec::new();
    // TODO: should be a better way to do this
    for n in 0..8 {
        vec.push(object.get_item(py, n).unwrap().extract::<f64>(py).unwrap());
    }
    vec
}
