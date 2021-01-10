extern crate rustneat;
extern crate cpython;
extern crate python3_sys as ffi;
extern crate ctrlc;

use cpython::{NoArgs, ObjectProtocol, PyObject, Python};
use ffi::PySys_SetArgv;
use rustneat::{Environment, Organism, Population};
use std::ffi::CString;
use std::process;
use std::cmp::Ordering;
use std::{sync::Mutex, ops::{Deref, DerefMut}};

#[cfg(feature = "telemetry")]
mod telemetry_helper;

struct LunarLander {
    env: PyObject
}

struct LunarLanderMultiprocess {
    lunar_lander_pool: Mutex<Vec<LunarLander>>
}

impl LunarLanderMultiprocess {
    fn new() -> LunarLanderMultiprocess {
        let cpus = LunarLanderMultiprocess::threads();
        let lunar_lander_pool = (0..cpus).map(|_| LunarLander::new()).collect::<Vec::<LunarLander>>();
        LunarLanderMultiprocess { lunar_lander_pool: Mutex::new(lunar_lander_pool) }
    }

    pub fn lunar_lander_test(&self, organism: &mut Organism, render: bool) -> f64 {
        let lunar_lander = {
            self.lunar_lander_pool.lock().unwrap().deref_mut().pop().unwrap()
        };
        let result = lunar_lander.lunar_lander_test(organism, render);
        self.lunar_lander_pool.lock().unwrap().push(lunar_lander);
        return result;
    }

    fn close(&self) {
        for lunar_lander in self.lunar_lander_pool.lock().unwrap().deref() {
            lunar_lander.close();
        }
    }

    fn threads() -> usize {
        //TODO: use multiple processes. Gil still blocking threads.
        num_cpus::get()
    }
}

impl Environment for LunarLanderMultiprocess {
    fn test(&self, organism: &mut Organism) -> f64 {
        self.lunar_lander_test(organism, false)
    }

    fn threads(&self) -> usize {
        LunarLanderMultiprocess::threads()
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

        let env = gym.call(py, "make", ("LunarLander-v2",), None).unwrap();

        LunarLander { env: env }
    }

    pub fn close(&self) {
        let gil = Python::acquire_gil();
        let py = gil.python();
        self.env.call_method(py, "close", NoArgs, None).unwrap();
    }

    pub fn lunar_lander_test(&self, organism: &mut Organism, render: bool) -> f64 {
        let gil = Python::acquire_gil();
        let py = gil.python();

        let mut observation: Vec<f64> = extract_observation(py, self.env.call_method(py, "reset", NoArgs, None).unwrap());
        let mut total_reward = 0f64;

        let mut output = vec![0f64, 0f64, 0f64, 0f64];
        while {
            if render {
                self.env.call_method(py, "render", NoArgs, None).unwrap();
            }
            let out_exp = output.clone().into_iter().map(|v| v.exp());
            let sum_exp: f64 = out_exp.clone().sum();
            let softmax = out_exp.map(|e| e / sum_exp); 
            let value = softmax
                            .enumerate()
                            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                            .map(|(index, _)| index).unwrap();

            organism.activate(observation.clone(), &mut output);
            let (observation_tmp, reward, done) = extract_step_result(
                py,
                self.env.call_method(py, "step", (value,), None)
                    .unwrap(),
            );
            observation = observation_tmp;
            total_reward += reward;

            !done
        } {}
        total_reward
    }
}

fn main() {
    let max_fitness = 300f64;

    #[allow(unused_must_use)] {
        ctrlc::set_handler(move  || {
            println!("Exiting...");
            process::exit(130);
        });
    }

    #[cfg(feature = "telemetry")]
    telemetry_helper::enable_telemetry(format!("?max_fitness={}", max_fitness).as_str(), true);

    let mut population = Population::create_population_initialized(150, 8, 4);
    let mut environment = LunarLanderMultiprocess::new();
    let mut champion: Option<Organism> = None;
    let mut generations = 0;
    while champion.is_none() {
        population.evolve();
        population.evaluate_in(&mut environment);
        match population.champion {
            Some(_) => {
                let tmp_champion = population.champion.clone().unwrap();
                if generations == 30 {
                    environment.lunar_lander_test(&mut tmp_champion.clone(), true);
                    generations = 0;
                }

                if tmp_champion.fitness >= max_fitness {
                    //check again. At least 2 successfully landing
                    if environment.lunar_lander_test(&mut tmp_champion.clone(), true) >= max_fitness {
                        champion = Some(tmp_champion);
                    }
                }
            },
            None => {}
        }
        generations += 1;
    }

    let result = champion.unwrap();
    environment.lunar_lander_test(&mut result.clone(), true);
    println!("{:?}", result.genome);
    environment.close();
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
