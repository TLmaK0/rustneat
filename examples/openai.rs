extern crate rustneat;

extern crate cpython;

extern crate python3_sys as ffi;

use cpython::{Python, PyResult, ObjectProtocol, NoArgs, PyObject};
use ffi::PySys_SetArgv;
use std::ffi::CString;

fn main() {
    let gil = Python::acquire_gil();
    hello(gil.python()).unwrap();
}

fn hello(py: Python) -> PyResult<()> {
    let argv = CString::new("").unwrap().as_ptr();
    unsafe {
      PySys_SetArgv(0, argv as *mut *mut i32);
    }
    let gym = py.import("gym")?;
    let env = gym.call(py, "make", ("CartPole-v0",), None).unwrap();
    env.call_method(py, "reset", NoArgs, None).unwrap();
    while {
        env.call_method(py, "render", NoArgs, None).unwrap();
        let action_space = env.getattr(py, "action_space").unwrap();
        let sample = action_space.call_method(py, "sample", NoArgs, None).unwrap();
        let (_observation, reward, done) =  extract_step_result(py, 
                                                             env.call_method(py, "step", (sample,), None).unwrap());
        println!("{:?}", reward);
        !done
    }{}

    Ok(())
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
