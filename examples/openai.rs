extern crate rustneat;

extern crate cpython;

extern crate python3_sys as ffi;

use cpython::{Python, PyDict, PyResult, ObjectProtocol, NoArgs};
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
    let reward = 0;
    let done = false;
    let ob = env.call_method(py, "reset", NoArgs, None).unwrap();
    let mut t = 0;
    while {
        t += 1;
        env.call_method(py, "render", NoArgs, None).unwrap();
        let action_space = env.getattr(py, "action_space").unwrap();
        let sample = action_space.call_method(py, "sample", NoArgs, None).unwrap();
        let observation = env.call_method(py, "step", (sample,), None).unwrap();
println!("{:?}", observation);
        t < 10
    }{}

    //let sys = py.import("sys")?;

    //let version: String = sys.get(py, "version")?.extract(py)?;

    //let locals = PyDict::new(py);
    //locals.set_item(py, "os", py.import("os")?)?;
    //let user: String = py.eval("os.getenv('USER') or os.getenv('USERNAME')", None, Some(&locals))?.extract(py)?;

    //println!("Hello {}, I'm Python {}", user, version);
    Ok(())
}

