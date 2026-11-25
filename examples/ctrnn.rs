// run with: cargo run --release --example ctrnn
// (graphs with: --features="telemetry ctrnn_telemetry")
// Explicit Euler (`Integrator::Euler`) against exponential Euler
// (`Integrator::ExponentialEuler`, the default): explicit Euler diverges for
// h/τ > 2, exponential Euler is stable for any step size. Step-size demo
// after http://www.tinyblueplanet.com/easy/FCSReport.pdf
extern crate rustneat;

mod ctrnn_common;

#[cfg(feature = "telemetry")]
use rustneat::{Ctrnn, CtrnnNeuralNetwork, Integrator};

#[cfg(feature = "telemetry")]
mod telemetry_helper;

#[cfg(feature = "telemetry")]
#[macro_use]
extern crate rusty_dashed;

#[cfg(feature = "telemetry")]
use self::rusty_dashed::Dashboard;

fn main() {
    // Deterministic text demo (see examples/ctrnn_common).
    print!("{}", ctrnn_common::report());

    // With the dashboard (one window each): the step-size runs with explicit
    // Euler, the same with exponential Euler, and a single neuron relaxing
    // from ±1.
    #[cfg(feature = "telemetry")]
    {
        show_graph();
        step_size_test(Integrator::Euler);
        close();
        open();
        step_size_test(Integrator::ExponentialEuler);
        close();
        open();
        minimal_ctrnn_node();
        close();
    }
}

#[cfg(feature = "telemetry")]
fn close() {
    std::thread::sleep(std::time::Duration::from_millis(2000));
    telemetry!("ctrnn1", 1.0, "window.close()");
}

#[cfg(feature = "telemetry")]
fn minimal_ctrnn_node() {
    std::thread::sleep(std::time::Duration::from_millis(2000));
    telemetry!("ctrnn1", 1.0, "clear('ctrnn1', 500, 1)");
    activate_minimal_ctrnn_node(1.0);
    telemetry!("ctrnn1", 1.0, "next_graph()");
    activate_minimal_ctrnn_node(-1.0);
}

#[cfg(feature = "telemetry")]
fn activate_minimal_ctrnn_node(initial: f64) {
    Ctrnn::default().activate_nn(
        5.0,
        0.01,
        &CtrnnNeuralNetwork {
            y: &[initial],
            tau: &[1.0],
            wji: &[0.0],
            theta: &[0.0],
            i: &[0f64],
        },
    );
}

/// One graph per step size (h/τ up to 2.1: larger ones leave the plot).
#[cfg(feature = "telemetry")]
fn step_size_test(integrator: Integrator) {
    std::thread::sleep(std::time::Duration::from_millis(1000));
    telemetry!("ctrnn1", 1.0, "clear('ctrnn1', 16, 2.5)");
    let step_sizes = [0.1, 0.5, 1.0, 1.5, 2.0, 2.1];
    for (n, &step_size) in step_sizes.iter().enumerate() {
        if n > 0 {
            telemetry!("ctrnn1", 1.0, "next_graph('ctrnn1')");
        }
        activate_step_size_test(integrator, step_size);
    }
}

#[cfg(feature = "telemetry")]
fn activate_step_size_test(integrator: Integrator, step_size: f64) {
    Ctrnn::with_integrator(integrator).activate_nn(
        ctrnn_common::STEPS as f64 * step_size,
        step_size,
        &CtrnnNeuralNetwork {
            y: &[ctrnn_common::STEP_SIZE_Y0],
            tau: &[1.0],
            wji: &[0.0],
            theta: &[0.0],
            i: &[ctrnn_common::STEP_SIZE_FIXED_POINT],
        },
    );
}

#[cfg(feature = "telemetry")]
fn show_graph() {
    let mut dashboard = Dashboard::new();
    dashboard.add_graph("ctrnn1", "ctrnn", 0, 0, 4, 4);
    telemetry_helper::run_server(dashboard, "", false);
    open();
}

#[cfg(feature = "telemetry")]
fn open() {
    telemetry_helper::open_url("", true);
}
