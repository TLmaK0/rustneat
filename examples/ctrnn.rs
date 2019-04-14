// run with: cargo run --release --example ctrnn --features="telemetry ctrnn_telemetry"
extern crate rustneat;

use rustneat::{Ctrnn, CtrnnNeuralNetwork};

#[cfg(feature = "telemetry")]
mod telemetry_helper;

#[cfg(feature = "telemetry")]
extern crate rusty_dashed;

#[cfg(feature = "telemetry")]
use self::rusty_dashed::Dashboard;

fn main() {
    let mut dashboard = Dashboard::new();
    dashboard.add_graph("ctrnn1", "ctrnn", 0, 0, 4, 4);
    telemetry_helper::run_server(dashboard, "", true);
    std::thread::sleep(std::time::Duration::from_millis(4000));

    let neurons_len = 2;
    let tau = vec![1.0; neurons_len];
    let theta  = vec![0.0; neurons_len];
    let wij = vec![0.0, 1.0, 0.0, 0.0];
    Ctrnn::default().activate_nn(
        100,
        &CtrnnNeuralNetwork {
            y: &vec![0.0; neurons_len],
            delta_t: 1.0,
            tau: &tau,
            wij: &wij,
            theta: &theta,
            i: &vec![0.0; neurons_len]
        },
    );
    std::thread::sleep(std::time::Duration::from_millis(4000));
}
