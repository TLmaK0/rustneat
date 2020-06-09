// run with: cargo run --release --example ctrnn --features="telemetry ctrnn_telemetry"
// try to reproduce results from: http://www.tinyblueplanet.com/easy/FCSReport.pdf
extern crate rustneat;

use rustneat::{Ctrnn, CtrnnNeuralNetwork};

#[cfg(feature = "telemetry")]
mod telemetry_helper;

#[cfg(feature = "telemetry")]
#[macro_use]
extern crate rusty_dashed;

#[cfg(feature = "telemetry")]
use self::rusty_dashed::Dashboard;

fn main() {
    #[cfg(feature = "telemetry")]
    show_graph();

    std::thread::sleep(std::time::Duration::from_millis(1000));

    minimal_ctrnn_node(0.1);
    telemetry!("ctrnn1", 1.0, "reset()");
    minimal_ctrnn_node(0.5);
    telemetry!("ctrnn1", 1.0, "reset()");
    minimal_ctrnn_node(1.0);
    telemetry!("ctrnn1", 1.0, "reset()");
    minimal_ctrnn_node(1.5);
    telemetry!("ctrnn1", 1.0, "reset()");
    minimal_ctrnn_node(2.0);
    telemetry!("ctrnn1", 1.0, "reset()");
    minimal_ctrnn_node(2.1);

    std::thread::sleep(std::time::Duration::from_millis(4000));
}

fn minimal_ctrnn_node(step_size: f64){
    Ctrnn::default().activate_nn(
        15,
        step_size,
        &CtrnnNeuralNetwork {
            y: &vec![1.0],
            tau: &vec![1.0],
            wji: &vec![
                0.0
            ],
            theta: &vec![0.0],
            i: &vec![0.5]
        },
    );
}

#[cfg(feature = "telemetry")]
fn show_graph(){
    let mut dashboard = Dashboard::new();
    dashboard.add_graph("ctrnn1", "ctrnn", 0, 0, 4, 4);
    telemetry_helper::run_server(dashboard, "", true);
}
