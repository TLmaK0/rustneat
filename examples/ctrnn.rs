// run with: cargo run --release --example ctrnn --features="telemetry ctrnn_telemetry"
// try to reproduce results from: http://www.tinyblueplanet.com/easy/FCSReport.pdf
extern crate rustneat;

use rustneat::{Ctrnn, CtrnnNeuralNetwork};

#[cfg(feature = "telemetry")]
mod telemetry_helper;

#[cfg(feature = "telemetry")]
extern crate rusty_dashed;

#[cfg(feature = "telemetry")]
use self::rusty_dashed::Dashboard;

fn main() {
    #[cfg(feature = "telemetry")]
    show_graph();

    std::thread::sleep(std::time::Duration::from_millis(4000));

    minimal_ctrnn_node();

    std::thread::sleep(std::time::Duration::from_millis(4000));
}

fn minimal_ctrnn_node(){
    Ctrnn::default().activate_nn(
        100,
        &CtrnnNeuralNetwork {
            y: &vec![1.0],
            delta_t: 0.01,
            tau: &vec![1.0],
            wji: &vec![
                0.0
            ],
            theta: &vec![0.0],
            i: &vec![0.0]
        },
    );
}

fn neurons_1_input_0(){
    Ctrnn::default().activate_nn(
        1,
        &CtrnnNeuralNetwork {
            y: &vec![0.0],
            delta_t: 1.0,
            tau: &vec![1.0],
            wji: &vec![
                //      i=0
                        0.0  // j=0
            ],
            theta: &vec![0.0],
            i: &vec![0.0]
        },
    );

    // Should return:
    // Matrix { rows: 1, cols: 1, data: [0.0] }
}

#[cfg(feature = "telemetry")]
fn show_graph(){
    let mut dashboard = Dashboard::new();
    dashboard.add_graph("ctrnn1", "ctrnn", 0, 0, 4, 4);
    telemetry_helper::run_server(dashboard, "", true);
}
