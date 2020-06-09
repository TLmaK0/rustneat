// run with: cargo run --release --example ctrnn --features="telemetry ctrnn_telemetry"
// try to reproduce results from: http://www.tinyblueplanet.com/easy/FCSReport.pdf
extern crate rustneat;

use rustneat::{Ctrnn, CtrnnNeuralNetwork};

mod telemetry_helper;

#[macro_use]
extern crate rusty_dashed;

use self::rusty_dashed::Dashboard;

fn main() {
    show_graph();
    step_size_test();
    close();
    open();
    minimal_ctrnn_node();
    close();
}

fn close() {
    std::thread::sleep(std::time::Duration::from_millis(2000));
    telemetry!("ctrnn1", 1.0, "window.close()");
}

fn minimal_ctrnn_node() {
    std::thread::sleep(std::time::Duration::from_millis(2000));
    telemetry!("ctrnn1", 1.0, "clear('ctrnn1', 500, 1)");
    activate_minimal_ctrnn_node(1.0);
    telemetry!("ctrnn1", 1.0, "next_graph()");
    activate_minimal_ctrnn_node(-1.0);
}

fn activate_minimal_ctrnn_node(initial: f64){
    Ctrnn::default().activate_nn(
        5.0,
        0.01,
        &CtrnnNeuralNetwork {
            y: &vec![initial],
            tau: &vec![1.0],
            wji: &vec![
                0.0
            ],
            theta: &vec![0.0],
            i: &vec![0f64]
        },
    );
}

fn step_size_test(){
    std::thread::sleep(std::time::Duration::from_millis(1000));
    telemetry!("ctrnn1", 1.0, "clear('ctrnn1', 16, 2.5)");

    activate_step_size_test(0.1);
    telemetry!("ctrnn1", 1.0, "next_graph('ctrnn1')");
    activate_step_size_test(0.5);
    telemetry!("ctrnn1", 1.0, "next_graph('ctrnn1')");
    activate_step_size_test(1.0);
    telemetry!("ctrnn1", 1.0, "next_graph('ctrnn1')");
    activate_step_size_test(1.5);
    telemetry!("ctrnn1", 1.0, "next_graph('ctrnn1')");
    activate_step_size_test(2.0);
    telemetry!("ctrnn1", 1.0, "next_graph('ctrnn1')");
    activate_step_size_test(2.1);
}

fn activate_step_size_test(step_size: f64){
    Ctrnn::default().activate_nn(
        15.0 * step_size,
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

fn show_graph(){
    let mut dashboard = Dashboard::new();
    dashboard.add_graph("ctrnn1", "ctrnn", 0, 0, 4, 4);
    telemetry_helper::run_server(dashboard, "", false);
    open();
}

fn open(){
    telemetry_helper::open_url("", true);
}
