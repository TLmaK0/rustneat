#[cfg(feature = "telemetry")]
extern crate open;

#[cfg(feature = "telemetry")]
extern crate rusty_dashed;

#[cfg(feature = "telemetry")]
use self::rusty_dashed::Dashboard;

#[allow(dead_code)]
pub fn main() {}

#[cfg(feature = "telemetry")]
#[allow(dead_code)]
pub fn enable_telemetry(query_string: &str, open: bool) {
    let mut dashboard = Dashboard::new();
    dashboard.add_graph("fitness1", "fitness", 0, 0, 4, 4);
    dashboard.add_graph("network1", "network", 4, 0, 4, 4);
    dashboard.add_graph("species1", "species", 0, 4, 4, 4);
    dashboard.add_graph("approximation1", "approximation", 4, 4, 2, 2);
    run_server(dashboard, query_string, open);
}

#[cfg(feature = "telemetry")]
pub fn run_server(dashboard: Dashboard, query_string: &str, open: bool) {
    rusty_dashed::Server::serve_dashboard(dashboard);
    open_url(query_string, open);
}

#[cfg(feature = "telemetry")]
pub fn open_url(query_string: &str, open: bool) {
    if !open {
        return;
    }

    let url = format!("http://localhost:3000{}", query_string);

    match open::that(url.clone()) {
        Err(_) => println!(
            "\nOpen browser and go to {:?} to see how neural network evolves\n",
            url
        ),
        _ => println!("Openning browser..."),
    }
}
