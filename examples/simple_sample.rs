extern crate rustneat;
#[macro_use]
extern crate rusty_dashed;

extern crate rand;

use rustneat::Environment;
use rustneat::Organism;
use rustneat::Population;
use rusty_dashed::Dashboard;


struct XORClassification;

impl Environment for XORClassification {
    fn test(&self, organism: &mut Organism) -> f64 {
        let mut output = vec![0f64];
        let mut distance: f64;
        organism.activate(&vec![0f64, 0f64], &mut output);
        let _tel1 = output[0];
        distance = (0f64 - output[0]).abs();
        organism.activate(&vec![0f64, 1f64], &mut output);
        let _tel2 = output[0];
        distance += (1f64 - output[0]).abs();
        organism.activate(&vec![1f64, 0f64], &mut output);
        let _tel3 = output[0];
        distance += (1f64 - output[0]).abs();
        organism.activate(&vec![1f64, 1f64], &mut output);
        let _tel4 = output[0];
        distance += (0f64 - output[0]).abs();

        let fitness = (4f64 - distance).powi(2);

        telemetry!("distance1",
                   0.01,
                   format!(r#"[
                               {{id:1,v:{}}},
                               {{id:2,v:{}}},
                               {{id:3,v:{}}},
                               {{id:4,v:{}}},
                               {{id:'space',v:0}},
                               {{id:'fitness',v:{}}}
                              ]"#,
                           _tel1,
                           _tel2,
                           _tel3,
                           _tel4,
                           fitness / 16.0));
        fitness
    }
}

fn main() {
    let mut dashboard = Dashboard::new();
    dashboard.add_graph("distance1", "distance", 0, 0, 4, 4);

    rusty_dashed::Server::serve_dashboard(dashboard);


    #[cfg(feature = "telemetry")]
    println!("\nGo to http://localhost:3000 to see how neural network evolves\n");

    let mut population = Population::create_population(150);
    let mut environment = XORClassification;
    let mut champion: Option<Organism> = None;
    while champion.is_none() {
        population.evolve();
        population.evaluate_in(&mut environment);
        for organism in &population.get_organisms() {
            if organism.fitness > 15.9f64 {
                champion = Some(organism.clone());
            }
        }
    }
    println!("{:?}", champion.unwrap().genome);
}
