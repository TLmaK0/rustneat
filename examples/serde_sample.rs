extern crate rand;
extern crate rustneat;

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::sync::{Arc, Mutex};

use rustneat::Environment;
use rustneat::Organism;
use rustneat::Population;

#[cfg(feature = "telemetry")]
mod telemetry_helper;

struct XORClassification;

impl Environment for XORClassification {
    fn test(&self, organism: &mut Organism) -> f64 {
        let mut output = vec![0f64];
        let mut distance: f64;
        organism.activate(vec![0f64, 0f64], &mut output);
        distance = (0f64 - output[0]).powi(2);
        organism.activate(vec![0f64, 1f64], &mut output);
        distance += (1f64 - output[0]).powi(2);
        organism.activate(vec![1f64, 0f64], &mut output);
        distance += (1f64 - output[0]).powi(2);
        organism.activate(vec![1f64, 1f64], &mut output);
        distance += (0f64 - output[0]).powi(2);

        let fitness = 16f64 / (1f64 + distance);

        fitness
    }
}

const POPULATION_PATH: &str = "population.json";

fn main() {
    let lock = Arc::new(Mutex::new(()));
    let l = Arc::clone(&lock);
    ctrlc::set_handler(move || {
        let _guard = l.lock().unwrap();
        std::process::exit(0);
    })
    .unwrap();

    #[cfg(feature = "telemetry")]
    telemetry_helper::enable_telemetry("?max_fitness=18", true);

    #[cfg(feature = "telemetry")]
    std::thread::sleep(std::time::Duration::from_millis(2000));

    let mut population = match File::open(POPULATION_PATH) {
        Ok(file) => match serde_json::from_reader(BufReader::new(file)) {
            Ok(population) => {
                println!("Loaded population from {}", POPULATION_PATH);
                population
            }
            Err(err) => {
                eprintln!("Error parsing {}: {}", POPULATION_PATH, err);
                return;
            }
        },
        Err(err) => {
            if err.kind() == std::io::ErrorKind::NotFound {
                Population::create_population(150)
            } else {
                eprintln!("Error reading {}: {}", POPULATION_PATH, err);
                return;
            }
        }
    };

    let mut environment = XORClassification;
    let mut champion: Option<Organism> = None;
    while champion.is_none() {
        population.evolve();
        population.evaluate_in(&mut environment);
        {
            let _guard = lock.lock().unwrap();
            match File::create(POPULATION_PATH) {
                Ok(file) => {
                    if let Err(err) = serde_json::to_writer(BufWriter::new(file), &population) {
                        eprintln!("Error serializing to {}: {}", POPULATION_PATH, err);
                        return;
                    }
                }
                Err(err) => {
                    eprintln!("Error writing to {}: {}", POPULATION_PATH, err);
                    return;
                }
            }
        }
        for organism in &population.get_organisms() {
            if organism.fitness > 15.5f64 {
                champion = Some(organism.clone());
            }
        }
    }
    println!("{}", serde_json::to_string_pretty(&champion.unwrap()).unwrap());
}
