extern crate rand;
extern crate rustneat;

#[cfg(feature = "telemetry")]
#[macro_use]
extern crate rusty_dashed;

#[cfg(feature = "telemetry")]
mod telemetry_helper;

use rustneat::{Environment, Organism, Population, NeuralNetwork};

static mut BEST_FITNESS: f64 = 0.0;
struct FunctionApproximation;

impl Environment for FunctionApproximation {
  fn test(&self, organism: &mut NeuralNetwork) -> f64 {
      let mut output = vec![0f64];
      let mut distance = 0f64;

      let mut outputs = Vec::new();

      for x in -10..11 {
          organism.activate(vec![x as f64 / 10f64], &mut output);
          distance += ((x as f64).powf(2f64) - (output[0] * 100f64)).abs();
          outputs.push([x, (output[0] * 100f64) as i64]);
      }

      let fitness = 100f64 / (1f64 + (distance / 1000.0));
      unsafe {
            if fitness > BEST_FITNESS {
                BEST_FITNESS = fitness;
      #[cfg(feature = "telemetry")]
      telemetry!("approximation1", 1.0, format!("{:?}", outputs));
            }
      }
      fitness
  }
}

fn main() {
    let mut population = Population::create_population(150);
    let mut environment = FunctionApproximation;
    let mut champion: Option<Organism> = None;

    #[cfg(feature = "telemetry")]
    telemetry_helper::enable_telemetry("?max_fitness=100&ioNeurons=1,2", true);

    #[cfg(feature = "telemetry")]
    std::thread::sleep(std::time::Duration::from_millis(2000));

    #[cfg(feature = "telemetry")]
    telemetry!("approximation1", 1.0, format!("{:?}", (-10..11).map(|x| [x, x * x]).collect::<Vec<_>>()));

    #[cfg(feature = "telemetry")]
    std::thread::sleep(std::time::Duration::from_millis(2000));

    while champion.is_none() {
        population.evolve();
        population.evaluate_in(&mut environment);
        for organism in &population.get_organisms() {
            if organism.fitness >= 96f64 {
                champion = Some(organism.clone());
            }
        }
    }
    println!("{:?}", champion.unwrap().genome);
}
