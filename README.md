### Rust NEAT

Implementation of NeuroEvolution of Augmenting Topologies NEAT http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf

This implementations uses a Continous-Time Recurrent Neural Network (CTRNN) (Yamauchi and Beer, 1994).

## Run test

To speed up tests, run they with ```--release``` (XOR classification should take less than a minute)

```
cargo test --release
```

## Sample usage

Create a new cargo project:

```
cargo init rustneatsample
cd rustneatsample
```

Add rustneat to Cargo.toml
```
[dependencies]
rustneat = "0.1.5"
```

Remove src/lib.rs and create a src/main.rs with:
```
extern crate rustneat;
use rustneat::neat::Environment as Environment;
use rustneat::neat::Organism as Organism;
use rustneat::neat::Population as Population;

struct XORClassification;

impl Environment for XORClassification{
    fn test(&self, organism: &mut Organism) -> f64 {
        let mut output = vec![0f64];
        let mut distance: f64;
        organism.activate(&vec![0f64,0f64], &mut output); 
        distance = (0f64 - output[0]).abs();
        organism.activate(&vec![0f64,1f64], &mut output); 
        distance += (1f64 - output[0]).abs();
        organism.activate(&vec![1f64,0f64], &mut output); 
        distance += (1f64 - output[0]).abs();
        organism.activate(&vec![1f64,1f64], &mut output); 
        distance += (0f64 - output[0]).abs();
        (4f64 - distance).powi(2)
    }
}

fn main(){
    let mut population = Population::create_population(150);
    let environment = XORClassification;
    let mut champion: Option<Organism> = None;
    while champion.is_none() {
        population.evolve();
        population.evaluate_in(&environment);
        for organism in &population.get_organisms() {
            if organism.fitness > 15.9f64 {
                champion = Some(organism.clone());
            }
        }
    }
    println!("{:?}", champion.unwrap().genome);
}
```

run the app and wait a minute for the result:
```
cargo run
```
