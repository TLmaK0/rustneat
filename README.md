### Rust NEAT
[![CI](https://github.com/TLmaK0/rustneat/actions/workflows/ci.yml/badge.svg)](https://github.com/TLmaK0/rustneat/actions/workflows/ci.yml)
[![Gitter](https://img.shields.io/gitter/room/nwjs/nw.js.svg)](https://gitter.im/rustneat/rustneat)

Implementation of **NeuroEvolution of Augmenting Topologies NEAT** http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf

This implementation uses a **CTRNN** (Continuous-Time Recurrent Neural Network) based on **On the Dynamics of Small Continuous-Time Recurrent Neural Network** (Beer, 1995) http://www.cs.uvm.edu/~jbongard/2014_CS206/Beer_CTRNNs.pdf

### CTRNN time constant (τ)

The time constant τ controls neuron response speed — like biological membrane resistance time. What matters is the ratio `dt/τ` where `dt=0.01` is the simulation step. Configurable via `MutationConfig::tau`:

- **Small τ** (e.g. 0.01): `dt/τ = 1.0` — neurons react instantly, state resets each step. Network behaves as **feedforward**. Use for stateless problems like XOR.
- **Large τ** (e.g. 0.1): `dt/τ = 0.1` — neurons update only 10% per step, retaining 90% of previous state. Network has **temporal memory**. Use for control tasks like Lunar Lander where the agent integrates information over time.
- **Very large τ** (e.g. 1.0): `dt/τ = 0.01` — neurons barely respond, very strong inertia. Needs many steps to react to new inputs.

## Telemetry Dashboard

![telemetry](docs/img/rustneat.png)

```bash
cargo run --release --example simple_sample --features=telemetry
```

Then go to `http://localhost:3000` to see how the neural network evolves.

![telemetry](docs/results/cart_pole_dashboard.gif)

## Cart Pole

![cart pole](docs/results/cart_pole.gif)

## Lunar Lander

NEAT+CTRNN agent evolved to land on the OpenAI Gym LunarLander-v3 environment using discrete actions with independent threshold-based output control.

![lunar lander](docs/results/lunar_lander.gif)

### Results
- **Verified average reward**: +70 (fitness 570 with +500 offset)
- **Peak reward**: +278 (fitness 778)
- **Solved threshold**: average reward +200

The agent uses 2 independent CTRNN outputs (main thruster, lateral direction) with lateral priority — a key insight for CTRNN control problems where argmax causes state lock-in.

### Run

```bash
cargo build --release --example openai_lunar_lander --features openai
./target/release/examples/openai_lunar_lander
```

### Test champion

```bash
cargo build --release --example test_champion --features openai
./target/release/examples/test_champion
```

## Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

## Run tests

To speed up tests, run them with `--release`. The XOR test runs one seeded attempt of at most 500 generations (seed 1; set `RUSTNEAT_XOR_SEED` to try another) and checks that the champion solves XOR in all 24 orders of the patterns:

```bash
cargo test --release
```

## Sample usage

Create a new cargo project and add rustneat to Cargo.toml:

```toml
[dependencies]
rustneat = "0.2.1"
```

Then use the library to implement XOR classification. `activate` keeps the
CTRNN state between calls (useful for control tasks, where one episode is many
calls); XOR is four independent cases, so the state is reset before each one:

```rust
extern crate rustneat;
use rustneat::Environment;
use rustneat::Organism;
use rustneat::Population;

const PATTERNS: [([f64; 2], f64); 4] = [
    ([0f64, 0f64], 0f64),
    ([0f64, 1f64], 1f64),
    ([1f64, 0f64], 1f64),
    ([1f64, 1f64], 0f64),
];

struct XORClassification;

impl Environment for XORClassification {
    fn test(&self, organism: &mut Organism) -> f64 {
        let mut output = vec![0f64];
        let mut distance = 0f64;
        for (inputs, expected) in PATTERNS.iter() {
            organism.reset_state(); // each pattern starts from rest
            organism.activate(inputs.to_vec(), &mut output);
            distance += (expected - output[0]).powi(2);
        }
        16f64 / (1f64 + distance)
    }
}

fn main() {
    // 2 inputs connected directly to 1 output, no hidden neurons:
    // NEAT adds neurons and connections as it needs them.
    let mut population = Population::create_population_initialized(150, 2, 1);
    let environment = XORClassification;
    let mut champion: Option<Organism> = None;
    while champion.is_none() {
        population.evolve();
        // Err if the environment returns NaN or an infinite fitness.
        population
            .evaluate_in(&environment)
            .expect("the environment must return a finite fitness");
        for organism in &population.get_organisms() {
            if organism.fitness > 15.5f64 {
                champion = Some(organism.clone());
            }
        }
    }
    println!("{:?}", champion.unwrap().genome);
}
```

For a reproducible run, seed the population:
`Population::create_population_initialized(150, 2, 1).with_seed(42)`.

## Develop

Check style guidelines with:

```bash
rustup component add rustfmt-preview
cargo fmt
```

## References

- **NeuroEvolution of Augmenting Topologies NEAT** http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf
- **On the Dynamics of Small Continuous-Time Recurrent Neural Network** (Beer, 1995) http://www.cs.uvm.edu/~jbongard/2014_CS206/Beer_CTRNNs.pdf
- **An Investigation into the Dynamics of a Continuous Time Recurrent Neural Network Node** http://www.tinyblueplanet.com/easy/FCSReport.pdf

## Thanks

Thanks for the icon nerves by Delwar Hossain from the Noun Project
