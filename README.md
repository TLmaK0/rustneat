### Rust NEAT
[![travis-ci](https://img.shields.io/travis/TLmaK0/rustneat/master.svg)](https://travis-ci.org/TLmaK0/rustneat)
[![Gitter](https://img.shields.io/gitter/room/nwjs/nw.js.svg)](https://gitter.im/rustneat/rustneat)

Implementation of **NeuroEvolution of Augmenting Topologies NEAT** http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf

This implementations uses a **Continuous-Time Recurrent Neural Network** (**CTRNN**) (Yamauchi and Beer, 1994).

![telemetry](docs/img/rustneat.png)

## Run test

To speed up tests, run them with `--release` (XOR classification/simple_sample should take less than a minute)

`cargo test --release`
## Run example

`cargo run --release --example simple_sample --features=telemetry`

then go to `http://localhost:3000` to see how neural network evolves

![telemetry](docs/results/cart_pole_dashboard.gif)

## Run openai tests

![telemetry](docs/results/cart_pole.gif)

Install python dependencies

```bash
sudo apt install python3
sudo apt install python3-pip
sudo apt install libpython3.5-dev
sudo pip3 install gym
sudo apt install nvidia-384
sudo apt install python3-opengl
```

Run test

```
cargo run --release --example openai --features=openai,telemetry
```
    
### Windows Openai

https://github.com/openai/gym/issues/11#issuecomment-242950165
```
Update to the latest version of Windows (>version 1607, "Anniversary Update")
Enable Windows Subsystem for Linux (WSL)
Open cmd, run bash
Install python & gym (using sudo, and NOT PIP to install gym). So by now you should probably be able to run things and get really nasty graphics related errors. This is because WSL doesn't support any displays, so we need to fake it.
Install vcXsrv, and run it (you should just have a little tray icon)
In bash run "export DISPLAY=:0" Now when you run it you should get a display to pop-up, there may be issues related to graphics drivers. Sadly, this is where the instructions diverge if you don't have an NVIDIA graphics card.
Get the drivers: "sudo apt-get install nvidia-319 nvidia-settings-319 nvidia-prime"
Run!
```

## Sample usage

Create a new cargo project:

Add rustneat to Cargo.toml
```
[dependencies]
rustneat = "0.2.1"
```

Then use the library i.e. to implement the above example, use the library as follows:

```rust
extern crate rustneat;
use rustneat::Environment;
use rustneat::Organism;
use rustneat::Population;

struct XORClassification;

impl Environment for XORClassification {
    fn test(&self, organism: &mut Organism) -> f64 {
        let mut output = vec![0f64];
        let mut distance: f64;
        organism.activate(&vec![0f64, 0f64], &mut output);
        distance = (0f64 - output[0]).abs();
        organism.activate(&vec![0f64, 1f64], &mut output);
        distance += (1f64 - output[0]).abs();
        organism.activate(&vec![1f64, 0f64], &mut output);
        distance += (1f64 - output[0]).abs();
        organism.activate(&vec![1f64, 1f64], &mut output);
        distance += (0f64 - output[0]).abs();
        (4f64 - distance).powi(2)
    }
}

fn main() {
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

```

# Develop
Check style guidelines with:

`rustup component add rustfmt-preview`
`cargo fmt -- --write-mode=diff`

# Thanks
Thanks for the icon nerves by Delwar Hossain from the Noun Project
