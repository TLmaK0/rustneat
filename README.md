### Rust NEAT

Implementation of NeuroEvolution of Augmenting Topologies NEAT http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf

This implementations uses a Continuous-Time Recurrent Neural Network (CTRNN) (Yamauchi and Beer, 1994).

## Run test

To speed up tests, run they with `--release` (XOR classification/simple_sample should take less than a minute)

```
`cargo test --release`
```
## Run example

`cargo run --release --example simple_sample`

## Sample usage

Create a new cargo project:

Add rustneat to Cargo.toml
```
[dependencies]
rustneat = "0.1.5"
```
