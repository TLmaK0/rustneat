#![cfg_attr(feature="clippy", feature(plugin))]
#![cfg_attr(feature="clippy", plugin(clippy))]
#![cfg_attr(feature="clippy", deny(clippy, unicode_not_nfc, wrong_pub_self_convention))]
#![cfg_attr(feature="clippy", allow(use_debug))]
#[macro_use]
extern crate lazy_static;
extern crate conv;
extern crate rand;
extern crate rulinalg;
extern crate num_cpus;
extern crate crossbeam;

pub use ctrnn::CtrnnNeuralNetwork;
pub use self::ctrnn::Ctrnn;
pub use self::environment::Environment;
pub use self::gene::Gene;
pub use self::genome::Genome;
pub use self::organism::Organism;
pub use self::population::Population;
pub use self::specie::Specie;
pub use self::species_evaluator::SpeciesEvaluator;

pub mod genome;
mod specie;
pub mod organism;
pub mod population;
mod mutation;
mod gene;
pub mod environment;
mod ctrnn;
mod species_evaluator;
