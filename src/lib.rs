#![deny(
    missing_docs,
    trivial_casts,
    trivial_numeric_casts,
    unsafe_code,
    unused_import_braces,
    unused_qualifications
)]
#![cfg_attr(feature = "clippy", feature(plugin))]
#![cfg_attr(feature = "clippy", plugin(clippy))]
#![cfg_attr(
    feature = "clippy",
    deny(clippy, unicode_not_nfc, wrong_pub_self_convention)
)]
#![cfg_attr(feature = "clippy", allow(use_debug, too_many_arguments))]

//! Implementation of `NeuroEvolution` of Augmenting Topologies [NEAT]
//! (http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
//! This implementation uses a Continuous-Time Recurrent Neural Network (CTRNN)
//! (Yamauchi and Beer, 1994).

#[cfg(feature = "telemetry")]
#[macro_use]
extern crate rusty_dashed;

extern crate conv;
extern crate crossbeam;
extern crate num_cpus;
extern crate rand;
extern crate rulinalg;

#[cfg(feature = "telemetry")]
extern crate serde_json;

pub use self::ctrnn::{Ctrnn, Integrator};
pub use self::environment::Environment;
pub use self::gene::Gene;
pub use self::genome::Genome;
pub use self::innovation::InnovationRegistry;
pub use self::mutation_config::{MutationConfig, SpeciesAssignment};
pub use self::organism::{Organism, OutputActivation};
pub use self::phenotype::Phenotype;
pub use self::population::{FitnessError, Population};
pub use self::specie::Specie;
pub use self::species_evaluator::SpeciesEvaluator;
pub use ctrnn::CtrnnNeuralNetwork;

/// The Rust code blocks of `README.md`, compiled and run as doc-tests so
/// the README cannot drift from the API.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;

mod ctrnn;
/// Trait to define test parameter
pub mod environment;
mod gene;
/// A collection of genes
pub mod genome;
mod innovation;
mod mutation;
/// Configurable mutation rates
pub mod mutation_config;
/// A genome plus fitness
pub mod organism;
mod phenotype;
/// A collection of species with champion
pub mod population;
mod specie;
mod species_evaluator;
