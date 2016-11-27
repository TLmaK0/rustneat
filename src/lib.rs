#![deny(missing_docs,
        trivial_casts, trivial_numeric_casts,
        unsafe_code,
        unused_import_braces, unused_qualifications)]

#![cfg_attr(feature="clippy", feature(plugin))]
#![cfg_attr(feature="clippy", plugin(clippy))]
#![cfg_attr(feature="clippy", deny(clippy, unicode_not_nfc, wrong_pub_self_convention))]
#![cfg_attr(feature="clippy", allow(use_debug, too_many_arguments))]

//! Implementation of NeuroEvolution of Augmenting Topologies [NEAT]
//! (http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
//! This implementation uses a Continuous-Time Recurrent Neural Network (CTRNN)
//! (Yamauchi and Beer, 1994).

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

/// A collection of genes
pub mod genome;
mod specie;
/// A genome plus fitness
pub mod organism;
/// A collection of species with champion
pub mod population;
mod mutation;
mod gene;
/// Trait to define test parameter
pub mod environment;
mod ctrnn;
mod species_evaluator;
