#![deny(
    missing_docs, trivial_casts, trivial_numeric_casts, unsafe_code, unused_import_braces,
    unused_qualifications
)]
#![cfg_attr(feature = "clippy", feature(plugin))]
#![cfg_attr(feature = "clippy", plugin(clippy))]
#![cfg_attr(feature = "clippy", deny(clippy, unicode_not_nfc, wrong_pub_self_convention))]
#![cfg_attr(feature = "clippy", allow(use_debug, too_many_arguments))]

//! Implementation of `NeuroEvolution` of Augmenting Topologies [NEAT]
//! (http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
//! This implementation uses a Continuous-Time Recurrent Neural Network (CTRNN)
//! (Yamauchi and Beer, 1994).

#[cfg(feature = "telemetry")]
#[macro_use]
extern crate rusty_dashed;

#[cfg(feature = "telemetry")]
#[macro_use]
extern crate serde_derive;

#[cfg(feature = "telemetry")]
extern crate serde_json;

pub use self::genome::*;
pub use self::environment::Environment;
pub use self::population::Population;
pub use self::specie::Specie;
pub use self::species_evaluator::SpeciesEvaluator;

/// Contains the definition of the genome of neural networks, which is the basic building block of
/// an organism (and in many cases, the only building block).
pub mod nn;
/// Trait to define test parameter
mod environment;
/// A collection of genes
mod genome;
/// A collection of species with champion
mod population;
mod specie;
mod species_evaluator;
