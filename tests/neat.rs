extern crate rustneat;
use rustneat::neat::*;

#[test]
fn should_be_able_to_generate_a_population(){
    let population = Population::create_population(10, 150);
    assert!(population.size() == 150);
}

#[test]
fn population_can_evolve(){
    let mut population = Population::create_population(10, 5);
    population.evolve();
    unimplemented!();
}

