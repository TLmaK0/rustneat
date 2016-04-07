extern crate rustneat;
use rustneat::neat::*;

#[test]
fn should_be_able_to_generate_a_population(){
    let population = Population::create_population(150);
    assert!(population.size() == 150);
}

#[test]
fn population_can_evolve(){
    let mut population = Population::create_population(1);
    population.evolve();
    let genome = &population.organisms[0].genome;
    assert!(genome.total_genes() == 1);
    assert!(genome.total_weights() != 0f64);
}

