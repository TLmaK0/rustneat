extern crate rustneat;
use rustneat::neat::*;

#[test]
fn two_genomes_without_differences_should_be_in_same_specie(){
    let mut genome1 = Genome::new();
    genome1.create_gene(1, 1, 1f64);
    genome1.create_gene(1, 2, 1f64);
    let mut genome2 = Genome::new();
    genome2.create_gene(1, 1, 0f64);
    genome2.create_gene(1, 2, 0f64);
    genome2.create_gene(1, 3, 0f64);
    assert!(genome1.is_same_specie(&genome2));
}

#[test]
fn two_genomes_with_enought_difference_should_be_in_different_species(){
    let mut genome1 = Genome::new();
    genome1.create_gene(1, 1, 1f64);
    genome1.create_gene(1, 2, 1f64);
    let mut genome2 = Genome::new();
    genome2.create_gene(1, 3, 1f64);
    genome2.create_gene(1, 4, 1f64);
    assert!(!genome1.is_same_specie(&genome2));
}

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

