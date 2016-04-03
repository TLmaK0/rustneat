extern crate rustneat;
use rustneat::neat::*;

#[test]
fn mutation_connection_weight(){
    let mut genome = Genome::new(10, 10);
    let mut gene = genome.create_gene(1, 1, 1f64);
    let orig_gene = gene.clone();
    genome.mutate_connection_weight(&mut gene);

    assert!(gene.weight != orig_gene.weight);
}

#[test]
fn mutation_add_connection(){
    let mut genome = Genome::new(10, 10);
    let new_gene = genome.mutate_add_connection(1, 2);

    assert!(new_gene.in_node_id == 1);
    assert!(new_gene.out_node_id == 2);
}

#[test]
fn mutation_add_node(){
    let mut genome = Genome::new(10, 10);
    let mut gene = genome.create_gene(1, 1, 1f64);
    let (new_gene1, new_gene2) = genome.mutate_add_node(&mut gene, 3);

    assert!(!gene.enabled);
    assert!(new_gene1.in_node_id == gene.in_node_id);
    assert!(new_gene1.out_node_id == 3);
    assert!(new_gene2.in_node_id == 3);
    assert!(new_gene2.out_node_id == gene.out_node_id);
}

#[test]
fn two_genomes_without_differences_should_be_in_same_specie(){
    let mut genome1 = Genome::new(10, 10);
    genome1.create_gene(1, 1, 1f64);
    genome1.create_gene(1, 2, 1f64);
    let mut genome2 = Genome::new(10, 10);
    genome2.create_gene(1, 1, 0f64);
    genome2.create_gene(1, 2, 0f64);
    genome2.create_gene(1, 3, 0f64);
    assert!(genome1.is_same_specie(&genome2));
}

#[test]
fn two_genomes_with_enought_difference_should_be_in_different_species(){
    let mut genome1 = Genome::new(10, 10);
    genome1.create_gene(1, 1, 1f64);
    genome1.create_gene(1, 2, 1f64);
    let mut genome2 = Genome::new(10, 10);
    genome2.create_gene(1, 3, 1f64);
    genome2.create_gene(1, 4, 1f64);
    assert!(!genome1.is_same_specie(&genome2));
}

#[test]
fn should_be_able_to_generate_a_population(){
    let population = Population::create_population(10, 10, 150);
    assert!(population.size() == 150);
}

#[test]
fn population_can_evolve(){
    let mut population = Population::create_population(10, 10, 5);
    let total_weights = population.total_weights();
    let total_genes = population.total_genes();

    population.evolve();

    assert!(population.size() == 5);
    assert!(population.total_weights() != total_weights || population.total_genes() != total_genes);
}
