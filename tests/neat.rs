extern crate rustneat;
use rustneat::neat::*;

#[test]
fn mutation_connection_weight(){
    let mut genome = Genome::new();
    let mut gene = genome.create_gene(1, 1, 1f64);
    let orig_gene = gene.clone();
    genome.mutate_connection_weight(&mut gene);

    assert!(gene.weight != orig_gene.weight);
}

#[test]
fn mutation_add_connection(){
    let mut genome = Genome::new();
    let new_gene = genome.mutate_add_connection(1, 2);

    assert!(new_gene.in_node_id == 1);
    assert!(new_gene.out_node_id == 2);
}

#[test]
fn mutation_add_node(){
    let mut genome = Genome::new();
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
