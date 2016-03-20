extern crate rustneat;
use rustneat::neat::*;

#[test]
fn mutation_connection_weight(){
    let mut genome = Genome::new();
    let mut gene = genome.create_gene();
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
    assert!(new_gene.innovation == 1);
}

#[test]
fn mutation_add_node(){
    let mut genome = Genome::new();
    let mut gene = genome.create_gene();
    let (new_gene1, new_gene2) = genome.mutate_add_node(&mut gene, 3);

    assert!(!gene.enabled);
    assert!(new_gene1.in_node_id == gene.in_node_id);
    assert!(new_gene1.out_node_id == 3);
    assert!(new_gene2.in_node_id == 3);
    assert!(new_gene2.out_node_id == gene.out_node_id);
    assert!(new_gene1.innovation == 2);
    assert!(new_gene2.innovation == 3);
}

