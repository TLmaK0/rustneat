extern crate rustneat;
use rustneat::neat::*;

#[test]
fn mutation_connection_weight(){
    let generation = Generation::new(); 
    let mut genome = Genome::new(generation);
    let mut gene = genome.create_gene();
    let orig_gene = gene.clone();
    genome.mutate_connection_weight(&mut gene);

    assert!(gene.weight != orig_gene.weight);
}

#[test]
fn mutation_add_connection(){
    let generation = Generation::new(); 
    let mut genome = Genome::new(generation);
    let new_gene = genome.mutate_add_connection(1, 2);

    assert!(new_gene.in_node_id == 1);
    assert!(new_gene.out_node_id == 2);
    assert!(new_gene.innovation == 1);
}

#[test]
fn mutation_add_node(){
    let generation = Generation::new(); 
    let mut genome = Genome::new(generation);
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

#[test]
fn mutation_on_same_gene_returns_same_innovation(){
    let generation = Generation::new(); 
    let mut genome = Genome::new(generation);
    let mut gene = genome.create_gene();
    let (new_gene1, new_gene2) = genome.mutate_add_node(&mut gene, 3);
    let (new_gene3, new_gene4) = genome.mutate_add_node(&mut gene, 3);

    assert!(new_gene1.innovation == new_gene3.innovation); 
    assert!(new_gene2.innovation == new_gene4.innovation); 
}

#[test]
fn mutation_on_different_gene_returns_different_innovation(){
    let generation = Generation::new(); 
    let mut genome = Genome::new(generation);
    let mut gene1 = genome.create_gene();
    let mut gene2 = genome.create_gene();
    let (new_gene1, new_gene2) = genome.mutate_add_node(&mut gene1, 3);
    let (new_gene3, new_gene4) = genome.mutate_add_node(&mut gene2, 3);

    assert!(new_gene1.innovation != new_gene3.innovation); 
    assert!(new_gene2.innovation != new_gene4.innovation); 
}


