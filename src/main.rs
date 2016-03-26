mod neat;

fn main(){
    let mut generation = neat::Generation::new();
    let mut genome = generation.create_genome(); 
    println!("NEAT");
    println!("Genome {:?}", genome);

    let mut gene = genome.create_gene(1, 1, 1f64); 
    println!("Original gen {:?}", gene); 
    genome.mutate_connection_weight(&mut gene);
    println!("Mutated weight gen {:?}", gene); 
    println!("Mutated connection {:?}", genome.mutate_add_connection(1, 2));
    println!("Mutated node {:?}", genome.mutate_add_node(&mut gene, 3));
    println!("Final gen {:?}", gene);
}

