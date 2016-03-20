mod neat;

fn main(){
    let mut genome = neat::Genome::new();
    println!("NEAT");
    println!("Genome {:?}", genome);

    let mut gene = genome.create_gene(); 
    println!("Original gen {:?}", gene); 
    neat::Mutation::connection_weight(&mut gene);
    println!("Mutated weight gen {:?}", gene); 
    println!("Mutated connections {:?}", neat::Mutation::add_connection(1, 2, &mut genome));
    println!("Mutated node {:?}", neat::Mutation::add_node(&mut gene, 3, &mut genome));
    println!("Final gen {:?}", gene);
}

