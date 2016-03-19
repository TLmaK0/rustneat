extern crate rand;

#[derive(Debug, Clone, Copy)]
struct ConnectionGene{
    in_node_id: u32,
    out_node_id: u32,
    weight: f64,
    enabled: bool,
    innovation_id: u32,
}

impl Default for ConnectionGene{
    fn default () -> ConnectionGene {
        ConnectionGene { in_node_id: 1, out_node_id: 1, weight: 1_f64, enabled: true, innovation_id: 0 }
    }
}

struct Mutation {
    gen: ConnectionGene
}

impl Mutation {
    fn new (gen_orig: ConnectionGene) -> Mutation {
        Mutation { gen: gen_orig }
    }

    fn mutate_weight (&self) -> ConnectionGene {
        ConnectionGene { weight: rand::random::<f64>(), ..self.gen }
    }
}

#[derive(Debug)]
struct Genome{
    connection_genes: Vec<ConnectionGene>
}

impl Default for Genome{
    fn default () -> Genome {
        Genome { connection_genes: vec![Default::default(); 0]}
    }
}

fn main(){
    let genome = Genome {..Default::default()};
    println!("NEAT");
    println!("{:?}", genome);
    let mutation = Mutation::new(ConnectionGene {..Default::default()});
    println!("{:?}", mutation.mutate_weight());
}
