extern crate rand;

#[derive(Debug, Clone, Copy)]
struct ConnectionGene{
    in_node_id: u32,
    out_node_id: u32,
    weight: f64,
    enabled: bool,
    innovation_id: u32,
}

impl ConnectionGene {
    fn generate_weight () -> f64 {
        rand::random::<f64>()
    }
}

impl Default for ConnectionGene{
    fn default () -> ConnectionGene {
        ConnectionGene { in_node_id: 1, out_node_id: 1, weight: ConnectionGene::generate_weight(), enabled: true, innovation_id: 0 }
    }
}

trait Mutation {
}

impl Mutation {
    fn connection_weight (gene: &mut ConnectionGene) {
        gene.weight = ConnectionGene::generate_weight()
    }

    fn add_connection (in_node_id: u32, out_node_id: u32) -> (ConnectionGene) {
        ConnectionGene { in_node_id: in_node_id, out_node_id: out_node_id, ..Default::default() } 
    }

    fn add_node (gene: &mut ConnectionGene, new_node_id: u32) -> (ConnectionGene, ConnectionGene) {
        gene.enabled = false;

        (ConnectionGene {
            in_node_id: gene.in_node_id,
            out_node_id: new_node_id,
            weight: 1f64,
            ..Default::default()
        },
        ConnectionGene {
            in_node_id: new_node_id,
            out_node_id: gene.out_node_id,
            weight: gene.weight,
            ..Default::default()
        })
    }
}

#[derive(Debug)]
struct Genome{
    connection_genes: Vec<ConnectionGene>
}

impl Genome{
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

    let mut gene = ConnectionGene {..Default::default()};
    println!("{:?}", gene); 
    Mutation::connection_weight(&mut gene);
    println!("{:?}", gene); 
    println!("{:?}", Mutation::add_connection(1, 2));
    println!("{:?}", Mutation::add_node(&mut gene, 3));
    println!("{:?}", gene);
}

#[test]
fn mutation_connection_weight(){
    let mut gene = ConnectionGene {..Default::default()};
    let orig_gene = gene.clone();
    Mutation::connection_weight(&mut gene);

    assert!(gene.weight != orig_gene.weight);
}

#[test]
fn mutation_add_connection(){
    let new_gene = Mutation::add_connection(1, 2);

    assert!(new_gene.in_node_id == 1);
    assert!(new_gene.out_node_id == 2);
}

#[test]
fn mutation_add_node(){
    let mut gene = ConnectionGene {..Default::default()};
    let (new_gene1, new_gene2) = Mutation::add_node(&mut gene, 3);

    assert!(!gene.enabled);
    assert!(new_gene1.in_node_id == gene.in_node_id);
    assert!(new_gene1.out_node_id == 3);
    assert!(new_gene2.in_node_id == 3);
    assert!(new_gene2.out_node_id == gene.out_node_id);
}
