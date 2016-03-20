use neat::connection_gene::ConnectionGene as ConnectionGene;
use neat::mutation::Mutation as Mutation;
use neat::generation::Generation as Generation;

#[derive(Debug)]
pub struct Genome{
    connection_genes: Vec<ConnectionGene>,
    pub generation: Generation,
}

impl Genome{
    pub fn new(generation: Generation) -> Genome {
        Genome { generation: generation, connection_genes: vec![Default::default(); 0]}
    }

    pub fn create_gene(&mut self) -> ConnectionGene {
        let gene = ConnectionGene {
            innovation: self.generation.get_innovation_id(),
            ..Default::default()
        };
        self.connection_genes.push(gene);
        gene
    }

    pub fn mutate_add_node(&mut self, gene: &mut ConnectionGene, new_node_id: u32) -> (ConnectionGene, ConnectionGene) {
        let (gene1, gene2) = Mutation::add_node(gene, new_node_id, self);
        self.connection_genes.push(gene1);
        self.connection_genes.push(gene2);
        (gene1, gene2) 
    }

    pub fn mutate_connection_weight(&mut self, gene: &mut ConnectionGene){
        Mutation::connection_weight(gene)
    }

    pub fn mutate_add_connection(&mut self, in_node_id: u32, out_node_id: u32) -> ConnectionGene {
        Mutation::add_connection(in_node_id, out_node_id, self)
    }
}
