extern crate conv;

use neat::connection_gene::ConnectionGene as ConnectionGene;
use neat::mutation::Mutation as Mutation;
use self::conv::prelude::*;


#[derive(Debug)]
pub struct Genome{
    connection_genes: Vec<ConnectionGene>,
}

impl Genome{
    pub fn new() -> Genome {
        Genome { connection_genes: vec![]}
    }

    pub fn create_gene(&mut self, in_node_id: u32, out_node_id: u32, weight: f64) -> ConnectionGene {
        let gene = ConnectionGene {
            in_node_id: in_node_id,
            out_node_id: out_node_id,
            weight: weight,
            ..Default::default()
        };
        self.connection_genes.push(gene);
        gene
    }

    pub fn mutate_add_node(&mut self, gene: &mut ConnectionGene, new_node_id: u32) -> (ConnectionGene, ConnectionGene) {
        let (gene1, gene2) = Mutation::add_node(gene, new_node_id);
        self.connection_genes.push(gene1);
        self.connection_genes.push(gene2);
        (gene1, gene2) 
    }

    pub fn mutate_connection_weight(&mut self, gene: &mut ConnectionGene){
        Mutation::connection_weight(gene)
    }

    pub fn mutate_add_connection(&mut self, in_node_id: u32, out_node_id: u32) -> ConnectionGene {
        Mutation::add_connection(in_node_id, out_node_id)
    }

    pub fn is_same_specie(&self, other: &Genome) -> bool{
        self.compatibility_distance(other) < 1f64
    }

    //http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf - Pag. 110
    //I have consider disjoint and excess genes as the same
    fn compatibility_distance(&self, other: &Genome) -> f64 {
        //TODO: optimize compatibility_distance
        let c2 = 0.5f64;
        let c3 = 0.5f64;

        //Number of excess
        let n1 = self.connection_genes.len().value_as::<f64>().unwrap();
        let n2 = other.connection_genes.len().value_as::<f64>().unwrap();

        let matching_genes  = self.connection_genes.iter().filter(|i1_gene| other.connection_genes.contains(i1_gene)).collect::<Vec<&ConnectionGene>>();

        let n3 = matching_genes.len().value_as::<f64>().unwrap();

        //Disjoint genes
        let d = n1 + n2 - (2f64 * n3);

        //average weight differences of matching genes
        let w1 = matching_genes.iter().fold(0f64, |acc, &m_gene| acc + (m_gene.weight + other.connection_genes.get(other.connection_genes.binary_search(m_gene).unwrap()).unwrap().weight)).abs();

        let w = w1 / n3;

        //compatibility distance
        let n = n1.max(n2);

        let delta = (c2 * d / n) + c3 * w;
        delta
    }
}
