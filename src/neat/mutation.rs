use neat::connection_gene::ConnectionGene as ConnectionGene;

pub trait Mutation {
}

impl Mutation {
    pub fn connection_weight (gene: &mut ConnectionGene) {
        gene.weight = ConnectionGene::generate_weight()
    }

    pub fn add_connection (in_node_id: u32, out_node_id: u32) -> (ConnectionGene) {
        ConnectionGene { 
            in_node_id: in_node_id,
            out_node_id: out_node_id,
            ..Default::default()
        } 
    }

    pub fn add_node (gene: &mut ConnectionGene, new_node_id: u32) -> (ConnectionGene, ConnectionGene) {
        gene.enabled = false;

        let gen1 = ConnectionGene {
            in_node_id: gene.in_node_id,
            out_node_id: new_node_id,
            weight: 1f64,
            ..Default::default()
        };

        let gen2 = ConnectionGene {
            in_node_id: new_node_id,
            out_node_id: gene.out_node_id,
            weight: gene.weight,
            ..Default::default()
        };

        (gen1, gen2)
    }
}
