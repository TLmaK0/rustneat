use neat::gene::Gene as Gene;

pub trait Mutation {
}

impl Mutation {
    pub fn connection_weight (gene: &mut Gene) {
        gene.weight = Gene::generate_weight()
    }

    pub fn add_connection (in_node_id: u32, out_node_id: u32) -> (Gene) {
        Gene { 
            in_node_id: in_node_id,
            out_node_id: out_node_id,
            ..Default::default()
        } 
    }

    pub fn add_node (gene: &mut Gene, new_node_id: u32) -> (Gene, Gene) {
        gene.enabled = false;

        let gen1 = Gene {
            in_node_id: gene.in_node_id,
            out_node_id: new_node_id,
            weight: 1f64,
            ..Default::default()
        };

        let gen2 = Gene {
            in_node_id: new_node_id,
            out_node_id: gene.out_node_id,
            weight: gene.weight,
            ..Default::default()
        };

        (gen1, gen2)
    }
}
