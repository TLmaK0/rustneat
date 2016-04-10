use neat::gene::Gene as Gene;

pub trait Mutation {
}

impl Mutation {
    pub fn connection_weight (gene: &mut Gene) {
        gene.weight = Gene::generate_weight()
    }

    pub fn add_connection (in_neuron_id: usize, out_neuron_id: usize) -> (Gene) {
        Gene { 
            in_neuron_id: in_neuron_id,
            out_neuron_id: out_neuron_id,
            ..Default::default()
        } 
    }

    pub fn add_neuron (gene: &mut Gene, new_neuron_id: usize) -> (Gene, Gene) {
        gene.enabled = false;

        let gen1 = Gene {
            in_neuron_id: gene.in_neuron_id,
            out_neuron_id: new_neuron_id,
            weight: 1f64,
            ..Default::default()
        };

        let gen2 = Gene {
            in_neuron_id: new_neuron_id,
            out_neuron_id: gene.out_neuron_id,
            weight: gene.weight,
            ..Default::default()
        };
        (gen1, gen2)
    }
}
