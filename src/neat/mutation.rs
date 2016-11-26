use neat::gene::Gene;

pub trait Mutation {}

impl Mutation {
    pub fn connection_weight(gene: &mut Gene, perturbation: bool) {
        let mut new_weight = Gene::generate_weight();
        if perturbation {
            new_weight = gene.weight() + new_weight;
        }
        gene.set_weight(new_weight);
    }

    pub fn add_connection(in_neuron_id: usize, out_neuron_id: usize) -> (Gene) {
        Gene {
            in_neuron_id: in_neuron_id,
            out_neuron_id: out_neuron_id,
            ..Default::default()
        }
    }

    pub fn add_neuron(gene: &mut Gene, new_neuron_id: usize) -> (Gene, Gene) {
        gene.set_disabled();

        let gen1 = Gene {
            in_neuron_id: gene.in_neuron_id(),
            out_neuron_id: new_neuron_id,
            weight: 1f64,
            ..Default::default()
        };

        let gen2 = Gene {
            in_neuron_id: new_neuron_id,
            out_neuron_id: gene.out_neuron_id(),
            weight: gene.weight(),
            ..Default::default()
        };
        (gen1, gen2)
    }

    pub fn toggle_expression(gene: &mut Gene) {
        if gene.enabled() {
            gene.set_disabled()
        } else {
            gene.set_enabled()
        }
    }
}

#[cfg(test)]
mod tests {
    use neat::*;
    use neat::mutation::Mutation;

    #[test]
    fn mutate_toggle_gene_should_toggle() {
        let mut gene = Gene {
            in_neuron_id: 0,
            out_neuron_id: 1,
            weight: 1f64,
            enabled: false,
        };

        Mutation::toggle_expression(&mut gene);
        assert!(gene.enabled == true);

        Mutation::toggle_expression(&mut gene);
        assert!(gene.enabled == false);
    }
}
