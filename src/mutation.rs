use gene::Gene;

pub trait Mutation {}

impl Mutation {
    pub fn connection_weight(gene: &mut Gene, perturbation: bool) {
        let mut new_weight = Gene::generate_weight();
        if perturbation {
            new_weight += gene.weight();
        }
        gene.set_weight(new_weight);
    }

    pub fn add_connection(in_neuron_id: usize, out_neuron_id: usize) -> (Gene) {
        Gene::new(in_neuron_id, out_neuron_id, Gene::generate_weight(), true)
    }

    pub fn add_neuron(gene: &mut Gene, new_neuron_id: usize) -> (Gene, Gene) {
        gene.set_disabled();

        let gen1 = Gene::new(gene.in_neuron_id(), new_neuron_id, 1f64, true);

        let gen2 = Gene::new(new_neuron_id, gene.out_neuron_id(), gene.weight(), true);
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
    use super::*;
    use gene::Gene;

    #[test]
    fn mutate_toggle_gene_should_toggle() {
        let mut gene = Gene::new(0, 1, 1f64, false);

        Mutation::toggle_expression(&mut gene);
        assert!(gene.enabled());

        Mutation::toggle_expression(&mut gene);
        assert!(!gene.enabled());
    }
}
