use crate::gene::Gene;
use rand::Rng;

pub trait Mutation {}

impl dyn Mutation {
    pub fn connection_weight<R: Rng + ?Sized>(gene: &mut Gene, perturbation: bool, rng: &mut R) {
        let mut new_weight = Gene::generate_weight_with_rng(rng);
        if perturbation {
            new_weight += gene.weight();
        }
        gene.set_weight(new_weight);
    }

    pub fn add_connection<R: Rng + ?Sized>(
        in_neuron_id: usize,
        out_neuron_id: usize,
        rng: &mut R,
    ) -> Gene {
        Gene::new(
            in_neuron_id,
            out_neuron_id,
            Gene::generate_weight_with_rng(rng),
            true,
        )
    }

    /// Disables `gene` and returns `in -> new` (weight `input_weight`) and
    /// `new -> out` (the weight of `gene`).
    pub fn add_neuron(gene: &mut Gene, new_neuron_id: usize, input_weight: f64) -> (Gene, Gene) {
        gene.set_disabled();

        let gen1 = Gene::new(gene.in_neuron_id(), new_neuron_id, input_weight, true);

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
    use crate::gene::Gene;

    #[test]
    fn mutate_toggle_gene_should_toggle() {
        let mut gene = Gene::new(0, 1, 1f64, false);

        <dyn Mutation>::toggle_expression(&mut gene);
        assert!(gene.enabled());

        <dyn Mutation>::toggle_expression(&mut gene);
        assert!(!gene.enabled());
    }
}
