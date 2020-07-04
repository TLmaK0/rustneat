use gene::Gene;
use mutation::Mutation;
use rand::{self, Closed01};
use std::cmp;

/// Vector of Genes
/// Holds a count of last neuron added, similar to Innovation number
#[derive(Default, Debug, Clone)]
#[cfg_attr(feature = "with_serde", derive(Serialize, Deserialize))]
pub struct Genome {
    genes: Vec<Gene>,
    last_neuron_id: usize,
}

const COMPATIBILITY_THRESHOLD: f64 = 3f64; //used to speciate organisms
const MUTATE_CONNECTION_WEIGHT: f64 = 0.90f64;
const MUTATE_ADD_CONNECTION: f64 = 0.005f64;
const MUTATE_ADD_NEURON: f64 = 0.004f64;
const MUTATE_TOGGLE_EXPRESSION: f64 = 0.001f64;
const MUTATE_CONNECTION_WEIGHT_PERTURBED_PROBABILITY: f64 = 0.90f64;
const MUTATE_TOGGLE_BIAS: f64 = 0.01;

impl Genome {
    /// May add a connection &| neuron &| mutat connection weight &|
    /// enable/disable connection
    pub fn mutate(&mut self) {
        if rand::random::<Closed01<f64>>().0 < MUTATE_ADD_CONNECTION || self.genes.is_empty() {
            self.mutate_add_connection();
        };

        if rand::random::<Closed01<f64>>().0 < MUTATE_ADD_NEURON {
            self.mutate_add_neuron();
        };

        if rand::random::<Closed01<f64>>().0 < MUTATE_CONNECTION_WEIGHT {
            self.mutate_connection_weight();
        };

        if rand::random::<Closed01<f64>>().0 < MUTATE_TOGGLE_EXPRESSION {
            self.mutate_toggle_expression();
        };

        if rand::random::<Closed01<f64>>().0 < MUTATE_TOGGLE_BIAS {
            self.mutate_toggle_bias();
        };
    }

    /// Mate two genes
    pub fn mate(&self, other: &Genome, fittest: bool) -> Genome {
        if fittest {
            self.mate_genes(other)
        } else {
            other.mate_genes(self)
        }
    }

    fn mate_genes(&self, other: &Genome) -> Genome {
        let mut genome = Genome::default();
        for gene in &self.genes {
            genome.add_gene({
                //Only mate half of the genes randomly
                if rand::random::<f64>() > 0.5f64 {
                    *gene
                } else {
                    match other.genes.binary_search(gene) {
                        Ok(position) => other.genes[position],
                        Err(_) => *gene,
                    }
                }
            });
        }
        genome
    }
    /// Get vector of all genes in this genome
    pub fn get_genes(&self) -> &Vec<Gene> {
        &self.genes
    }

    /// only allow connected nodes
    #[deprecated(since = "0.3.0", note = "please use `add_gene` instead")]
    pub fn inject_gene(&mut self, in_neuron_id: usize, out_neuron_id: usize, weight: f64) {
        let gene = Gene::new(in_neuron_id, out_neuron_id, weight, true, false);
        self.add_gene(gene);
    }
    /// Number of genes
    pub fn len(&self) -> usize {
        self.last_neuron_id + 1 // first neuron id is 0
    }
    /// is genome empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn mutate_add_connection(&mut self) {
        let mut rng = rand::thread_rng();
        let neuron_ids_to_connect = {
            if self.last_neuron_id == 0 {
                vec![0, 0]
            } else {
                rand::seq::sample_iter(&mut rng, 0..self.last_neuron_id + 1, 2).unwrap()
            }
        };
        self.add_connection(neuron_ids_to_connect[0], neuron_ids_to_connect[1]);
    }

    fn mutate_connection_weight(&mut self) {
        for gene in &mut self.genes {
            Mutation::connection_weight(
                gene,
                rand::random::<f64>() < MUTATE_CONNECTION_WEIGHT_PERTURBED_PROBABILITY,
            );
        }
    }

    fn mutate_toggle_expression(&mut self) {
        let mut rng = rand::thread_rng();
        let selected_gene = rand::seq::sample_iter(&mut rng, 0..self.genes.len(), 1).unwrap()[0];
        Mutation::toggle_expression(&mut self.genes[selected_gene]);
    }

    fn mutate_toggle_bias(&mut self) {
        let mut rng = rand::thread_rng();
        let selected_gene = rand::seq::sample_iter(&mut rng, 0..self.genes.len(), 1).unwrap()[0];
        Mutation::toggle_bias(&mut self.genes[selected_gene]);
    }

    fn mutate_add_neuron(&mut self) {
        let (gene1, gene2) = {
            let mut rng = rand::thread_rng();
            let selected_gene =
                rand::seq::sample_iter(&mut rng, 0..self.genes.len(), 1).unwrap()[0];
            let gene = &mut self.genes[selected_gene];
            self.last_neuron_id += 1;
            Mutation::add_neuron(gene, self.last_neuron_id)
        };
        self.add_gene(gene1);
        self.add_gene(gene2);
    }

    fn add_connection(&mut self, in_neuron_id: usize, out_neuron_id: usize) {
        let gene = Mutation::add_connection(in_neuron_id, out_neuron_id);
        self.add_gene(gene);
    }

    /// Add a new gene and checks if is allowd. Only can connect next neuron or already connected
    /// neurons.
    pub fn add_gene(&mut self, gene: Gene) {
        let max_neuron_id = self.last_neuron_id + 1;

        if gene.in_neuron_id() == gene.out_neuron_id() && gene.in_neuron_id() > max_neuron_id {
            panic!(
                "Try to create a gene neuron unconnected, max neuron id {}, {} -> {}",
                max_neuron_id,
                gene.in_neuron_id(),
                gene.out_neuron_id()
            );
        }

        if gene.in_neuron_id() > self.last_neuron_id {
            self.last_neuron_id = gene.in_neuron_id();
        }
        if gene.out_neuron_id() > self.last_neuron_id {
            self.last_neuron_id = gene.out_neuron_id();
        }
        match self.genes.binary_search(&gene) {
            Ok(pos) => self.genes[pos].set_enabled(),
            Err(_) => self.genes.push(gene),
        }
        self.genes.sort();
    }

    /// Compare another Genome for species equality
    // TODO This should be impl Eq
    pub fn is_same_specie(&self, other: &Genome) -> bool {
        self.compatibility_distance(other) < COMPATIBILITY_THRESHOLD
    }

    /// Total weigths of all genes
    pub fn total_weights(&self) -> f64 {
        let mut total = 0f64;
        for gene in &self.genes {
            total += gene.weight();
        }
        total
    }

    /// Total num genes
    // TODO len() is enough
    pub fn total_genes(&self) -> usize {
        self.genes.len()
    }

    // http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf - Pag. 110
    // I have considered disjoint and excess genes as the same
    fn compatibility_distance(&self, other: &Genome) -> f64 {
        // TODO: optimize this method
        let c2 = 1f64;
        let c3 = 0.4f64;

        // Number of excess
        let n1 = self.genes.len();
        let n2 = other.genes.len();
        let n = cmp::max(n1, n2);

        if n == 0 {
            return 0f64; // no genes in any genome, the genomes are equal
        }

        let z = if n < 20 { 1f64 } else { n as f64 };

        let matching_genes = self
            .genes
            .iter()
            .filter(|i1_gene| other.genes.contains(i1_gene))
            .collect::<Vec<&Gene>>();
        let n3 = matching_genes.len();

        // Disjoint / excess genes
        let d = n1 + n2 - (2 * n3);

        // average weight differences of matching genes
        let w1 = matching_genes.iter().fold(0f64, |acc, &m_gene| {
            acc + (m_gene.weight()
                - &other.genes[other.genes.binary_search(m_gene).unwrap()].weight())
                .abs()
        });

        let w = if n3 == 0 { 0f64 } else { w1 / n3 as f64 };

        // compatibility distance
        (c2 * d as f64 / z) + c3 * w
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::EPSILON;

    #[test]
    fn mutation_connection_weight() {
        let mut genome = Genome::default();
        genome.add_gene(Gene::new(0, 0, 1f64, true, false));
        let orig_gene = genome.genes[0];
        genome.mutate_connection_weight();
        // These should not be same size
        assert!((genome.genes[0].weight() - orig_gene.weight()).abs() > EPSILON);
    }

    #[test]
    fn mutation_add_connection() {
        let mut genome = Genome::default();
        genome.add_connection(1, 2);

        assert!(genome.genes[0].in_neuron_id() == 1);
        assert!(genome.genes[0].out_neuron_id() == 2);
    }

    #[test]
    fn mutation_add_neuron() {
        let mut genome = Genome::default();
        genome.mutate_add_connection();
        genome.mutate_add_neuron();
        assert!(!genome.genes[0].enabled());
        assert!(genome.genes[1].in_neuron_id() == genome.genes[0].in_neuron_id());
        assert!(genome.genes[1].out_neuron_id() == 1);
        assert!(genome.genes[2].in_neuron_id() == 1);
        assert!(genome.genes[2].out_neuron_id() == genome.genes[0].out_neuron_id());
    }

    #[test]
    #[should_panic(expected = "Try to create a gene neuron unconnected, max neuron id 1, 2 -> 2")]
    fn try_to_inject_a_unconnected_neuron_gene_should_panic() {
        let mut genome1 = Genome::default();
        genome1.add_gene(Gene::new(2, 2, 0.5f64, true, false));
    }

    #[test]
    fn two_genomes_without_differences_should_be_in_same_specie() {
        let mut genome1 = Genome::default();
        genome1.add_gene(Gene::new(0, 0, 1f64, true, false));
        genome1.add_gene(Gene::new(0, 1, 1f64, true, false));
        let mut genome2 = Genome::default();
        genome2.add_gene(Gene::new(0, 0, 0f64, true, false));
        genome2.add_gene(Gene::new(0, 1, 0f64, true, false));
        genome2.add_gene(Gene::new(0, 2, 0f64, true, false));
        assert!(genome1.is_same_specie(&genome2));
    }

    #[test]
    fn two_genomes_with_enought_difference_should_be_in_different_species() {
        let mut genome1 = Genome::default();
        genome1.add_gene(Gene::new(0, 0, 1f64, true, false));
        genome1.add_gene(Gene::new(0, 1, 1f64, true, false));
        let mut genome2 = Genome::default();
        genome2.add_gene(Gene::new(0, 0, 5f64, true, false));
        genome2.add_gene(Gene::new(0, 1, 5f64, true, false));
        genome2.add_gene(Gene::new(0, 2, 1f64, true, false));
        genome2.add_gene(Gene::new(0, 3, 1f64, true, false));
        assert!(!genome1.is_same_specie(&genome2));
    }

    #[test]
    fn already_existing_gene_should_be_not_duplicated() {
        let mut genome1 = Genome::default();
        genome1.add_gene(Gene::new(0, 0, 1f64, true, false));
        genome1.add_connection(0, 0);
        assert_eq!(genome1.genes.len(), 1);
        assert!((genome1.get_genes()[0].weight() - 1f64).abs() < EPSILON);
    }

    #[test]
    fn adding_an_existing_gene_disabled_should_enable_original() {
        let mut genome1 = Genome::default();
        genome1.add_gene(Gene::new(0, 1, 0f64, true, false));
        genome1.mutate_add_neuron();
        assert!(!genome1.genes[0].enabled());
        assert!(genome1.genes.len() == 3);
        genome1.add_connection(0, 1);
        assert!(genome1.genes[0].enabled());
        assert!((genome1.genes[0].weight() - 0f64).abs() < EPSILON);
        assert_eq!(genome1.genes.len(), 3);
    }

    #[test]
    fn genomes_with_same_genes_with_little_differences_on_weight_should_be_in_same_specie() {
        let mut genome1 = Genome::default();
        genome1.add_gene(Gene::new(0, 0, 16f64, true, false));
        let mut genome2 = Genome::default();
        genome2.add_gene(Gene::new(0, 0, 16.1f64, true, false));
        assert!(genome1.is_same_specie(&genome2));
    }

    #[test]
    fn genomes_with_same_genes_with_big_differences_on_weight_should_be_in_other_specie() {
        let mut genome1 = Genome::default();
        genome1.add_gene(Gene::new(0, 0, 5f64, true, false));
        let mut genome2 = Genome::default();
        genome2.add_gene(Gene::new(0, 0, 15f64, true, false));
        assert!(!genome1.is_same_specie(&genome2));
    }
}
