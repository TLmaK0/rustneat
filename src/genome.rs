use crate::gene::Gene;
use crate::mutation::Mutation;
use crate::mutation_config::MutationConfig;
use rand::Rng;
use std::cmp;

/// Vector of Genes
/// Holds a count of last neuron added, similar to Innovation number
#[derive(Default, Debug, Clone)]
pub struct Genome {
    genes: Vec<Gene>,
    last_neuron_id: usize,
}

pub(crate) const MUTATE_CONNECTION_WEIGHT: f64 = 0.90f64;
pub(crate) const MUTATE_ADD_CONNECTION: f64 = 0.005f64;
pub(crate) const MUTATE_ADD_NEURON: f64 = 0.004f64;
pub(crate) const MUTATE_TOGGLE_EXPRESSION: f64 = 0.001f64;
pub(crate) const MUTATE_CONNECTION_WEIGHT_PERTURBED_PROBABILITY: f64 = 0.90f64;
pub(crate) const MUTATE_TOGGLE_BIAS: f64 = 0.01;
pub(crate) const COMPATIBILITY_THRESHOLD: f64 = 3f64;

impl Genome {
    /// Create a genome from serialized gene data
    pub fn from_genes(genes: Vec<Gene>, last_neuron_id: usize) -> Genome {
        let mut genome = Genome {
            genes: Vec::new(),
            last_neuron_id,
        };
        for gene in genes {
            // Directly add gene without validation since we're reconstructing
            match genome.genes.binary_search(&gene) {
                Ok(pos) => genome.genes[pos].set_enabled(),
                Err(_) => genome.genes.push(gene),
            }
        }
        genome.genes.sort();
        genome
    }

    ///Add initial input and output neurons interconnected
    pub fn new_initialized(input_neurons: usize, output_neurons: usize) -> Genome {
        Genome::new_initialized_with_rng(input_neurons, output_neurons, &mut rand::thread_rng())
    }

    /// Like `new_initialized`, drawing the initial weights in [-1, 1] from `rng`
    pub fn new_initialized_with_rng<R: Rng + ?Sized>(
        input_neurons: usize,
        output_neurons: usize,
        rng: &mut R,
    ) -> Genome {
        let mut genome = Genome::default();
        for i in 0..input_neurons {
            for o in 0..output_neurons {
                let weight = Gene::generate_weight_with_rng(rng);
                genome.add_gene(Gene::new(i, input_neurons + o, weight, true, false));
            }
        }
        genome
    }

    /// Create genome with input/output neuron IDs but NO connections.
    /// NEAT will discover connections through mutation.
    pub fn new_unconnected(input_neurons: usize, output_neurons: usize) -> Genome {
        Genome {
            genes: Vec::new(),
            last_neuron_id: if input_neurons + output_neurons > 0 {
                input_neurons + output_neurons - 1
            } else {
                0
            },
        }
    }

    /// May add a connection &| neuron &| mutat connection weight &|
    /// enable/disable connection
    pub fn mutate(&mut self) {
        self.mutate_with_config(&MutationConfig::default());
    }

    /// Mutate using specific mutation rates from config
    pub fn mutate_with_config(&mut self, config: &MutationConfig) {
        self.mutate_with_config_and_rng(config, &mut rand::thread_rng());
    }

    /// Mutate using specific mutation rates from config, drawing every random
    /// decision from `rng`
    pub fn mutate_with_config_and_rng<R: Rng + ?Sized>(
        &mut self,
        config: &MutationConfig,
        rng: &mut R,
    ) {
        if rng.gen::<f64>() < config.add_connection_rate || self.genes.is_empty() {
            self.mutate_add_connection_with_config(config, rng);
        };

        if rng.gen::<f64>() < config.add_neuron_rate {
            self.mutate_add_neuron(rng);
        };

        if rng.gen::<f64>() < config.weight_mutation_rate {
            self.mutate_connection_weight_with_config(config, rng);
        };

        if rng.gen::<f64>() < config.toggle_expression_rate {
            self.mutate_toggle_expression(rng);
        };

        if rng.gen::<f64>() < config.toggle_bias_rate {
            self.mutate_toggle_bias(rng);
        };
    }

    /// Mate two genes
    pub fn mate(&self, other: &Genome, fittest: bool) -> Genome {
        self.mate_with_rng(other, fittest, &mut rand::thread_rng())
    }

    /// Mate two genes, drawing every random decision from `rng`
    pub fn mate_with_rng<R: Rng + ?Sized>(
        &self,
        other: &Genome,
        fittest: bool,
        rng: &mut R,
    ) -> Genome {
        if fittest {
            self.mate_genes(other, rng)
        } else {
            other.mate_genes(self, rng)
        }
    }

    fn mate_genes<R: Rng + ?Sized>(&self, other: &Genome, rng: &mut R) -> Genome {
        let mut genome = Genome::default();
        // NEAT paper: 40% of crossovers use average weights for matching genes
        let use_average_weights = rng.gen::<f64>() < 0.4;

        for gene in &self.genes {
            let mut child_gene = match other.genes.binary_search(gene) {
                Ok(position) => {
                    // Matching gene in both parents
                    if use_average_weights {
                        // Inherit average weight from both parents
                        let mut avg_gene = *gene;
                        avg_gene.set_weight((gene.weight() + other.genes[position].weight()) / 2.0);
                        avg_gene
                    } else if rng.gen::<f64>() > 0.5 {
                        *gene
                    } else {
                        other.genes[position]
                    }
                }
                Err(_) => {
                    // Disjoint/excess gene: inherit from fitter parent (self)
                    *gene
                }
            };

            // NEAT rule: if gene is disabled in either parent, 25% chance offspring has it disabled
            // (reduced from 75% to allow more gene reactivation)
            let other_gene_disabled = other
                .genes
                .binary_search(gene)
                .ok()
                .map(|pos| !other.genes[pos].enabled())
                .unwrap_or(false);

            if !gene.enabled() || other_gene_disabled {
                if rng.gen::<f64>() < 0.25 {
                    child_gene.set_disabled();
                } else {
                    child_gene.set_enabled();
                }
            }

            genome.add_gene(child_gene);
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

    fn mutate_add_connection_with_config<R: Rng + ?Sized>(
        &mut self,
        config: &MutationConfig,
        rng: &mut R,
    ) {
        let neuron_ids_to_connect = {
            if self.last_neuron_id == 0 {
                vec![0, 0]
            } else {
                pick_two_neurons(rng, self.last_neuron_id + 1)
            }
        };
        let gene = Gene::new(
            neuron_ids_to_connect[0],
            neuron_ids_to_connect[1],
            Gene::generate_weight_in_range_with_rng(config.weight_init_range, rng),
            true,
            false,
        );
        self.add_gene(gene);
    }

    fn mutate_connection_weight_with_config<R: Rng + ?Sized>(
        &mut self,
        config: &MutationConfig,
        rng: &mut R,
    ) {
        for gene in &mut self.genes {
            if rng.gen::<f64>() < config.weight_perturbation_rate {
                // Perturbation: add small random value
                let perturbation =
                    Gene::generate_weight_in_range_with_rng(config.weight_mutate_power, rng);
                gene.set_weight(gene.weight() + perturbation);
            } else {
                // Replace with new random weight
                gene.set_weight(Gene::generate_weight_in_range_with_rng(
                    config.weight_init_range,
                    rng,
                ));
            }
        }
    }

    fn mutate_toggle_expression<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        let selected_gene = rng.gen_range(0..self.genes.len());
        <dyn Mutation>::toggle_expression(&mut self.genes[selected_gene]);
    }

    fn mutate_toggle_bias<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        let selected_gene = rng.gen_range(0..self.genes.len());
        <dyn Mutation>::toggle_bias(&mut self.genes[selected_gene]);
    }

    fn mutate_add_neuron<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        let (gene1, gene2) = {
            let selected_gene = rng.gen_range(0..self.genes.len());
            let gene = &mut self.genes[selected_gene];
            self.last_neuron_id += 1;
            <dyn Mutation>::add_neuron(gene, self.last_neuron_id)
        };
        self.add_gene(gene1);
        self.add_gene(gene2);
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
        self.is_same_specie_with_threshold(other, COMPATIBILITY_THRESHOLD)
    }

    /// Compare another Genome for species equality using a custom threshold
    pub fn is_same_specie_with_threshold(&self, other: &Genome, threshold: f64) -> bool {
        self.compatibility_distance(other) < threshold
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
        let c3 = 0.2f64;

        // Number of excess
        let n1 = self.genes.len();
        let n2 = other.genes.len();
        let n = cmp::max(n1, n2);

        if n == 0 {
            return 0f64; // no genes in any genome, the genomes are equal
        }

        let matching_genes = self
            .genes
            .iter()
            .filter(|i1_gene| other.genes.contains(i1_gene))
            .collect::<Vec<&Gene>>();
        let n3 = matching_genes.len();

        // Disjoint / excess genes
        let d = n1 + n2 - (2 * n3);

        // average weight differences of matching genes
        let mut w = matching_genes.iter().fold(0f64, |acc, &m_gene| {
            acc + (m_gene.weight()
                - &other.genes[other.genes.binary_search(m_gene).unwrap()].weight())
                .abs()
        });

        // if no matching genes then are completely different
        w = if n3 == 0 {
            COMPATIBILITY_THRESHOLD
        } else {
            w / n3 as f64
        };

        // compatibility distance
        (c2 * d as f64 / n as f64) + c3 * w
    }
}

/// Two distinct neuron ids in `0..neurons` (`neurons >= 2`), as
/// `[in_neuron_id, out_neuron_id]`.
///
/// Reservoir sampling exactly as `rand::seq::sample_iter(rng, 0..neurons, 2)`
/// did in rand 0.4, so the upgrade keeps the same distribution of connection
/// directions (the pair's order is not uniform: it depends on the ids).
/// Changing it to a uniformly ordered pair changes the XOR results a lot,
/// so it is left as it was.
fn pick_two_neurons<R: Rng + ?Sized>(rng: &mut R, neurons: usize) -> Vec<usize> {
    let mut reservoir = vec![0, 1];
    for (i, neuron_id) in (2..neurons).enumerate() {
        let k = rng.gen_range(0..i + 3);
        if k < 2 {
            reservoir[k] = neuron_id;
        }
    }
    reservoir
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::f64::EPSILON;

    fn rng() -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(0)
    }

    impl Genome {
        fn mutate_add_connection(&mut self) {
            self.mutate_add_connection_with_config(&MutationConfig::default(), &mut rng());
        }

        fn mutate_connection_weight(&mut self) {
            self.mutate_connection_weight_with_config(&MutationConfig::default(), &mut rng());
        }

        fn add_connection(&mut self, in_neuron_id: usize, out_neuron_id: usize) {
            let gene = <dyn Mutation>::add_connection(in_neuron_id, out_neuron_id, &mut rng());
            self.add_gene(gene);
        }
    }

    #[test]
    fn pick_two_neurons_keeps_rand_0_4_order_distribution() {
        // rand 0.4 sample_iter(0..3, 2): [0,1] then 2 replaces slot k in 0..3
        // if k < 2, so the only possible pairs are [2,1], [0,2], [0,1].
        let mut rng = rng();
        let mut seen = std::collections::BTreeMap::new();
        for _ in 0..3000 {
            *seen.entry(pick_two_neurons(&mut rng, 3)).or_insert(0) += 1;
        }
        let pairs: Vec<&Vec<usize>> = seen.keys().collect();
        assert_eq!(pairs, vec![&vec![0, 1], &vec![0, 2], &vec![2, 1]]);
        for &count in seen.values() {
            assert!(count > 800 && count < 1200, "{:?}", seen);
        }
        assert_eq!(pick_two_neurons(&mut rng, 2), vec![0, 1]);
    }

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
        genome.mutate_add_neuron(&mut rng());
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
        genome2.add_gene(Gene::new(0, 0, 20f64, true, false));
        genome2.add_gene(Gene::new(0, 1, 20f64, true, false));
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
        genome1.mutate_add_neuron(&mut rng());
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
        genome2.add_gene(Gene::new(0, 0, 25f64, true, false));
        assert!(!genome1.is_same_specie(&genome2));
    }

    #[test]
    fn genomes_initialized_has_correct_neurons() {
        let genome1 = Genome::new_initialized(2, 3);
        assert_eq!(genome1.total_genes(), 6);
        assert_eq!(genome1.genes[0].in_neuron_id(), 0);
        assert_eq!(genome1.genes[0].out_neuron_id(), 2);
        assert_eq!(genome1.genes[1].in_neuron_id(), 0);
        assert_eq!(genome1.genes[1].out_neuron_id(), 3);
        assert_eq!(genome1.genes[2].in_neuron_id(), 0);
        assert_eq!(genome1.genes[2].out_neuron_id(), 4);
        assert_eq!(genome1.genes[3].in_neuron_id(), 1);
        assert_eq!(genome1.genes[3].out_neuron_id(), 2);
        assert_eq!(genome1.genes[4].in_neuron_id(), 1);
        assert_eq!(genome1.genes[4].out_neuron_id(), 3);
        assert_eq!(genome1.genes[5].in_neuron_id(), 1);
        assert_eq!(genome1.genes[5].out_neuron_id(), 4);
    }

    #[test]
    fn crossover_disabled_gene_should_stay_disabled_25_percent() {
        // Parent 1 has disabled gene, parent 2 has enabled gene
        let mut genome1 = Genome::default();
        genome1.add_gene(Gene::new(0, 1, 1.0, false, false)); // disabled

        let mut genome2 = Genome::default();
        genome2.add_gene(Gene::new(0, 1, 1.0, true, false)); // enabled

        // Run many trials to check probability
        let trials = 1000;
        let mut disabled_count = 0;
        for _ in 0..trials {
            let child = genome1.mate(&genome2, true);
            if !child.genes[0].enabled() {
                disabled_count += 1;
            }
        }

        // Should be approximately 25% disabled (allow ±10% margin)
        // (reduced from NEAT's 75% to allow more gene reactivation)
        let disabled_ratio = disabled_count as f64 / trials as f64;
        assert!(
            disabled_ratio > 0.15 && disabled_ratio < 0.35,
            "Expected ~25% disabled, got {}%",
            disabled_ratio * 100.0
        );
    }
}
