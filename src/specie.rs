use crate::genome::Genome;
use crate::innovation::InnovationRegistry;
use crate::mutation_config::{MateSelection, MutationConfig};
use crate::organism::Organism;
use crate::population::shift_to_non_negative;
use conv::prelude::*;
use rand::seq::SliceRandom;
use rand::Rng;

/// A species (several organisms) and associated fitnesses
#[derive(Debug, Clone)]
pub struct Specie {
    representative: Genome,
    /// Best fitness any organism of this species ever had (-inf before the
    /// first evaluation).
    best_fitness: f64,
    /// Generations this species has produced offspring.
    age: usize,
    /// `age` when `best_fitness` last improved.
    age_last_improvement: usize,
    /// All orgnamisms in this species
    pub organisms: Vec<Organism>,
    /// Allows to set an id to identify it
    pub id: i64,
}

impl Specie {
    /// Create a new species from a Genome
    pub fn new(genome: Genome) -> Specie {
        Specie {
            organisms: vec![],
            representative: genome,
            best_fitness: f64::NEG_INFINITY,
            age: 0,
            age_last_improvement: 0,
            id: 0,
        }
    }

    /// Add an Organism
    pub fn add(&mut self, organism: Organism) {
        self.organisms.push(organism);
    }

    /// Check if another organism is of the same species as this one, with
    /// the default coefficients and threshold (those of
    /// `MutationConfig::default()`), not those of the population.
    #[deprecated(note = "ignores the population's config: use `Specie::match_genome_with_config`")]
    pub fn match_genome(&self, organism: &Organism) -> bool {
        self.match_genome_with_config(organism, &MutationConfig::default())
    }

    /// Check if another organism is of the same species using a custom
    /// threshold and the default coefficients.
    #[deprecated(
        note = "ignores the population's coefficients: use `Specie::match_genome_with_config`"
    )]
    pub fn match_genome_with_threshold(&self, organism: &Organism, threshold: f64) -> bool {
        self.distance_with_config(organism, &MutationConfig::default()) < threshold
    }

    /// Check if another organism is of the same species with the
    /// compatibility coefficients and threshold of `config`.
    pub fn match_genome_with_config(&self, organism: &Organism, config: &MutationConfig) -> bool {
        self.representative
            .is_same_specie_with_config(&organism.genome, config)
    }

    /// Compatibility distance between the representative of this species
    /// and `organism`, with the coefficients of `config`.
    pub fn distance_with_config(&self, organism: &Organism, config: &MutationConfig) -> f64 {
        self.representative
            .compatibility_distance_with_config(&organism.genome, config)
    }

    /// Fitness of the most performant organism, whatever its sign (NaN
    /// is ignored; `-inf` if there is none).
    pub fn calculate_champion_fitness(&self) -> f64 {
        Organism::fittest(&self.organisms).map_or(f64::NEG_INFINITY, |organism| organism.fitness)
    }

    /// Update stagnation tracking. Call once after every evaluation: the
    /// species improves only when its champion beats the best fitness the
    /// species ever had. Returns true if it improved.
    pub fn update_stagnation(&mut self) -> bool {
        let current_best = self
            .organisms
            .iter()
            .map(|organism| organism.fitness)
            .fold(f64::NEG_INFINITY, f64::max);
        if current_best > self.best_fitness {
            self.best_fitness = current_best;
            self.age_last_improvement = self.age;
            true
        } else {
            false
        }
    }

    /// Best fitness the species ever had, as seen by `update_stagnation`.
    pub fn best_fitness(&self) -> f64 {
        self.best_fitness
    }

    /// Check if species has stagnated: `max_generations` or more
    /// generations without improving its best fitness
    pub fn is_stagnant(&self, max_generations: usize) -> bool {
        self.generations_without_improvement() >= max_generations
    }

    /// Get generations since last improvement
    pub fn generations_without_improvement(&self) -> usize {
        self.age.saturating_sub(self.age_last_improvement)
    }

    /// Work out average fitness of this species
    pub fn calculate_average_fitness(&self) -> f64 {
        let organisms_count = self.organisms.len().value_as::<f64>().unwrap();
        if organisms_count == 0f64 {
            return 0f64;
        }

        let total_fitness = self
            .organisms
            .iter()
            .fold(0f64, |total, organism| total + organism.fitness);

        total_fitness / organisms_count
    }

    /// Mate and generate offspring, delete old organisms and use the children
    /// as "new" species. `population_organisms` are the candidates for an
    /// interspecies crossover (see `MutationConfig::interspecies_mate_rate`):
    /// the parents of the other species (`Specie::parents_with_config`).
    /// Uses the default `MutationConfig`, an entropy-seeded generator
    /// (`rand::thread_rng`) and no innovation registry.
    #[deprecated(
        note = "use `Specie::generate_offspring_with_innovations` with the population's config, registry and generator"
    )]
    pub fn generate_offspring(
        &mut self,
        num_of_organisms: usize,
        population_organisms: &[Organism],
    ) {
        let mates: Vec<&Organism> = population_organisms.iter().collect();
        self.generate_offspring_with_mates(
            num_of_organisms,
            &mates,
            &MutationConfig::default(),
            None,
            &mut rand::thread_rng(),
        );
    }

    /// Generate offspring with a specific mutation config, an entropy-seeded
    /// generator (`rand::thread_rng`) and no innovation registry.
    #[deprecated(
        note = "use `Specie::generate_offspring_with_innovations` with the population's registry and generator"
    )]
    pub fn generate_offspring_with_config(
        &mut self,
        num_of_organisms: usize,
        population_organisms: &[Organism],
        base_config: &MutationConfig,
    ) {
        let mates: Vec<&Organism> = population_organisms.iter().collect();
        self.generate_offspring_with_mates(
            num_of_organisms,
            &mates,
            base_config,
            None,
            &mut rand::thread_rng(),
        );
    }

    /// Generate offspring with a specific mutation config, drawing every random
    /// decision (selection, crossover, mutation) from `rng`, without
    /// innovation registry: each genome numbers its new neurons on its own,
    /// so the same split in two children may get different ids and
    /// different splits the same id. The random draws are those of
    /// `generate_offspring_with_innovations`, which `Population` uses.
    #[deprecated(
        note = "neuron ids are per genome: use `Specie::generate_offspring_with_innovations` with the population's registry (`Population::innovations`)"
    )]
    pub fn generate_offspring_with_config_and_rng<R: Rng + ?Sized>(
        &mut self,
        num_of_organisms: usize,
        population_organisms: &[Organism],
        base_config: &MutationConfig,
        rng: &mut R,
    ) {
        let mates: Vec<&Organism> = population_organisms.iter().collect();
        self.generate_offspring_with_mates(num_of_organisms, &mates, base_config, None, rng);
    }

    /// Mate and generate offspring with the mutation config `base_config`,
    /// drawing every random decision (selection, crossover, mutation) from
    /// `rng` and taking the ids of new neurons from `innovations`, the
    /// registry of the population. The old organisms are replaced by the
    /// children. `population_organisms` are the candidates for an
    /// interspecies crossover (see `MutationConfig::interspecies_mate_rate`):
    /// the parents of the other species (`Specie::parents_with_config`).
    pub fn generate_offspring_with_innovations<R: Rng + ?Sized>(
        &mut self,
        num_of_organisms: usize,
        population_organisms: &[Organism],
        base_config: &MutationConfig,
        innovations: &mut InnovationRegistry,
        rng: &mut R,
    ) {
        let mates: Vec<&Organism> = population_organisms.iter().collect();
        self.generate_offspring_with_mates(
            num_of_organisms,
            &mates,
            base_config,
            Some(innovations),
            rng,
        );
    }

    /// Organisms allowed to reproduce with the default configuration: the
    /// fittest `survival_threshold` (at least one).
    #[deprecated(
        note = "ignores the population's `survival_threshold`: use `Specie::parents_with_config`"
    )]
    pub fn parents(&self) -> impl Iterator<Item = &Organism> {
        self.parents_with_config(&MutationConfig::default())
    }

    /// Organisms allowed to reproduce: the fittest
    /// `config.survival_threshold` of the species (at least one).
    pub fn parents_with_config(&self, config: &MutationConfig) -> impl Iterator<Item = &Organism> {
        let mut sorted: Vec<&Organism> = self.organisms.iter().collect();
        sorted.sort();
        sorted.truncate(self.parents_len(config));
        sorted.into_iter()
    }

    fn parents_len(&self, config: &MutationConfig) -> usize {
        ((self.organisms.len() as f64 * config.survival_threshold) as usize).max(1)
    }

    /// Generates the offspring; `other_species_parents` are the candidates
    /// for an interspecies crossover. A species without organisms has no
    /// parents, so it gets no offspring and stays empty (#19).
    pub(crate) fn generate_offspring_with_mates<R: Rng + ?Sized>(
        &mut self,
        num_of_organisms: usize,
        other_species_parents: &[&Organism],
        base_config: &MutationConfig,
        mut innovations: Option<&mut InnovationRegistry>,
        rng: &mut R,
    ) {
        if self.organisms.is_empty() {
            return;
        }
        self.age += 1;

        // Elitism: a species with enough offspring keeps its champion.
        let copy_champion = if num_of_organisms >= base_config.champion_copy_min_offspring.max(1) {
            1
        } else {
            0
        };

        let organisms_to_mate = self.parents_len(base_config);
        self.organisms.sort();
        self.organisms.truncate(organisms_to_mate);

        let mut offspring: Vec<Organism> = {
            let selected_organisms = self.select_parents(num_of_organisms - copy_champion, rng);
            selected_organisms
                .iter()
                .map(|organism_pos| {
                    self.create_child(
                        &self.organisms[*organism_pos],
                        other_species_parents,
                        base_config,
                        innovations.as_deref_mut(),
                        rng,
                    )
                })
                .collect::<Vec<Organism>>()
        };

        if copy_champion == 1 {
            let mut elite = Organism::fittest(&self.organisms)
                .unwrap_or(&self.organisms[0])
                .clone();
            // Keep the fitness of the elite copy (skip re-evaluation) only if
            // it is a real one: an organism that was never evaluated (as in
            // the first `evolve`) or got a non finite fitness is evaluated
            // again. With noisy fitness (`reevaluate_elites`) elites are
            // always evaluated again.
            elite.preserve_fitness =
                !base_config.reevaluate_elites && elite.is_evaluated() && elite.fitness.is_finite();
            offspring.push(elite);
        }
        self.organisms = offspring;
    }

    /// Indices of `count` organisms drawn by fitness-proportionate
    /// selection (roulette wheel) on their adjusted fitness, shifted like
    /// the offspring shares between species: if any is negative, the worst
    /// gets weight 0 and the rest their distance to it, so the fittest is
    /// favoured whatever the sign; non-negative fitness is used as is. Non
    /// finite fitness gets weight 0.
    fn select_parents<R: Rng + ?Sized>(&self, count: usize, rng: &mut R) -> Vec<usize> {
        let adjusted: Vec<f64> = self
            .organisms
            .iter()
            .map(|organism| organism.adjusted_fitness)
            .collect();
        let weights = shift_to_non_negative(&adjusted);
        let total: f64 = weights.iter().sum();
        (0..count)
            .map(|_| {
                if total <= 0.0 {
                    // Uniform selection if no organism has a positive weight
                    rng.gen_range(0..weights.len())
                } else {
                    let spin = rng.gen_range(0.0..total);
                    let mut cumulative = 0.0;
                    weights
                        .iter()
                        .position(|weight| {
                            cumulative += weight;
                            cumulative >= spin
                        })
                        .unwrap_or(0)
                }
            })
            .collect()
    }

    /// Choice a new representative of the specie at random, from an
    /// entropy-seeded generator (`rand::thread_rng`).
    #[deprecated(note = "not seedable: use `Specie::choose_new_representative_with_rng`")]
    pub fn choose_new_representative(&mut self) {
        self.choose_new_representative_with_rng(&mut rand::thread_rng());
    }

    /// Choice a new representative of the specie at random from `rng`
    pub fn choose_new_representative_with_rng<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        self.representative = self.organisms.choose(rng).unwrap().genome.clone();
    }

    /// Get a genome representitive of this species.
    pub fn get_representative_genome(&self) -> Genome {
        self.representative.clone()
    }

    /// Clear existing organisms in this species.
    pub fn remove_organisms(&mut self) {
        self.organisms = vec![];
    }

    /// Returns true if specie hasn't organisms
    pub fn is_empty(&self) -> bool {
        self.organisms.is_empty()
    }

    /// Apply fitness sharing: divide each organism's fitness by species size
    /// This prevents large species from dominating the population
    pub fn adjust_fitness(&mut self) {
        let species_size = self.organisms.len() as f64;
        if species_size == 0.0 {
            return;
        }
        for organism in &mut self.organisms {
            organism.adjusted_fitness = organism.fitness / species_size;
        }
    }

    /// Create a new child by mutation only (probability
    /// `config.mutation_probability`, 0.4 by default; the NEAT paper uses
    /// 0.25) or by crossover and then mutation.
    fn create_child<R: Rng + ?Sized>(
        &self,
        organism: &Organism,
        other_species_parents: &[&Organism],
        config: &MutationConfig,
        innovations: Option<&mut InnovationRegistry>,
        rng: &mut R,
    ) -> Organism {
        if rng.gen::<f64>() < config.mutation_probability
            || self.organisms.len() + other_species_parents.len() < 2
        {
            // Mutation only (asexual reproduction)
            organism.mutate_with_registry(config, innovations, rng)
        } else {
            // Crossover, then mutation unless `mate_only_probability` skips
            // it (never for a crossover with itself). No draw when it is 0,
            // so the random sequence is the same as before it existed.
            let (child, with_itself) =
                self.create_child_by_mate(organism, other_species_parents, config, rng);
            if !with_itself
                && config.mate_only_probability > 0f64
                && rng.gen::<f64>() < config.mate_only_probability
            {
                child
            } else {
                child.mutate_with_registry(config, innovations, rng)
            }
        }
    }

    /// Crossover of `organism` with a parent of another species (probability
    /// `config.interspecies_mate_rate`, if there is another species) or of
    /// this one, chosen as `config.mate_selection` says. Also returns
    /// whether the mate is `organism` itself.
    fn create_child_by_mate<R: Rng + ?Sized>(
        &self,
        organism: &Organism,
        other_species_parents: &[&Organism],
        config: &MutationConfig,
        rng: &mut R,
    ) -> (Organism, bool) {
        let mate = if !other_species_parents.is_empty()
            && rng.gen::<f64>() < config.interspecies_mate_rate
        {
            other_species_parents[rng.gen_range(0..other_species_parents.len())]
        } else {
            let index = match config.mate_selection {
                MateSelection::Uniform => rng.gen_range(0..self.organisms.len()),
                MateSelection::FitnessProportional => self.select_parents(1, rng)[0],
            };
            &self.organisms[index]
        };
        (
            organism.mate_with_config_and_rng(mate, config, rng),
            std::ptr::eq(organism, mate),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::Genome;
    use crate::organism::Organism;

    #[test]
    fn specie_should_return_correct_average_fitness() {
        let mut specie = Specie::new(Genome::default());
        let mut organism1 = Organism::new(Genome::default());
        organism1.fitness = 10f64;

        let mut organism2 = Organism::new(Genome::default());
        organism2.fitness = 15f64;

        let mut organism3 = Organism::new(Genome::default());
        organism3.fitness = 20f64;

        specie.add(organism1);
        specie.add(organism2);
        specie.add(organism3);

        assert!((specie.calculate_average_fitness() - 15f64).abs() < f64::EPSILON);
    }

    fn set_fitnesses(specie: &mut Specie, fitnesses: &[f64]) {
        for (organism, fitness) in specie.organisms.iter_mut().zip(fitnesses) {
            organism.fitness = *fitness;
        }
    }

    /// The average is only a getter: a rising average without a better
    /// champion is not an improvement, so the species stagnates.
    #[test]
    fn stagnation_follows_the_best_fitness_not_the_average() {
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
        let genome = Genome::new_initialized_with_rng(2, 1, &mut rng);
        let mut specie = Specie::new(genome.clone());
        for _ in 0..4 {
            specie.add(Organism::new(genome.clone()));
        }
        set_fitnesses(&mut specie, &[10.0, 0.0, 0.0, 0.0]);
        assert!(specie.update_stagnation());
        assert_eq!(specie.best_fitness(), 10.0);

        for generation in 1..=16 {
            let organisms = specie.organisms.clone();
            specie.generate_offspring_with_innovations(
                4,
                &organisms,
                &MutationConfig::default(),
                &mut InnovationRegistry::default(),
                &mut rng,
            );
            // The average rises every generation, the best stays below 10.
            let fitness = (generation as f64 * 0.5).min(9.0);
            set_fitnesses(&mut specie, &[fitness; 4]);
            specie.calculate_average_fitness();
            specie.calculate_average_fitness();
            assert!(!specie.update_stagnation());
            assert_eq!(specie.generations_without_improvement(), generation);
        }
        assert!(specie.is_stagnant(15));
        assert_eq!(specie.best_fitness(), 10.0);

        // A better champion resets the counter.
        set_fitnesses(&mut specie, &[11.0, 0.0, 0.0, 0.0]);
        assert!(specie.update_stagnation());
        assert_eq!(specie.generations_without_improvement(), 0);
        assert!(!specie.is_stagnant(15));
    }

    /// Negative fitnesses are an improvement over "never evaluated".
    #[test]
    fn first_evaluation_is_an_improvement_even_with_negative_fitness() {
        let mut specie = Specie::new(Genome::default());
        let mut organism = Organism::new(Genome::default());
        organism.fitness = -5.0;
        specie.add(organism);
        assert!(specie.update_stagnation());
        assert_eq!(specie.best_fitness(), -5.0);
    }

    /// A species of 10 organisms with genes `0 -> 2`, `1 -> 2` (fitness
    /// 1..=10) and, from another species, 4 fitter organisms that also have
    /// `0 -> 3`, `3 -> 2`: a child with neuron 3 has a parent of the other
    /// species.
    fn two_species() -> (Specie, Vec<Organism>) {
        let genome = Genome::new_initialized_with_rng(2, 1, &mut rand::thread_rng());
        let mut specie = Specie::new(genome.clone());
        for fitness in 1..=10 {
            let mut organism = Organism::new(genome.clone());
            organism.fitness = fitness as f64;
            specie.add(organism);
        }
        let mut other = genome;
        other.add_gene(crate::Gene::new(0, 3, 1.0, true));
        other.add_gene(crate::Gene::new(3, 2, 1.0, true));
        let others = (0..4)
            .map(|_| {
                let mut organism = Organism::new(other.clone());
                organism.fitness = 100.0;
                organism
            })
            .collect();
        (specie, others)
    }

    /// Only crossover, no mutation, the given interspecies rate; the
    /// fittest half reproduces.
    fn crossover_only(interspecies: f64) -> MutationConfig {
        MutationConfig::builder()
            .mutation_probability(0.0)
            .add_connection_rate(0.0)
            .add_neuron_rate(0.0)
            .weight_mutation_rate(0.0)
            .bias_mutation_rate(0.0)
            .toggle_expression_rate(0.0)
            .interspecies_mate_rate(interspecies)
            .survival_threshold(0.5)
            .build()
    }

    fn children_with_foreign_parent(specie: &Specie) -> usize {
        specie
            .organisms
            .iter()
            .filter(|organism| organism.genome.has_neuron(3))
            .count()
    }

    #[test]
    fn default_interspecies_mate_rate_is_the_neat_paper_one() {
        assert_eq!(MutationConfig::default().interspecies_mate_rate, 0.001);
    }

    #[test]
    fn interspecies_mating_follows_the_configured_rate() {
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
        for (rate, low, high) in [(0.0, 0, 0), (1.0, 999, 999), (0.3, 250, 350)] {
            let (mut specie, others) = two_species();
            specie.generate_offspring_with_innovations(
                1000,
                &others,
                &crossover_only(rate),
                &mut InnovationRegistry::default(),
                &mut rng,
            );
            // 999 children and the copy of the champion.
            let foreign = children_with_foreign_parent(&specie);
            assert!(
                foreign >= low && foreign <= high,
                "rate {}: {} children with a parent of another species",
                rate,
                foreign
            );
        }
    }

    #[test]
    fn interspecies_mating_without_other_species_mates_inside() {
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
        let (mut specie, _) = two_species();
        specie.generate_offspring_with_innovations(
            50,
            &[],
            &crossover_only(1.0),
            &mut InnovationRegistry::default(),
            &mut rng,
        );
        assert_eq!(specie.organisms.len(), 50);
        assert_eq!(children_with_foreign_parent(&specie), 0);
    }

    /// A species of 10 organisms with the same genome and fitness 1..=10:
    /// a crossover child of two of them is a copy of that genome unless it
    /// is mutated.
    fn clones_specie() -> (Specie, Genome) {
        let genome = Genome::new_initialized_with_rng(2, 1, &mut rand::thread_rng());
        let mut specie = Specie::new(genome.clone());
        for fitness in 1..=10 {
            let mut organism = Organism::new(genome.clone());
            organism.fitness = fitness as f64;
            specie.add(organism);
        }
        (specie, genome)
    }

    /// Every child goes through crossover and every mutation changes all
    /// the weights, unless `mate_only` skips it.
    fn always_mate_config(mate_only: f64) -> MutationConfig {
        MutationConfig::builder()
            .mutation_probability(0.0)
            .add_connection_rate(0.0)
            .add_neuron_rate(0.0)
            .weight_mutation_rate(1.0)
            .weight_gene_mutation_rate(1.0)
            .interspecies_mate_rate(0.0)
            .mate_only_probability(mate_only)
            .survival_threshold(0.5)
            .build()
    }

    fn unchanged_children(specie: &Specie, genome: &Genome) -> usize {
        specie
            .organisms
            .iter()
            .filter(|organism| weights(&organism.genome) == weights(genome))
            .count()
    }

    /// Connections and their weights (`Gene` equality ignores the weight).
    fn weights(genome: &Genome) -> Vec<(usize, usize, f64)> {
        genome
            .get_genes()
            .iter()
            .map(|g| (g.in_neuron_id(), g.out_neuron_id(), g.weight()))
            .collect()
    }

    #[test]
    fn default_mate_only_probability_mutates_every_crossover_child() {
        assert_eq!(MutationConfig::default().mate_only_probability, 0.0);
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
        let (mut specie, genome) = clones_specie();
        specie.generate_offspring_with_innovations(
            1000,
            &[],
            &always_mate_config(0.0),
            &mut InnovationRegistry::default(),
            &mut rng,
        );
        // Only the copy of the champion keeps the genome.
        assert_eq!(unchanged_children(&specie, &genome), 1);
    }

    /// #38 point 3 / NEAT `mate_only_prob`: a crossover child is left
    /// unmutated with this probability, except when both parents are the
    /// same organism (the child would be a plain copy).
    #[test]
    fn mate_only_probability_skips_the_mutation_of_crossover_children() {
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
        for (mate_only, low, high) in [(1.0, 700, 900), (0.5, 330, 480)] {
            let (mut specie, genome) = clones_specie();
            specie.generate_offspring_with_innovations(
                1000,
                &[],
                &always_mate_config(mate_only),
                &mut InnovationRegistry::default(),
                &mut rng,
            );
            // 5 parents: a fifth of the crossovers are with the same
            // organism and are mutated anyway.
            let unchanged = unchanged_children(&specie, &genome);
            assert!(
                unchanged >= low && unchanged <= high,
                "mate_only {}: {} unchanged children",
                mate_only,
                unchanged
            );
        }
    }

    /// Children (crossover only, no mutation) of a species whose parents
    /// are four organisms of fitness 10 with weight 1 and one of fitness 1
    /// with weight 7, that carry the weight 7.
    fn children_with_weak_weight(selection: MateSelection) -> usize {
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
        let with_weight = |weight: f64| {
            let genes = vec![
                crate::Gene::new(0, 2, weight, true),
                crate::Gene::new(1, 2, weight, true),
            ];
            Genome::from_genes_with_io(genes, 2, 2, 1)
        };
        let mut specie = Specie::new(with_weight(1.0));
        for fitness in [10.0, 10.0, 10.0, 10.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5] {
            let weight = if fitness == 1.0 { 7.0 } else { 1.0 };
            let mut organism = Organism::new(with_weight(weight));
            organism.fitness = fitness;
            specie.add(organism);
        }
        specie.adjust_fitness();
        let mut config = crossover_only(0.0);
        config.crossover_weight_averaging_rate = 0.0;
        config.mate_selection = selection;
        specie.generate_offspring_with_innovations(
            1000,
            &[],
            &config,
            &mut InnovationRegistry::default(),
            &mut rng,
        );
        specie
            .organisms
            .iter()
            .filter(|o| o.genome.get_genes().iter().any(|g| g.weight() == 7.0))
            .count()
    }

    #[test]
    fn default_mate_selection_is_uniform_over_the_parents() {
        assert_eq!(
            MutationConfig::default().mate_selection,
            MateSelection::Uniform
        );
        // The weak parent is the mate of about a fifth of the crossovers.
        let weak = children_with_weak_weight(MateSelection::Uniform);
        assert!(weak > 70, "{} children with the weak weight", weak);
    }

    /// #98: the mate can also be drawn by fitness, like the first parent.
    #[test]
    fn fitness_proportional_mate_selection_favours_the_fittest() {
        // The weak parent has 1/41 of the fitness.
        let weak = children_with_weak_weight(MateSelection::FitnessProportional);
        assert!(weak < 50, "{} children with the weak weight", weak);
    }

    #[test]
    fn parents_are_the_fittest_half_with_threshold_one_half() {
        let (specie, _) = two_species();
        let half = MutationConfig::builder().survival_threshold(0.5).build();
        let mut fitnesses: Vec<f64> = specie
            .parents_with_config(&half)
            .map(|o| o.fitness)
            .collect();
        fitnesses.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(fitnesses, vec![6.0, 7.0, 8.0, 9.0, 10.0]);
        let mut single = Specie::new(Genome::default());
        single.add(Organism::new(Genome::default()));
        assert_eq!(
            single
                .parents_with_config(&MutationConfig::default())
                .count(),
            1
        );
    }

    /// #19: generating offspring from a species without organisms panicked
    /// (`gen_range` on an empty range). Without parents there are no
    /// children: the species stays empty.
    #[test]
    fn generate_offspring_of_an_empty_species_does_not_panic() {
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
        let config = MutationConfig::default();
        let mates = vec![Organism::new(Genome::new_initialized_with_rng(
            2,
            1,
            &mut rand::thread_rng(),
        ))];
        for size in [0, 1, 2, 10] {
            let mut specie = Specie::new(Genome::new_initialized_with_rng(
                2,
                1,
                &mut rand::thread_rng(),
            ));
            assert!(specie.is_empty());
            specie.generate_offspring_with_innovations(
                size,
                &mates,
                &config,
                &mut InnovationRegistry::default(),
                &mut rng,
            );
            assert!(specie.is_empty(), "offspring size {}", size);
            specie.generate_offspring_with_innovations(
                size,
                &[],
                &config,
                &mut InnovationRegistry::default(),
                &mut rng,
            );
            assert!(specie.is_empty(), "offspring size {} without mates", size);
        }
    }

    #[test]
    fn adjust_fitness_should_divide_by_species_size() {
        let mut specie = Specie::new(Genome::default());

        let mut organism1 = Organism::new(Genome::default());
        organism1.fitness = 30.0;

        let mut organism2 = Organism::new(Genome::default());
        organism2.fitness = 60.0;

        let mut organism3 = Organism::new(Genome::default());
        organism3.fitness = 90.0;

        specie.add(organism1);
        specie.add(organism2);
        specie.add(organism3);

        specie.adjust_fitness();

        // Each fitness should be divided by species size (3)
        assert!((specie.organisms[0].adjusted_fitness - 10.0).abs() < f64::EPSILON);
        assert!((specie.organisms[1].adjusted_fitness - 20.0).abs() < f64::EPSILON);
        assert!((specie.organisms[2].adjusted_fitness - 30.0).abs() < f64::EPSILON);
    }

    /// Species with the given fitnesses (fitness sharing already applied)
    /// and how many times each organism is picked as a parent in `draws`
    /// roulette spins with a fixed seed.
    fn parent_counts(fitnesses: &[f64], draws: usize) -> Vec<usize> {
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
        let genome = Genome::new_initialized_with_rng(2, 1, &mut rng);
        let mut specie = Specie::new(genome.clone());
        for _ in fitnesses {
            specie.add(Organism::new(genome.clone()));
        }
        set_fitnesses(&mut specie, fitnesses);
        specie.adjust_fitness();
        let mut counts = vec![0; fitnesses.len()];
        for index in specie.select_parents(draws, &mut rng) {
            counts[index] += 1;
        }
        counts
    }

    /// Criterion of #98: with every fitness in [-100, -90] the fittest
    /// organism is picked more often than the worst (it used to be uniform).
    #[test]
    fn selection_favours_the_fittest_when_all_fitness_is_negative() {
        let fitnesses = [-90.0, -92.0, -94.0, -96.0, -98.0, -100.0];
        let draws = 6000;
        let counts = parent_counts(&fitnesses, draws);
        let (fittest, worst) = (counts[0], counts[fitnesses.len() - 1]);
        assert!(fittest > worst, "counts {:?}", counts);
        // Shifted weights 10, 8, 6, 4, 2, 0: the fittest gets a third of the
        // draws, well above the uniform share (1/6).
        assert!(fittest > draws / 4, "counts {:?}", counts);
        assert_eq!(worst, 0, "counts {:?}", counts);
    }

    /// With mixed signs a negative organism above the worst still gets a
    /// share (it used to get weight 0, like the worst).
    #[test]
    fn selection_ranks_negative_fitness_with_mixed_signs() {
        let counts = parent_counts(&[5.0, -1.0, -5.0], 6000);
        assert!(counts[0] > counts[1], "counts {:?}", counts);
        assert!(counts[1] > counts[2], "counts {:?}", counts);
    }

    /// Non-negative fitness keeps its plain proportions (no shift).
    #[test]
    fn selection_keeps_proportions_of_non_negative_fitness() {
        let counts = parent_counts(&[3.0, 1.0, 0.0], 8000);
        assert_eq!(counts[2], 0, "counts {:?}", counts);
        let ratio = counts[0] as f64 / counts[1] as f64;
        assert!((ratio - 3.0).abs() < 0.3, "counts {:?}", counts);
    }
}
