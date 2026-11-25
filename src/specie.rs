use crate::genome::Genome;
use crate::innovation::InnovationRegistry;
use crate::mutation_config::MutationConfig;
use crate::organism::Organism;
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

const BEST_ORGANISMS_THRESHOLD: f64 = 0.5f64; // Only top 50% can reproduce

/// Weight of an organism in the roulette: its adjusted fitness, or 0 if it
/// is not a positive finite number.
fn selection_weight(organism: &Organism) -> f64 {
    let weight = organism.adjusted_fitness;
    if weight.is_finite() && weight > 0.0 {
        weight
    } else {
        0.0
    }
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

    /// Check if another organism is of the same species as this one.
    pub fn match_genome(&self, organism: &Organism) -> bool {
        self.representative.is_same_specie(&organism.genome)
    }

    /// Check if another organism is of the same species using a custom threshold.
    pub fn match_genome_with_threshold(&self, organism: &Organism, threshold: f64) -> bool {
        self.representative
            .is_same_specie_with_threshold(&organism.genome, threshold)
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
    /// the parents of the other species (`Specie::parents`).
    pub fn generate_offspring(
        &mut self,
        num_of_organisms: usize,
        population_organisms: &[Organism],
    ) {
        self.generate_offspring_with_config(
            num_of_organisms,
            population_organisms,
            &MutationConfig::default(),
        );
    }

    /// Generate offspring with a specific mutation config (supports adaptive mutation)
    pub fn generate_offspring_with_config(
        &mut self,
        num_of_organisms: usize,
        population_organisms: &[Organism],
        base_config: &MutationConfig,
    ) {
        self.generate_offspring_with_config_and_rng(
            num_of_organisms,
            population_organisms,
            base_config,
            &mut rand::thread_rng(),
        );
    }

    /// Generate offspring with a specific mutation config, drawing every random
    /// decision (selection, crossover, mutation) from `rng`. New neurons are
    /// numbered by each genome on its own; `Population` uses
    /// `generate_offspring_with_innovations`.
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

    /// Like `generate_offspring_with_config_and_rng`, taking the ids of new
    /// neurons from `innovations`, the registry of the population.
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

    /// Organisms allowed to reproduce: the fittest half (at least one),
    /// the ones `generate_offspring` keeps as parents.
    pub fn parents(&self) -> impl Iterator<Item = &Organism> {
        let mut sorted: Vec<&Organism> = self.organisms.iter().collect();
        sorted.sort();
        sorted.truncate(self.parents_len());
        sorted.into_iter()
    }

    fn parents_len(&self) -> usize {
        ((self.organisms.len() as f64 * BEST_ORGANISMS_THRESHOLD) as usize).max(1)
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

        // Always copy champion (elitism)
        let copy_champion = if num_of_organisms > 1 { 1 } else { 0 };

        let organisms_to_mate = self.parents_len();
        self.organisms.sort();
        self.organisms.truncate(organisms_to_mate);

        let mut offspring: Vec<Organism> = {
            // Fitness-proportionate selection (roulette wheel) using adjusted_fitness

            // Calculate total adjusted fitness for roulette wheel
            let total_adjusted_fitness: f64 = self.organisms.iter().map(selection_weight).sum();

            let mut selected_organisms = vec![];
            for _ in 0..num_of_organisms - copy_champion {
                if total_adjusted_fitness <= 0.0 {
                    // Fallback to uniform selection if no positive fitness
                    selected_organisms.push(rng.gen_range(0..self.organisms.len()));
                } else {
                    // Roulette wheel selection
                    let spin = rng.gen_range(0.0..total_adjusted_fitness);
                    let mut cumulative = 0.0;
                    let mut selected = 0;
                    for (i, organism) in self.organisms.iter().enumerate() {
                        cumulative += selection_weight(organism);
                        if cumulative >= spin {
                            selected = i;
                            break;
                        }
                    }
                    selected_organisms.push(selected);
                }
            }
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
            // again.
            elite.preserve_fitness = elite.is_evaluated() && elite.fitness.is_finite();
            offspring.push(elite);
        }
        self.organisms = offspring;
    }

    /// Choice a new representative of the specie at random
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
        let mutate = |parent: &Organism, rng: &mut R| match innovations {
            Some(registry) => parent.mutate_with_innovations(config, registry, rng),
            None => parent.mutate_with_config_and_rng(config, rng),
        };
        if rng.gen::<f64>() < config.mutation_probability
            || self.organisms.len() + other_species_parents.len() < 2
        {
            // Mutation only (asexual reproduction)
            mutate(organism, rng)
        } else {
            // Crossover, then mutation
            let child = self.create_child_by_mate(organism, other_species_parents, config, rng);
            mutate(&child, rng)
        }
    }

    /// Crossover of `organism` with a parent of another species (probability
    /// `config.interspecies_mate_rate`, if there is another species) or of
    /// this one.
    fn create_child_by_mate<R: Rng + ?Sized>(
        &self,
        organism: &Organism,
        other_species_parents: &[&Organism],
        config: &MutationConfig,
        rng: &mut R,
    ) -> Organism {
        let mate = if !other_species_parents.is_empty()
            && rng.gen::<f64>() < config.interspecies_mate_rate
        {
            other_species_parents[rng.gen_range(0..other_species_parents.len())]
        } else {
            &self.organisms[rng.gen_range(0..self.organisms.len())]
        };
        organism.mate_with_config_and_rng(mate, config, rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::Genome;
    use crate::organism::Organism;
    use std::f64::EPSILON;

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

        assert!((specie.calculate_average_fitness() - 15f64).abs() < EPSILON);
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
            specie.generate_offspring_with_config_and_rng(
                4,
                &organisms,
                &MutationConfig::default(),
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
        let genome = Genome::new_initialized(2, 1);
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

    /// Only crossover, no mutation, the given interspecies rate.
    fn crossover_only(interspecies: f64) -> MutationConfig {
        MutationConfig::new()
            .mutation_probability(0.0)
            .add_connection_rate(0.0)
            .add_neuron_rate(0.0)
            .weight_mutation_rate(0.0)
            .bias_mutation_rate(0.0)
            .toggle_expression_rate(0.0)
            .interspecies_mate_rate(interspecies)
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
            specie.generate_offspring_with_config_and_rng(
                1000,
                &others,
                &crossover_only(rate),
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
        specie.generate_offspring_with_config_and_rng(50, &[], &crossover_only(1.0), &mut rng);
        assert_eq!(specie.organisms.len(), 50);
        assert_eq!(children_with_foreign_parent(&specie), 0);
    }

    #[test]
    fn parents_are_the_fittest_half() {
        let (specie, _) = two_species();
        let mut fitnesses: Vec<f64> = specie.parents().map(|o| o.fitness).collect();
        fitnesses.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(fitnesses, vec![6.0, 7.0, 8.0, 9.0, 10.0]);
        let mut single = Specie::new(Genome::default());
        single.add(Organism::new(Genome::default()));
        assert_eq!(single.parents().count(), 1);
    }

    /// #19: generating offspring from a species without organisms panicked
    /// (`gen_range` on an empty range). Without parents there are no
    /// children: the species stays empty.
    #[test]
    fn generate_offspring_of_an_empty_species_does_not_panic() {
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
        let config = MutationConfig::default();
        let mates = vec![Organism::new(Genome::new_initialized(2, 1))];
        for size in [0, 1, 2, 10] {
            let mut specie = Specie::new(Genome::new_initialized(2, 1));
            assert!(specie.is_empty());
            specie.generate_offspring_with_config_and_rng(size, &mates, &config, &mut rng);
            assert!(specie.is_empty(), "offspring size {}", size);
            specie.generate_offspring_with_config_and_rng(size, &[], &config, &mut rng);
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
        assert!((specie.organisms[0].adjusted_fitness - 10.0).abs() < EPSILON);
        assert!((specie.organisms[1].adjusted_fitness - 20.0).abs() < EPSILON);
        assert!((specie.organisms[2].adjusted_fitness - 30.0).abs() < EPSILON);
    }
}
