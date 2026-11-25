use crate::environment::Environment;
use crate::genome::Genome;
use crate::organism::Organism;
use crate::specie::Specie;
use std::sync::mpsc;

/// Evaluates all organisms across species using the environment
pub struct SpeciesEvaluator<'a> {
    threads: usize,
    environment: &'a dyn Environment,
}

impl<'a> SpeciesEvaluator<'a> {
    /// Create a new evaluator with the given environment.
    pub fn new(environment: &'a dyn Environment) -> SpeciesEvaluator<'a> {
        SpeciesEvaluator {
            threads: environment.threads(),
            environment,
        }
    }

    /// Evaluate all organisms and return the champion: the organism with
    /// the highest fitness, whatever its sign (ties: the first one; NaN
    /// fitness is ignored). If there is no organism, or all of them got
    /// NaN, it returns an empty organism with fitness `-inf`.
    pub fn evaluate(&self, species: &mut [Specie]) -> Organism {
        let champion = if self.threads <= 1 {
            self.evaluate_single(species)
        } else {
            self.evaluate_parallel(species)
        };
        champion.unwrap_or_else(|| {
            let mut empty = Organism::new(Genome::default());
            empty.fitness = f64::NEG_INFINITY;
            empty
        })
    }

    fn evaluate_single(&self, species: &mut [Specie]) -> Option<Organism> {
        let mut champion: Option<Organism> = None;

        for specie in species.iter_mut() {
            if !specie.organisms.is_empty() {
                test_batch(self.environment, &mut specie.organisms);
                champion =
                    Organism::fittest(champion.iter().chain(specie.organisms.iter())).cloned();
            }
        }

        champion
    }

    fn evaluate_parallel(&self, species: &mut [Specie]) -> Option<Organism> {
        let original_sizes: Vec<usize> = species.iter().map(|s| s.organisms.len()).collect();

        let mut all_organisms: Vec<Organism> = species
            .iter_mut()
            .flat_map(|s| s.organisms.drain(..))
            .collect();

        if all_organisms.is_empty() {
            return None;
        }

        let chunk_size = all_organisms.len().div_ceil(self.threads);
        let (tx, rx) = mpsc::channel();

        crossbeam::scope(|scope| {
            for (chunk_index, chunk) in all_organisms.chunks_mut(chunk_size).enumerate() {
                let tx = tx.clone();
                let env = self.environment;

                scope.spawn(move |_| {
                    test_batch(env, chunk);
                    let local_champion = Organism::fittest(chunk.iter()).cloned();
                    tx.send((chunk_index, local_champion)).unwrap();
                });
            }
        })
        .unwrap();

        // Chunks finish in any order; reduce them in chunk order so ties are
        // broken exactly as in `evaluate_single` (first organism wins) and the
        // result does not depend on thread scheduling.
        drop(tx);
        let mut local_champions: Vec<(usize, Option<Organism>)> = rx.iter().collect();
        local_champions.sort_by_key(|(chunk_index, _)| *chunk_index);
        let champion =
            Organism::fittest(local_champions.iter().filter_map(|(_, c)| c.as_ref())).cloned();

        let mut drain_iter = all_organisms.into_iter();
        for (specie, &size) in species.iter_mut().zip(original_sizes.iter()) {
            specie.organisms = drain_iter.by_ref().take(size).collect();
        }

        champion
    }
}

/// Let the environment evaluate `organisms` and mark all of them as
/// evaluated (elite copies it skipped already were).
fn test_batch(environment: &dyn Environment, organisms: &mut [Organism]) {
    environment.test_batch(organisms);
    for organism in organisms.iter_mut() {
        organism.mark_evaluated();
    }
}
