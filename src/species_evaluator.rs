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

    /// Evaluate all organisms and return the champion
    pub fn evaluate(&self, species: &mut Vec<Specie>) -> Organism {
        if self.threads <= 1 {
            self.evaluate_single(species)
        } else {
            self.evaluate_parallel(species)
        }
    }

    fn evaluate_single(&self, species: &mut Vec<Specie>) -> Organism {
        let mut champion = Organism::new(Genome::default());

        for specie in species.iter_mut() {
            if !specie.organisms.is_empty() {
                self.environment.test_batch(&mut specie.organisms);
                for org in &specie.organisms {
                    if org.fitness > champion.fitness {
                        champion = org.clone();
                    }
                }
            }
        }

        champion
    }

    fn evaluate_parallel(&self, species: &mut Vec<Specie>) -> Organism {
        let original_sizes: Vec<usize> = species.iter().map(|s| s.organisms.len()).collect();

        let mut all_organisms: Vec<Organism> = species
            .iter_mut()
            .flat_map(|s| s.organisms.drain(..))
            .collect();

        if all_organisms.is_empty() {
            return Organism::new(Genome::default());
        }

        let chunk_size = (all_organisms.len() + self.threads - 1) / self.threads;
        let (tx, rx) = mpsc::channel();

        crossbeam::scope(|scope| {
            for chunk in all_organisms.chunks_mut(chunk_size) {
                let tx = tx.clone();
                let env = self.environment;

                scope.spawn(move |_| {
                    env.test_batch(chunk);

                    let mut local_champion = Organism::new(Genome::default());
                    for org in chunk.iter() {
                        if org.fitness > local_champion.fitness {
                            local_champion = org.clone();
                        }
                    }
                    tx.send(local_champion).unwrap();
                });
            }
        })
        .unwrap();

        let num_chunks = (all_organisms.len() + chunk_size - 1) / chunk_size;
        let mut champion = Organism::new(Genome::default());
        for _ in 0..num_chunks {
            let local_champion = rx.recv().unwrap();
            if local_champion.fitness > champion.fitness {
                champion = local_champion;
            }
        }

        let mut drain_iter = all_organisms.into_iter();
        for (specie, &size) in species.iter_mut().zip(original_sizes.iter()) {
            specie.organisms = drain_iter.by_ref().take(size).collect();
        }

        champion
    }
}
