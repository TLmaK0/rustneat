use crossbeam::{self, Scope};
use environment::Environment;
use genome::Genome;
use num_cpus;
use organism::Organism;
use specie::Specie;
use std::sync::mpsc;
use std::sync::mpsc::{Receiver, Sender};

/// Calculate fitness and champions for a species
pub struct SpeciesEvaluator<'a> {
    threads: usize,
    environment: &'a mut Environment,
}

impl<'a> SpeciesEvaluator<'a> {
    /// Take an environment that will test organisms.
    pub fn new(environment: &mut Environment) -> SpeciesEvaluator {
        SpeciesEvaluator {
            threads: num_cpus::get(),
            environment: environment,
        }
    }

    /// return champion fitness
    pub fn evaluate(&self, species: &mut Vec<Specie>) -> Organism {
        let mut champion: Organism = Organism::new(Genome::default());

        for specie in species {
            if specie.organisms.is_empty() {
                continue;
            }

            let organisms_by_thread = (specie.organisms.len() + self.threads - 1) / self.threads; // round up
            let (tx, rx): (Sender<Organism>, Receiver<Organism>) = mpsc::channel();
            crossbeam::scope(|scope| {
                let threads_used = self.dispatch_organisms(
                    specie.organisms.as_mut_slice(),
                    organisms_by_thread,
                    0,
                    &tx,
                    scope,
                );
                for _ in 0..threads_used {
                    let champion_candidate = rx.recv().unwrap();
                    if champion_candidate.fitness > champion.fitness {
                        champion = champion_candidate;
                    }
                }
            });
        }
        champion
    }

    fn dispatch_organisms<'b>(
        &'b self,
        organisms: &'b mut [Organism],
        organisms_by_thread: usize,
        threads_used: usize,
        tx: &Sender<Organism>,
        scope: &Scope<'b>,
    ) -> usize {
        if organisms.len() <= organisms_by_thread {
            self.evaluate_organisms(organisms, tx.clone(), scope);
        } else {
            match organisms.split_at_mut(organisms_by_thread) {
                (thread_organisms, remaining_organisms) => {
                    self.evaluate_organisms(thread_organisms, tx.clone(), scope);
                    if remaining_organisms.len() > 0 {
                        return self.dispatch_organisms(
                            remaining_organisms,
                            organisms_by_thread,
                            threads_used + 1,
                            tx,
                            scope,
                        );
                    }
                }
            }
        }
        threads_used + 1
    }

    fn evaluate_organisms<'b>(
        &'b self,
        organisms: &'b mut [Organism],
        tx: Sender<Organism>,
        scope: &Scope<'b>,
    ) {
        scope.spawn(move || {
            let mut champion = Organism::new(Genome::default());
            for organism in &mut organisms.iter_mut() {
                organism.fitness = self.environment.test(organism);
                if organism.fitness > champion.fitness {
                    champion = organism.clone();
                }
            }
            tx.send(champion).unwrap();
        });
    }
}
