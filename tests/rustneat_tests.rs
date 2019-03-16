
#[cfg(test)]
mod test {
    use rustneat::{Environment, Organism, Population, NeuralNetwork, Params};

    struct X;

    impl Environment for X {
        fn test(&self, _: &mut NeuralNetwork) -> f64 {
            0.1234
        }
    }

    struct XORClassification;

    impl Environment for XORClassification {
        fn test(&self, organism: &mut NeuralNetwork) -> f64 {
            let target_inputs = vec![vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0]];
            let target_outputs = vec![0.0, 1.0, 1.0, 0.0];
            let mut output = vec![0.0];
            let mut distance: f64 = 0.0;
            for (i, o) in target_inputs.iter().zip(target_outputs) {
                organism.activate(i.clone(), &mut output);
                distance += (o - output[0]).powi(2);

            }
            16.0 / (1.0 + distance)
        }
    }

    #[test]
    fn can_generate_a_population() {
        let population = Population::<NeuralNetwork>::create_population(150);
        assert!(population.size() == 150);
    }

    #[test]
    fn population_can_evolve() {
        let p = Params {
            mutation_pr: 1.0, // because mutation ensures we have connections
            ..Default::default()
        };
        let mut population = Population::create_population(2);
        population.evolve(&mut X, &p);
        let genome = &population.get_organisms().next().unwrap().genome;
        assert_eq!(genome.connections.len(), 1);
        assert_ne!(genome.total_weights(), 0.0);
    }

    #[test]
    fn population_can_be_tested_on_environment() {
        let mut population = Population::create_population(10);
        population.evolve(&mut X, &Params::default());
        assert_eq!(population.get_organisms().next().unwrap().fitness, 0.1234);
    }

    #[test]
    fn can_solve_xor() {
        const MAX_GENERATIONS: usize = 800;
        let p = Params::default();
        let start_genome = NeuralNetwork::with_neurons(3);
        let mut population = Population::create_population_from(start_genome, 150);
        let mut environment = XORClassification;
        let mut champion: Option<Organism> = None;
        let mut i = 0;
        while champion.is_none() && i < MAX_GENERATIONS {
            population.evolve(&mut environment, &p);
            for organism in population.get_organisms() {
                if organism.fitness > 15.8 {
                    champion = Some(organism.clone());
                }
            }
            i += 1;
        }
        let Organism {genome: champion, fitness: _} = champion.as_mut().unwrap();
        println!("Solved in {} generations", i);


        let mut output = vec![0.0];
        champion.activate(vec![0.0, 0.0], &mut output);
        assert!(output[0].abs() < 0.1);
        champion.activate(vec![0.0, 1.0], &mut output);
        assert!(output[0] > 0.9);
        champion.activate(vec![1.0, 0.0], &mut output);
        assert!(output[0] > 0.9);
        champion.activate(vec![1.0, 1.0], &mut output);
        assert!(output[0] < 0.1);
    }

    #[test]
    fn xor_can_only_improve() {
        const MAX_GENERATIONS: usize = 200;
        let p = Params::default();
        let mut population = Population::create_population(150);
        let mut environment = XORClassification;
        let mut best_fitness = std::f64::MIN;
        for _ in 0..MAX_GENERATIONS {
            population.evolve(&mut environment, &p);

            let mut best_fitness_in_gen = std::f64::MIN;
            for organism in population.get_organisms() {
                if organism.fitness > best_fitness_in_gen {
                    best_fitness_in_gen = organism.fitness;
                }
            }

            // println!("{} > {}?", champion.fitness, best_fitness);
            assert!(best_fitness_in_gen >= best_fitness);
            if best_fitness_in_gen > best_fitness {
                best_fitness = best_fitness_in_gen;
            }
        }
    }
}
