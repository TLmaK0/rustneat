#[cfg(test)]
mod test {
    use rustneat::{Environment, NeatParams, NeuralNetwork, Organism, Population};

    struct X;

    impl Environment for X {
        fn test(&self, _: &mut NeuralNetwork) -> f64 {
            0.1234
        }
    }

    struct XORClassification;

    impl Environment for XORClassification {
        fn test(&self, organism: &mut NeuralNetwork) -> f64 {
            let nn = organism.make_network();
            let target_inputs = vec![
                vec![0.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 0.0],
                vec![1.0, 1.0],
            ];
            let target_outputs = vec![0.0, 1.0, 1.0, 0.0];
            let mut output = vec![0.0];
            let mut distance: f64 = 0.0;
            for (i, o) in target_inputs.iter().zip(target_outputs) {
                nn.activate(i.clone(), &mut output);
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
        let p = NeatParams {
            mutation_pr: 1.0, // because mutation ensures we have connections
            mutate_del_conn_pr: 0.0,
            ..NeatParams::default(1, 1)
        };
        let mut population = Population::create_population(2);
        population.evolve(&mut X, &p,true);
        let genome = &population.get_organisms().next().unwrap().genome;
        assert_eq!(genome.connections.len(), 1);
    }

    #[test]
    fn population_can_be_tested_on_environment() {
        let mut population = Population::create_population(10);
        population.evolve(&mut X, &NeatParams::default(0, 0),true);
        assert_eq!(population.get_organisms().next().unwrap().fitness, 0.1234);
    }

    #[test]
    fn can_solve_xor() {
        const MAX_GENERATIONS: usize = 1500;
        let p = NeatParams::optimized_for_xor3(2, 1);
        let start_genome = NeuralNetwork::with_neurons(3);
        let mut population = Population::create_population_from(start_genome, 150);
        let mut environment = XORClassification;
        let mut champion: Option<Organism> = None;
        let mut i = 0;
        while champion.is_none() && i < MAX_GENERATIONS {
            population.evolve(&mut environment, &p,true);
            for organism in population.get_organisms() {
                // Test whether there is any organism that solves the problem
                let mut output = vec![0.0; 4];
                let nn = organism.genome.make_network();
                nn.activate(vec![0.0, 0.0], &mut output[0..1]);
                nn.activate(vec![0.0, 1.0], &mut output[1..2]);
                nn.activate(vec![1.0, 0.0], &mut output[2..3]);
                nn.activate(vec![1.0, 1.0], &mut output[3..]);
                if output[0].abs() < 0.1
                    && (1.0 - output[1]).abs() < 0.1
                    && (1.0 - output[2]).abs() < 0.1
                    && output[3].abs() < 0.1
                {
                    champion = Some(organism.clone());
                }
            }
            i += 1;
        }
        println!("Solved in {} generations", i);
    }

    #[test]
    fn xor_can_only_improve() {
        const MAX_GENERATIONS: usize = 200;
        let p = NeatParams::default(2, 1);
        let mut population = Population::create_population(150);
        let mut environment = XORClassification;
        let mut best_fitness = std::f64::MIN;
        for _ in 0..MAX_GENERATIONS {
            population.evolve(&mut environment, &p,true);

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
