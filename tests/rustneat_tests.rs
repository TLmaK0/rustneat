
#[cfg(test)]
mod test {
    use rustneat::{Environment, Organism, Population, NeuralNetwork};

    struct X;

    impl Environment for X {
        fn test(&self, _: &mut NeuralNetwork) -> f64 {
            0.1234
        }
    }

    struct XORClassification;

    impl Environment for XORClassification {
        fn test(&self, organism: &mut NeuralNetwork) -> f64 {
            let mut output = vec![0.0];
            let mut distance: f64;
            organism.activate(vec![0.0, 0.0], &mut output);
            distance = (0.0 - output[0]).abs();
            organism.activate(vec![0.0, 1.0], &mut output);
            distance += (1.0 - output[0]).abs();
            organism.activate(vec![1.0, 0.0], &mut output);
            distance += (1.0 - output[0]).abs();
            organism.activate(vec![1.0, 1.0], &mut output);
            distance += (0.0 - output[0]).abs();
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
        let mut population = Population::create_population(1);
        population.evolve(&mut X);
        let genome = &population.get_organisms()[0].genome;
        assert_eq!(genome.connections.len(), 1);
        assert_ne!(genome.total_weights(), 0.0);
    }

    #[test]
    fn population_can_be_tested_on_environment() {
        let mut population = Population::create_population(10);
        population.evolve(&mut X);
        assert_eq!(population.get_organisms()[0].fitness, 0.1234);
    }

    #[test]
    fn network_can_solve_xor_classification() {
        const MAX_GENERATIONS: usize = 400;
        let mut population = Population::create_population(150);
        let mut environment = XORClassification;
        let mut champion: Option<Organism> = None;
        let mut i = 0;
        while champion.is_none() && i < MAX_GENERATIONS {
            population.evolve(&mut environment);
            for organism in &population.get_organisms() {
                if organism.fitness > 15.9 {
                    champion = Some(organism.clone());
                }
            }
            i += 1;
        }
        let Organism {genome: champion, fitness: _} = champion.as_mut().unwrap();
        let mut output = vec![0.0];
        champion.activate(vec![0.0, 0.0], &mut output);
        // println!("Output[0] = {}", output[0]);
        assert!(output[0] < 0.1);
        champion.activate(vec![0.0, 1.0], &mut output);
        assert!(output[0] > 0.9);
        champion.activate(vec![1.0, 0.0], &mut output);
        assert!(output[0] > 0.9);
        champion.activate(vec![1.0, 1.0], &mut output);
        assert!(output[0] < 0.1);
    }
}
