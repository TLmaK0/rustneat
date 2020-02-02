#[cfg(test)]
mod test {
    use rustneat::{Environment, NeuralNetwork, Organism, Population};

    struct MyEnvironment;

    impl Environment<NeuralNetwork> for MyEnvironment {
        fn test(&self, _: &mut NeuralNetwork) -> f64 {
            0.1234f64
        }
    }

    struct XORClassification;

    impl Environment<NeuralNetwork> for XORClassification {
        fn test(&self, organism: &mut NeuralNetwork) -> f64 {
            let mut output = vec![0f64];
            let mut distance: f64;
            organism.activate(vec![0f64, 0f64], &mut output);
            distance = (0f64 - output[0]).abs();
            organism.activate(vec![0f64, 1f64], &mut output);
            distance += (1f64 - output[0]).abs();
            organism.activate(vec![1f64, 0f64], &mut output);
            distance += (1f64 - output[0]).abs();
            organism.activate(vec![1f64, 1f64], &mut output);
            distance += (0f64 - output[0]).abs();
            16.0 / (1.0 + distance)
        }
    }

    #[test]
    fn should_be_able_to_generate_a_population() {
        let population = Population::<NeuralNetwork>::create_population(150);
        assert!(population.size() == 150);
    }

    #[test]
    fn population_can_evolve() {
        let mut population = Population::<NeuralNetwork>::create_population(1);
        population.evolve();
        let genome = &population.get_organisms()[0].genome;
        assert_eq!(genome.total_genes(), 1);
        assert_ne!(genome.total_weights(), 0f64);
    }

    #[test]
    fn population_can_be_tested_on_environment() {
        let mut population = Population::<NeuralNetwork>::create_population(10);
        let mut environment = MyEnvironment;
        population.evaluate_in(&mut environment);
        assert!(population.get_organisms()[0].fitness == 0.1234f64);
    }
}
