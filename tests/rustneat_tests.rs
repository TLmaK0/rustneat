extern crate rustneat;

#[cfg(test)]
mod test {
    use rustneat::{Environment, Organism, Population};

    struct MyEnvironment;

    impl Environment for MyEnvironment {
        fn test(&self, _: &mut Organism) -> f64 {
            0.1234f64
        }
    }

    struct XORClassification;

    impl Environment for XORClassification {
        fn test(&self, organism: &mut Organism) -> f64 {
            let mut output = vec![0f64];
            let mut distance: f64;
            organism.activate(&vec![0f64, 0f64], &mut output);
            distance = (0f64 - output[0]).abs();
            organism.activate(&vec![0f64, 1f64], &mut output);
            distance += (1f64 - output[0]).abs();
            organism.activate(&vec![1f64, 0f64], &mut output);
            distance += (1f64 - output[0]).abs();
            organism.activate(&vec![1f64, 1f64], &mut output);
            distance += (0f64 - output[0]).abs();
            (4f64 - distance).powi(2)
        }
    }

    #[test]
    fn should_be_able_to_generate_a_population() {
        let population = Population::create_population(150);
        assert!(population.size() == 150);
    }

    #[test]
    fn population_can_evolve() {
        let mut population = Population::create_population(1);
        population.evolve();
        let genome = &population.get_organisms()[0].genome;
        assert_eq!(genome.total_genes(), 1);
        assert_ne!(genome.total_weights(), 0f64);
    }

    #[test]
    fn population_can_be_tested_on_environment() {
        let mut population = Population::create_population(10);
        let mut environment = MyEnvironment;
        population.evaluate_in(&mut environment);
        assert!(population.get_organisms()[0].fitness == 0.1234f64);
    }

    #[test]
    fn network_should_be_able_to_solve_xor_classification() {
        let mut population = Population::create_population(150);
        let mut environment = XORClassification;
        let mut champion_option: Option<Organism> = None;
        while champion_option.is_none() {
            population.evolve();
            population.evaluate_in(&mut environment);
            for organism in &population.get_organisms() {
                if organism.fitness > 15.9f64 {
                    champion_option = Some(organism.clone());
                }
            }
        }
        let champion = champion_option.as_mut().unwrap();
        let mut output = vec![0f64];
        champion.activate(&vec![0f64, 0f64], &mut output);
        assert!(output[0] < 0.1f64);
        champion.activate(&vec![0f64, 1f64], &mut output);
        assert!(output[0] > 0.9f64);
        champion.activate(&vec![1f64, 0f64], &mut output);
        assert!(output[0] > 0.9f64);
        champion.activate(&vec![1f64, 1f64], &mut output);
        assert!(output[0] < 0.1f64);
    }
}
