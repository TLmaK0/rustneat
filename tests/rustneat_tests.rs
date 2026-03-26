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
        let environment = MyEnvironment;
        population.evaluate_in(&environment);
        assert!(population.get_organisms()[0].fitness == 0.1234f64);
    }

    struct XORClassification;

    impl Environment for XORClassification {
        fn test(&self, organism: &mut Organism) -> f64 {
            let mut output = vec![0f64];
            let mut distance: f64;
            organism.activate(vec![0f64, 0f64], &mut output);
            distance = (0f64 - output[0]).powi(2);
            organism.activate(vec![0f64, 1f64], &mut output);
            distance += (1f64 - output[0]).powi(2);
            organism.activate(vec![1f64, 0f64], &mut output);
            distance += (1f64 - output[0]).powi(2);
            organism.activate(vec![1f64, 1f64], &mut output);
            distance += (0f64 - output[0]).powi(2);

            16f64 / (1f64 + distance)
        }
    }

    #[test]
    fn network_should_be_able_to_solve_xor_classification() {
        let environment = XORClassification;

        for _attempt in 0..20 {
            let mut population = Population::create_population(150);
            let mut champion_option: Option<Organism> = None;

            for _gen in 0..1000 {
                population.evolve();
                population.evaluate_in(&environment);
                for organism in &population.get_organisms() {
                    if organism.fitness > 15.9f64 {
                        champion_option = Some(organism.clone());
                    }
                }
                if champion_option.is_some() {
                    break;
                }
            }

            if let Some(ref mut champion) = champion_option {
                let mut output = vec![0f64];
                champion.reset_state();
                champion.activate(vec![0f64, 0f64], &mut output);
                if output[0] >= 0.2f64 {
                    continue;
                }
                champion.activate(vec![0f64, 1f64], &mut output);
                if output[0] <= 0.8f64 {
                    continue;
                }
                champion.activate(vec![1f64, 0f64], &mut output);
                if output[0] <= 0.8f64 {
                    continue;
                }
                champion.activate(vec![1f64, 1f64], &mut output);
                if output[0] >= 0.2f64 {
                    continue;
                }
                return;
            }
        }
        panic!("Failed to solve XOR after 5 attempts of 500 generations each");
    }
}
