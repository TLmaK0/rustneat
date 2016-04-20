extern crate rustneat;

#[cfg(test)]
mod test{
    use rustneat::neat::*;

    struct MyEnvironment;

    impl Environment for MyEnvironment{
        fn test(&self, _: &mut Organism) -> f64 {
            0.1234f64
        }
    }

    struct XORClassification;

    impl Environment for XORClassification{
        fn test(&self, organism: &mut Organism) -> f64 {
            let mut output = vec![0f64];
            let mut distance: f64;

            organism.activate(&vec![0f64,0f64], &mut output); 
            distance = (output[0]).abs();
            organism.activate(&vec![0f64,1f64], &mut output); 
            distance += (1f64 - output[0]).abs();
            organism.activate(&vec![1f64,0f64], &mut output); 
            distance += (1f64 - output[0]).abs();
            organism.activate(&vec![1f64,1f64], &mut output); 
            distance += (output[0]).abs();
            (4f64 - distance).powi(2)
        }
    }

    #[test]
    fn should_be_able_to_generate_a_population(){
        let population = Population::create_population(150);
        assert!(population.size() == 150);
    }

    #[test]
    fn population_can_evolve(){
        let mut population = Population::create_population(1);
        population.evolve();
        let genome = &population.organisms[0].genome;
        assert!(genome.total_genes() == 1);
        assert!(genome.total_weights() != 0f64);
    }

    #[test]
    fn population_can_be_tested_on_environment(){
        let mut population = Population::create_population(10);
        let environment = MyEnvironment;
        population.evaluate_in(&environment);
        assert!(population.organisms[0].fitness == 0.1234f64);
    }

    #[test]
    fn network_should_be_able_to_solve_xor_classification(){
        let mut population = Population::create_population(100);
        let environment = XORClassification;
        let mut found = false;
        let mut champion: Option<Organism> = None;
        let mut partial_champion: Option<Organism> = None;
        let mut generation = 0;
        let mut actual_fitness = 0f64;
        let mut max_neurons = 0;
        while !found {
            population.evolve();
            population.evaluate_in(&environment);
            for organism in &population.organisms {
                if organism.fitness > actual_fitness {
                    partial_champion = Some(organism.clone());
                    actual_fitness = organism.fitness;
                }

                if organism.genome.len() > max_neurons {
                    max_neurons = organism.genome.len();
                }

                if organism.fitness > 15.9f64 {
                    champion = Some(organism.clone());
                    found = true;
                }
            }
            println!("Generation: {:?}, fitness: {:?}, neurons: {:?}", generation, actual_fitness, max_neurons);
            generation += 1;
            if generation == 500 {
                found = true;
            }
        }
        assert!(champion.is_some(), "Not able to solve XOR classification");
    }
}
