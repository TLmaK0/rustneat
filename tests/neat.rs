extern crate rustneat;

#[cfg(test)]
mod test{
    use rustneat::neat::*;

    struct MyEnvironment;

    impl Environment for MyEnvironment{
        fn test(&self, _: &Organism) -> f64 {
            0.1234f64
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
    fn pupulation_can_be_tested_on_environment(){
        let mut population = Population::create_population(1);
        let environment = MyEnvironment;
        population.evaluate_in(&environment);
        assert!(population.organisms[0].fitness == 0.1234f64);
    }
}
