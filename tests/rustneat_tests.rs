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
        let population = Population::create_population_initialized(150, 2, 1);
        assert!(population.size() == 150);
    }

    #[test]
    fn population_can_evolve() {
        let mut population = Population::create_population_initialized(1, 2, 1);
        population.evolve();
        let genome = &population.get_organisms()[0].genome;
        assert!(genome.total_genes() >= 2);
        assert_ne!(genome.total_weights(), 0f64);
    }

    #[test]
    fn population_can_be_tested_on_environment() {
        let mut population = Population::create_population_initialized(10, 2, 1);
        let environment = MyEnvironment;
        population.evaluate_in(&environment).unwrap();
        assert!(population.get_organisms()[0].fitness == 0.1234f64);
    }

    const XOR_PATTERNS: [([f64; 2], f64); 4] = [
        ([0f64, 0f64], 0f64),
        ([0f64, 1f64], 1f64),
        ([1f64, 0f64], 1f64),
        ([1f64, 1f64], 0f64),
    ];

    /// XOR as four independent cases: the CTRNN state is reset before each
    /// pattern, so the answer to one cannot depend on the ones before it.
    struct XORClassification;

    impl Environment for XORClassification {
        fn test(&self, organism: &mut Organism) -> f64 {
            let mut output = vec![0f64];
            let mut distance = 0f64;
            for (inputs, expected) in XOR_PATTERNS.iter() {
                organism.reset_state();
                organism.activate(inputs.to_vec(), &mut output);
                distance += (expected - output[0]).powi(2);
            }
            16f64 / (1f64 + distance)
        }
    }

    /// All the orders in which the four XOR patterns can be presented.
    fn all_orders() -> Vec<Vec<usize>> {
        fn permute(prefix: Vec<usize>, rest: Vec<usize>, out: &mut Vec<Vec<usize>>) {
            if rest.is_empty() {
                out.push(prefix);
                return;
            }
            for i in 0..rest.len() {
                let mut prefix = prefix.clone();
                let mut rest = rest.clone();
                prefix.push(rest.remove(i));
                permute(prefix, rest, out);
            }
        }
        let mut orders = vec![];
        permute(vec![], (0..XOR_PATTERNS.len()).collect(), &mut orders);
        orders
    }

    /// True if `organism` classifies every XOR pattern (below 0.2 for 0,
    /// above 0.8 for 1) in every order, resetting its state before each one.
    fn solves_xor_in_every_order(organism: &Organism) -> bool {
        all_orders().iter().all(|order| {
            let mut organism = organism.clone();
            order.iter().all(|&k| {
                let (inputs, expected) = XOR_PATTERNS[k];
                let mut output = vec![0f64];
                organism.reset_state();
                organism.activate(inputs.to_vec(), &mut output);
                if expected > 0.5 {
                    output[0] > 0.8
                } else {
                    output[0] < 0.2
                }
            })
        })
    }

    #[test]
    fn there_are_24_orders_of_the_xor_patterns() {
        let orders = all_orders();
        assert_eq!(orders.len(), 24);
        let mut unique = orders.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), 24);
    }

    /// Seed of the XOR test: 1, or the value of the environment variable
    /// `RUSTNEAT_XOR_SEED` (to check that the test passes with other seeds).
    fn xor_seed() -> u64 {
        match std::env::var("RUSTNEAT_XOR_SEED") {
            Ok(seed) => seed.parse().expect("RUSTNEAT_XOR_SEED must be a u64"),
            Err(_) => 1,
        }
    }
    const XOR_GENERATIONS: usize = 500;
    const XOR_TARGET: f64 = 15.5;

    #[test]
    fn network_should_be_able_to_solve_xor_classification() {
        let seed = xor_seed();
        let environment = XORClassification;
        let mut population = Population::create_population_initialized(150, 2, 1).with_seed(seed);

        let mut champion: Option<Organism> = None;
        let mut generations = 0;
        while generations < XOR_GENERATIONS && champion.is_none() {
            population.evolve();
            population.evaluate_in(&environment).unwrap();
            generations += 1;
            champion = population
                .get_organisms()
                .into_iter()
                .find(|organism| organism.fitness > XOR_TARGET);
        }

        let champion = champion.unwrap_or_else(|| {
            panic!(
                "no organism reached fitness {} in 1 attempt of {} generations (seed {})",
                XOR_TARGET, generations, seed
            )
        });
        assert!(
            solves_xor_in_every_order(&champion),
            "the champion of generation {} (seed {}) reached fitness {} but does not \
             solve XOR in every order of the patterns with its state reset before each one: {:?}",
            generations,
            seed,
            champion.fitness,
            champion.genome
        );
    }
}
