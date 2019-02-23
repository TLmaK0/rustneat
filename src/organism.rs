// TODO: tests ...
/*
#[cfg(test)]
use gene::Gene;

#[cfg(test)]
mod tests {
    use super::*;
    use genome::Genome;

    #[test]
    fn should_propagate_signal_without_hidden_layers() {
        let mut organism = Organism::new(Genome::default());
        organism.genome.add_gene(Gene::new(0, 1, 5f64, true, false));
        let sensors = vec![7.5];
        let mut output = vec![0f64];
        organism.activate(sensors, &mut output);
        assert!(
            output[0] > 0.9f64,
            format!("{:?} is not bigger than 0.9", output[0])
        );

        let mut organism = Organism::new(Genome::default());
        organism.genome.add_gene(Gene::new(0, 1, -2f64, true, false));
        let sensors = vec![1f64];
        let mut output = vec![0f64];
        organism.activate(sensors, &mut output);
        assert!(
            output[0] < 0.1f64,
            format!("{:?} is not smaller than 0.1", output[0])
        );
    }

    #[test]
    fn should_propagate_signal_over_hidden_layers() {
        let mut organism = Organism::new(Genome::default());
        organism.genome.add_gene(Gene::new(0, 1, 0f64, true, false));
        organism.genome.add_gene(Gene::new(0, 2, 5f64, true, false));
        organism.genome.add_gene(Gene::new(2, 1, 5f64, true, false));
        let sensors = vec![0f64];
        let mut output = vec![0f64];
        organism.activate(sensors, &mut output);
        assert!(
            output[0] > 0.9f64,
            format!("{:?} is not bigger than 0.9", output[0])
        );
    }

    #[test]
    fn should_work_with_cyclic_networks() {
        let mut organism = Organism::new(Genome::default());
        organism.genome.add_gene(Gene::new(0, 1, 2f64, true, false));
        organism.genome.add_gene(Gene::new(1, 2, 2f64, true, false));
        organism.genome.add_gene(Gene::new(2, 1, 2f64, true, false));
        let mut output = vec![0f64];
        organism.activate(vec![1f64], &mut output);
        assert!(
            output[0] > 0.9,
            format!("{:?} is not bigger than 0.9", output[0])
        );

        let mut organism = Organism::new(Genome::default());
        organism.genome.add_gene(Gene::new(0, 1, -2f64, true, false));
        organism.genome.add_gene(Gene::new(1, 2, -2f64, true, false));
        organism.genome.add_gene(Gene::new(2, 1, -2f64, true, false));
        let mut output = vec![0f64];
        organism.activate(vec![1f64], &mut output);
        assert!(
            output[0] < 0.1,
            format!("{:?} is not smaller than 0.1", output[0])
        );
    }

    #[test]
    fn activate_organims_sensor_without_enough_neurons_should_ignore_it() {
        let mut organism = Organism::new(Genome::default());
        organism.genome.add_gene(Gene::new(0, 1, 1f64, true, false));
        let sensors = vec![0f64, 0f64, 0f64];
        let mut output = vec![0f64];
        organism.activate(sensors, &mut output);
    }

    #[test]
    fn should_allow_multiple_output() {
        let mut organism = Organism::new(Genome::default());
        organism.genome.add_gene(Gene::new(0, 1, 1f64, true, false));
        let sensors = vec![0f64];
        let mut output = vec![0f64, 0f64];
        organism.activate(sensors, &mut output);
    }

    #[test]
    fn should_be_able_to_get_matrix_representation_of_the_neuron_connections() {
        let mut organism = Organism::new(Genome::default());
        organism.genome.add_gene(Gene::new(0, 1, 1f64, true, false));
        organism.genome.add_gene(Gene::new(1, 2, 0.5f64, true, false));
        organism.genome.add_gene(Gene::new(2, 1, 0.5f64, true, false));
        organism.genome.add_gene(Gene::new(2, 2, 0.75f64, true, false));
        organism.genome.add_gene(Gene::new(1, 0, 1f64, true, false));
        assert_eq!(
            organism.get_weights(),
            vec![0.0, 1.0, 0.0, 1.0, 0.0, 0.5, 0.0, 0.5, 0.75]
        );
    }

    #[test]
    fn should_not_raise_exception_if_less_neurons_than_required() {
        let mut organism = Organism::new(Genome::default());
        organism.genome.add_gene(Gene::new(0, 1, 1f64, true, false));
        let sensors = vec![0f64, 0f64, 0f64];
        let mut output = vec![0f64, 0f64, 0f64];
        organism.activate(sensors, &mut output);
    }
}
*/
