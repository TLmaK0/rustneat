use neat::connection_gene::ConnectionGene as ConnectionGene;

#[derive(Debug)]
pub struct Genome{
    connection_genes: Vec<ConnectionGene>,
    pub global_innovation: u32,
}

impl Genome{
    pub fn new() -> Genome {
        Genome {..Default::default()}
    }

    pub fn create_gene(&mut self) -> ConnectionGene {
        self.global_innovation += 1;
        let gene = ConnectionGene {
            innovation: self.global_innovation,
            ..Default::default()
        };
        self.connection_genes.push(gene);
        gene
    }
}

impl Default for Genome{
    fn default () -> Genome {
        Genome { global_innovation: 0,
                 connection_genes: vec![Default::default(); 0]}
    }
}
