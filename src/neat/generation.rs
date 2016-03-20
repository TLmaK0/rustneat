use neat::connection_gene::ConnectionGene as ConnectionGene;
use std::collections::HashMap;

#[derive(Debug)]
pub struct Generation {
    global_innovations: HashMap<u32, (u32, u32)>,
    last_innovation_key: u32
}

impl Generation {
    pub fn new() -> Generation {
        Generation { global_innovations: HashMap::new(), last_innovation_key: 0u32 }
    }

    pub fn get_innovation_id(&mut self) -> u32 {
        self.last_innovation_key += 1;
        self.last_innovation_key
    }

    pub fn get_innovation_ids_by_gen(&mut self, gen: ConnectionGene) -> (u32, u32) {
        match self.global_innovations.get(&gen.innovation).cloned(){
            Some(innovation_pair) => innovation_pair,
            None =>{
                let innovations = (self.get_innovation_id(), self.get_innovation_id());
                self.global_innovations.insert(gen.innovation, innovations);
                innovations
            }
        }
    }
}
