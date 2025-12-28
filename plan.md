# Plan: Lunar Lander with NEAT

## Objective
Demonstrate that RustNEAT can solve complex control problems through evolutionary RL, using the OpenAI Gym Lunar Lander environment.

## Status: WORKING - Multiprocessing ~5.3 gen/s

### Current Architecture
```
openai_lunar_lander.rs
├── PyOrganism - PyO3 wrapper for Organism (neural network)
├── create_organism(genes, neurons_len) - Reconstructs genome from serialized genes
└── LunarLanderMultiprocess::test_batch(organisms)
    └── pool.starmap(worker, batch_data) - Parallel evaluation

lunar_lander_worker.py
├── init_worker() - Creates persistent gym environment per worker
└── evaluate_organism(genes, render) - Runs episode using rustneat_py
```

### Configuration
- Population: 150 organisms
- max_steps: 1000 per episode
- Workers: 24 (2x CPUs with pool initializer)
- Target fitness: 300.0

## Completed: API Improvements

### Environment Trait Changes
```rust
pub trait Environment: Sync {
    fn test(&self, organism: &mut Organism) -> f64 { ... }
    fn test_batch(&self, organisms: &mut [Organism]) { ... }
    fn threads(&self) -> usize { num_cpus::get() }
}
```
- `test()` now optional with default implementation
- `test_batch()` receives all organisms for batch evaluation
- Both methods use mutable references (avoids cloning overhead)

### Genome Reconstruction
Added `Genome::from_genes(genes, last_neuron_id)` to reconstruct genomes from serialized data in Python workers.

## Completed: Performance Optimization

### Results
| Version | Performance | Workers | Speedup |
|---------|-------------|---------|---------|
| Single-threaded (Rust CTRNN) | 2.4 gen/s | 1 | 1x |
| Multiprocessing (pool.starmap) | 5.3 gen/s | 24 | 2.2x |

### Analysis
- Best observed single-sample fitness: 278.84 (near target 300)
- Verified that genome reconstruction works correctly
- Verification tests show proper variance (not all identical values)

## Completed: Hyperparameter Tuner

### New Modules
- `MutationConfig` - Configurable mutation rates per problem
- `SearchSpace` - Define parameter ranges for optimization
- `HyperTuner` - Random search hyperparameter optimization

### Usage
```rust
let search_space = SearchSpace::new()
    .add_connection_rate(0.01..=0.10)
    .add_neuron_rate(0.01..=0.05)
    .weight_mutation_rate(0.70..=0.95);

let tuner = HyperTuner::new(search_space)
    .population_size(150)
    .input_neurons(8)
    .output_neurons(4)
    .generations_per_trial(100)
    .num_trials(15)
    .early_stop_fitness(300.0);

let result = tuner.optimize(&environment);
```

### Run Tuner
```bash
cargo build --release --example lunar_lander_tuner --features openai
./target/release/examples/lunar_lander_tuner
# → Saves best_config.json
```

### Config Persistence
- `lunar_lander_tuner` saves best config to `best_config.json`
- `openai_lunar_lander` loads config from `best_config.json` if exists
- Default config included in repo

## Completed: NEAT Selection Algorithm Fix

### Problem Identified
El algoritmo de selección tenía bugs críticos:
- **Bug en `get_best_species()`**: Retornaba la 2da peor especie en lugar de las 2 mejores
- **Elitismo débil**: Solo copiaba campeón si especie tenía >5 organismos

### Cambios Implementados
1. ✓ Fix `get_best_species()` - ordenar descendente y retornar `[0..2]`
2. ✓ Elitismo mejorado - copiar campeón siempre que `num_organisms > 1`

### Experimentos Realizados
| Configuración | Éxito (10s) | Notas |
|--------------|-------------|-------|
| Original (bugs) | ~45% | Seleccionaba peores especies |
| Fixes 1-2 | 93% | **Mejor resultado** |
| + Threshold 25% | ~17% | Pérdida de diversidad |
| + Ruleta | 73% | Convergencia prematura |
| + Threshold 25% + Ruleta | ~17% | Peor combinación |

### Conclusiones
- La selección proporcional al fitness requiere **fitness sharing** (dividir por tamaño de especie)
- Para XOR, la exploración (diversidad) es más importante que la explotación (selección)
- **Configuración óptima actual**: Todos se reproducen + selección proporcional (ruleta) + fitness sharing + elitismo

### Referencias
- [NEAT Paper](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
- [NEAT-Python](https://neat-python.readthedocs.io/en/latest/neat_overview.html)

## Completed: Fitness Sharing

### Implementación
- `Organism.adjusted_fitness` - nuevo campo para fitness ajustado
- `Specie.adjust_fitness()` - divide fitness por tamaño de especie
- Llamado en `Population.evaluate_in()` después de evaluar todos los organismos
- Selección proporcional (ruleta) usando `adjusted_fitness`

### Resultados XOR (10s timeout)
| Configuración | Éxito | Notas |
|--------------|-------|-------|
| Uniforme + sin sharing | ~93% | Baseline anterior |
| Ruleta + sin sharing | ~73% | Convergencia prematura |
| Ruleta + fitness sharing | ~80% | Mejora con sharing |

### Conclusión
Fitness sharing mejora la selección proporcional pero la selección uniforme sigue siendo competitiva para XOR.
Para problemas más complejos (Lunar Lander), la selección proporcional + sharing debería escalar mejor.

## Completed: Crossover Disabled Gene Rule

### Problem
El crossover no seguía la regla de NEAT para genes deshabilitados:
- Si un gen está **disabled** en alguno de los padres, hay **75% de probabilidad** de que el hijo lo tenga disabled.

### Implementación
Modificado `Genome::mate_genes()` en `src/genome.rs`:
```rust
// NEAT rule: if gene is disabled in either parent, 75% chance offspring has it disabled
if !gene.enabled() || other_gene_disabled {
    if rand::random::<f64>() < 0.75 {
        child_gene.set_disabled();
    } else {
        child_gene.set_enabled();
    }
}
```

### Test
Agregado test estadístico que verifica la proporción ~75% disabled en 1000 trials.

### Resultados XOR
- 47/50 éxitos (94%) con timeout 10s
- Consistente con resultados anteriores (~93%)

## Next Steps

### Options to Explore
1. **Longer training runs** - Continue evolution to reach 300 target
2. **Native parallelism** - Rust-native physics simulator (no Python)
3. **GPU acceleration** - Vectorized CTRNN on GPU

## Setup

```bash
cargo build --release --example openai_lunar_lander --features openai
./target/release/examples/openai_lunar_lander
```

## Notes
- Branch: `feature/openai_lunar_lander`
- Target fitness: 300.0
