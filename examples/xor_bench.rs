//! XOR benchmark: runs XOR many times and prints one `SUMMARY` line so every
//! change can be measured against the same yardstick.
//!
//! usage: `xor_bench <runs> <max_gens> <mode> [key=value ...]`
//!
//! modes:
//! * `default` - `create_population(pop)`, as `examples/simple_sample.rs`
//! * `init`    - `create_population_initialized_with_config(pop, 2, 1, cfg)`
//! * `initb`   - like `init` with 3 inputs (x1, x2, constant 1.0)
//! * `uncon`   - `create_population_unconnected_with_config(pop, 2, 1, cfg)`
//!
//! keys: `addc`, `addn`, `wpow`, `winit`, `wrate`, `thr`, `mutp`, `tog`,
//! `bias`, `tau` (MutationConfig), `pop` (population size, default 150),
//! `reset` (0/1: reset the CTRNN state before each pattern, default 0),
//! `target` (fitness that counts as solved, default 15.5 of 16),
//! `seed` (run `i` uses `Population::with_seed(seed + i)`, so every run is
//! reproducible and independent; without it each run seeds from entropy).
//!
//! The environment variable `BENCH_THREADS` sets how many runs go in
//! parallel (default 6).
//!
//! Reference ("bench A"): `xor_bench 50 500 init reset=1`.
extern crate rustneat;
use rustneat::{Environment, MutationConfig, Organism, Population};
use std::sync::{Arc, Mutex};
use std::time::Instant;

const PATTERNS: [(f64, f64, f64); 4] = [(0., 0., 0.), (0., 1., 1.), (1., 0., 1.), (1., 1., 0.)];
const USAGE: &str = "usage: xor_bench <runs> <max_gens> <default|init|initb|uncon> [key=value ...]";

#[derive(Clone, Copy)]
struct Xor {
    bias_input: bool,
    reset: bool,
}

impl Xor {
    fn outputs(&self, o: &mut Organism, order: &[usize], reset_each: bool) -> [f64; 4] {
        let mut res = [0f64; 4];
        for &k in order {
            let (a, b, _) = PATTERNS[k];
            if reset_each {
                o.reset_state();
            }
            let mut out = vec![0f64];
            let input = if self.bias_input {
                vec![a, b, 1.0]
            } else {
                vec![a, b]
            };
            o.activate(input, &mut out);
            res[k] = out[0];
        }
        res
    }

    fn fit(res: &[f64; 4]) -> f64 {
        let d: f64 = PATTERNS
            .iter()
            .zip(res.iter())
            .map(|(p, r)| (p.2 - r).powi(2))
            .sum();
        16.0 / (1.0 + d)
    }
}

impl Environment for Xor {
    fn test(&self, o: &mut Organism) -> f64 {
        // Same as examples/simple_sample.rs (no reset) unless reset=1.
        let r = self.outputs(o, &[0, 1, 2, 3], self.reset);
        Xor::fit(&r)
    }

    fn threads(&self) -> usize {
        1
    }
}

#[derive(Debug, Clone, Default)]
struct RunResult {
    solved_gen: Option<usize>,
    pop_min: usize,
    pop_max: usize,
    pop_end: usize,
    species_max: usize,
    species_end: usize,
    neurons: usize,
    genes: usize,
    enabled: usize,
    max_abs_w: f64,
    fit_reset_each: f64,
    fit_reversed: f64,
    fit_same: f64,
    champ_regress: usize,
    best_fit: f64,
    secs: f64,
}

struct Bench {
    runs: usize,
    max_gens: usize,
    mode: String,
    pop_size: usize,
    target: f64,
    seed: Option<u64>,
    cfg: MutationConfig,
    env: Xor,
    extra: Vec<String>,
}

fn parse_args(args: &[String]) -> Result<Bench, String> {
    if args.len() < 4 {
        return Err(USAGE.to_string());
    }
    let runs = args[1]
        .parse()
        .map_err(|_| format!("bad <runs>: {}\n{}", args[1], USAGE))?;
    let max_gens = args[2]
        .parse()
        .map_err(|_| format!("bad <max_gens>: {}\n{}", args[2], USAGE))?;
    let mode = args[3].clone();
    if !["default", "init", "initb", "uncon"].contains(&mode.as_str()) {
        return Err(format!("unknown mode {}\n{}", mode, USAGE));
    }
    let mut b = MutationConfig::new();
    let mut pop_size = 150usize;
    let mut reset = false;
    let mut target = 15.5f64;
    let mut seed = None;
    for kv in &args[4..] {
        let (k, v) = kv
            .split_once('=')
            .ok_or_else(|| format!("expected key=value, got {}", kv))?;
        if k == "seed" {
            // Parsed as an integer: going through f64 would round big seeds.
            seed = Some(
                v.parse::<u64>()
                    .map_err(|_| format!("bad value for seed: {}", v))?,
            );
            continue;
        }
        let v: f64 = v
            .parse()
            .map_err(|_| format!("bad value for {}: {}", k, v))?;
        b = match k {
            "addc" => b.add_connection_rate(v),
            "addn" => b.add_neuron_rate(v),
            "wpow" => b.weight_mutate_power(v),
            "winit" => b.weight_init_range(v),
            "wrate" => b.weight_mutation_rate(v),
            "thr" => b.compatibility_threshold(v),
            "mutp" => b.mutation_probability(v),
            "tog" => b.toggle_expression_rate(v),
            "bias" => b.toggle_bias_rate(v),
            "tau" => b.tau(v),
            "pop" => {
                pop_size = v as usize;
                b
            }
            "reset" => {
                reset = v != 0.0;
                b
            }
            "target" => {
                target = v;
                b
            }
            _ => return Err(format!("unknown key {}", k)),
        };
    }
    Ok(Bench {
        runs,
        max_gens,
        env: Xor {
            bias_input: mode == "initb",
            reset,
        },
        mode,
        pop_size,
        target,
        seed,
        cfg: b.build(),
        extra: args[4..].to_vec(),
    })
}

fn new_population(bench: &Bench, seed: Option<u64>) -> Population {
    let (n, cfg) = (bench.pop_size, bench.cfg);
    let seeded = |p: Population| match seed {
        Some(s) => p.with_seed(s),
        None => p,
    };
    match bench.mode.as_str() {
        "default" => {
            // Seed before overriding the config so the initial organisms keep
            // the default tau, exactly as without seed.
            let mut p = seeded(Population::create_population(n));
            p.mutation_config = cfg;
            p
        }
        "init" => seeded(Population::create_population_initialized_with_config(
            n, 2, 1, cfg,
        )),
        "initb" => seeded(Population::create_population_initialized_with_config(
            n, 3, 1, cfg,
        )),
        "uncon" => seeded(Population::create_population_unconnected_with_config(
            n, 2, 1, cfg,
        )),
        _ => unreachable!("mode validated in parse_args"),
    }
}

fn run_once(bench: &Bench, seed: Option<u64>) -> RunResult {
    let t0 = Instant::now();
    let env = bench.env;
    let mut pop = new_population(bench, seed);
    let mut r = RunResult {
        pop_min: usize::MAX,
        ..Default::default()
    };
    let mut champ: Option<Organism> = None;
    let mut last_champ_fit = 0.0f64;
    for g in 0..bench.max_gens {
        pop.evolve();
        pop.evaluate_in(&env);
        let n = pop.size();
        r.pop_min = r.pop_min.min(n);
        r.pop_max = r.pop_max.max(n);
        r.pop_end = n;
        r.species_end = pop.species.len();
        r.species_max = r.species_max.max(pop.species.len());
        if let Some(c) = &pop.champion {
            if c.fitness < last_champ_fit - 1e-12 {
                r.champ_regress += 1;
            }
            last_champ_fit = c.fitness;
        }
        for o in pop.get_organisms() {
            r.best_fit = r.best_fit.max(o.fitness);
            if o.fitness > bench.target && champ.is_none() {
                champ = Some(o);
            }
        }
        if champ.is_some() {
            r.solved_gen = Some(g + 1);
            break;
        }
    }
    if let Some(mut c) = champ {
        r.neurons = c.genome.len();
        r.genes = c.genome.total_genes();
        r.enabled = c.genome.get_genes().iter().filter(|g| g.enabled()).count();
        r.max_abs_w = c
            .genome
            .get_genes()
            .iter()
            .fold(0.0, |m, g| g.weight().abs().max(m));
        c.reset_state();
        r.fit_same = Xor::fit(&env.outputs(&mut c, &[0, 1, 2, 3], env.reset));
        c.reset_state();
        r.fit_reset_each = Xor::fit(&env.outputs(&mut c, &[0, 1, 2, 3], true));
        c.reset_state();
        r.fit_reversed = Xor::fit(&env.outputs(&mut c, &[3, 2, 1, 0], false));
    }
    r.secs = t0.elapsed().as_secs_f64();
    r
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn summary(bench: &Bench, rs: &[RunResult]) -> String {
    let mut gens: Vec<usize> = rs.iter().filter_map(|r| r.solved_gen).collect();
    gens.sort_unstable();
    let solved = gens.len();
    let median = if solved > 0 { gens[solved / 2] } else { 0 };
    let gens_mean = mean(&gens.iter().map(|&g| g as f64).collect::<Vec<_>>());
    let s: Vec<&RunResult> = rs.iter().filter(|r| r.solved_gen.is_some()).collect();
    let avg = |f: fn(&RunResult) -> f64| mean(&s.iter().map(|r| f(r)).collect::<Vec<_>>());
    let robust = |f: fn(&RunResult) -> f64| s.iter().filter(|r| f(r) > bench.target).count();
    let unsolved_best: Vec<f64> = rs
        .iter()
        .filter(|r| r.solved_gen.is_none())
        .map(|r| r.best_fit)
        .collect();
    format!(
        "SUMMARY mode={} args={:?} runs={} max_gens={} solved={} ({:.0}%) gens_median={} \
         gens_mean={:.1} gens_min={} gens_max={} | champion: neurons_avg={:.1} genes_avg={:.1} \
         enabled_avg={:.1} max|w|_avg={:.1} | reproduced_same_order={}/{} \
         still_solved_reset_each={}/{} still_solved_reversed_order={}/{} | pop_min={} pop_max={} \
         species_max={} species_end_avg={:.1} champ_regress_runs={} best_fit_unsolved_avg={:.2} \
         total_secs={:.1}",
        bench.mode,
        bench.extra,
        rs.len(),
        bench.max_gens,
        solved,
        100.0 * solved as f64 / rs.len().max(1) as f64,
        median,
        gens_mean,
        gens.first().copied().unwrap_or(0),
        gens.last().copied().unwrap_or(0),
        avg(|r| r.neurons as f64),
        avg(|r| r.genes as f64),
        avg(|r| r.enabled as f64),
        avg(|r| r.max_abs_w),
        robust(|r| r.fit_same),
        solved,
        robust(|r| r.fit_reset_each),
        solved,
        robust(|r| r.fit_reversed),
        solved,
        rs.iter().map(|r| r.pop_min).min().unwrap_or(0),
        rs.iter().map(|r| r.pop_max).max().unwrap_or(0),
        rs.iter().map(|r| r.species_max).max().unwrap_or(0),
        mean(&rs.iter().map(|r| r.species_end as f64).collect::<Vec<_>>()),
        rs.iter().filter(|r| r.champ_regress > 0).count(),
        mean(&unsolved_best),
        rs.iter().map(|r| r.secs).sum::<f64>(),
    )
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bench = match parse_args(&args) {
        Ok(b) => Arc::new(b),
        Err(e) => {
            eprintln!("{}", e);
            std::process::exit(2);
        }
    };
    let nthreads: usize = std::env::var("BENCH_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(6);
    let results = Arc::new(Mutex::new(Vec::<RunResult>::new()));
    let next = Arc::new(Mutex::new(0usize));
    let handles: Vec<_> = (0..nthreads)
        .map(|_| {
            let (bench, results, next) = (bench.clone(), results.clone(), next.clone());
            std::thread::spawn(move || loop {
                let run = {
                    let mut n = next.lock().unwrap();
                    if *n >= bench.runs {
                        break;
                    }
                    *n += 1;
                    *n - 1
                };
                let seed = bench.seed.map(|s| s.wrapping_add(run as u64));
                let r = run_once(&bench, seed);
                println!("run={} seed={:?} {:?}", run, seed, r);
                results.lock().unwrap().push(r);
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    let rs = results.lock().unwrap();
    println!("{}", summary(&bench, &rs));
}
