//! The XOR benchmark of `examples/xor_bench.rs`: argument parsing, one run
//! and the `SUMMARY` line. Tested by `tests/xor_bench_tests.rs`.

use rustneat::{
    Environment, Integrator, MutationConfig, Organism, OutputActivation, Population,
    SpeciesAssignment,
};
use std::time::Instant;

pub const PATTERNS: [(f64, f64, f64); 4] = [(0., 0., 0.), (0., 1., 1.), (1., 0., 1.), (1., 1., 0.)];

/// The usage text: modes, keys and the meaning of the `SUMMARY` fields.
pub const USAGE: &str = "\
usage: xor_bench <runs> <max_gens> <default|init|initb|uncon> [key=value ...]

modes:
  `default` deprecated `create_population(pop)` (a single neuron, no inputs
            or outputs), kept only to compare with older measurements
  `init`    `create_population_initialized_with_config(pop, 2, 1, cfg)`
  `initb`   like `init` with 3 inputs (x1, x2, constant 1.0)
  `uncon`   `create_population_unconnected_with_config(pop, 2, 1, cfg)`

keys (every key not given keeps the library default, MutationConfig::default()):
  bench:       `pop` (population size, default 150), `reset` (0/1: reset the
               CTRNN state before each pattern during training, default 0),
               `target` (fitness that counts as solved, default 15.5 of 16),
               `seed` (run i uses Population::with_seed(seed + i), so every
               run is reproducible and independent; without it each run
               seeds from entropy)
  structure:   `addc`, `addn` (add connection / add neuron rate), `tog`
               (toggle expression rate), `splitw`, `splitb` (weight into and
               bias of the neuron added by a split)
  weights:     `wrate` (weight mutation rate), `wgene` (per-gene weight
               mutation probability), `wpert` (perturbation vs replacement
               probability), `wpow` (gaussian sigma), `winit` (initial
               range), `wmax` (weight limit)
  biases:      `bias` (bias mutation rate), `bgene`, `bpert`, `bpow`, `bmax`
               (as the weight keys)
  ctrnn:       `tau` (default tau), `dt` (integration step), `step` (time per
               activation), `integ` (0 = exponential Euler, the default;
               1 = explicit Euler), `out` (0 = sigmoid output, the default;
               1 = linear), `taurate`, `taugene`, `taupow` (tau mutation
               rate, per-neuron rate and log-normal sigma), `taumin`,
               `taumax` (tau range)
  speciation:  `thr` (compatibility threshold), `c1`, `c2`, `c3`, `cb`
               (coefficients: excess, disjoint, weights, biases), `ct`
               (tau coefficient), `nsmall` (N = 1 below this many genes;
               20 = paper, `inf` = never normalize), `tspec` (target number
               of species; 0 = fixed threshold), `tstep`, `tmin`, `tmax`
               (dynamic threshold step, minimum and maximum), `assign`
               (0 = first compatible species, the default; 1 = closest),
               `stag` (generations without improvement before a species is
               removed), `elit` (best species protected from that removal)
  reproduction: `mutp` (probability of mutation without crossover), `dis`
               (probability that a gene disabled in a parent is disabled in
               the child), `xsp` (interspecies crossover rate), `xavg`
               (probability that a crossover averages matching weights),
               `surv` (fraction of each species allowed to reproduce),
               `champ` (fewest offspring for which a species keeps a copy
               of its champion; `inf` = never), `reeval` (0/1: evaluate the
               elites again every generation)

The environment variable BENCH_THREADS sets how many runs go in parallel
(default 6).

champion checks (SUMMARY, counted over the solved runs; the champion is the
first organism over `target`, its state is reset before every check):
  reproduced_as_trained              patterns in training order, resetting
                                     the state before each pattern only if
                                     `reset=1` (the training evaluation)
  still_solved_reset_each            training order, state reset before
                                     each pattern
  still_solved_reversed_no_reset     reversed order, state NOT reset between
                                     patterns (does the answer depend on
                                     the previous patterns?)
  still_solved_reversed_reset_each   reversed order, state reset before each
                                     pattern
  still_solved_all_orders_reset_each all 24 orders, state reset before each
                                     pattern (solved only if every order is)

Reference (\"bench A\"): xor_bench 50 500 init reset=1";

/// Names of the champion checks in the `SUMMARY` line, in order.
pub const SUMMARY_CHECKS: [&str; 5] = [
    "reproduced_as_trained",
    "still_solved_reset_each",
    "still_solved_reversed_no_reset",
    "still_solved_reversed_reset_each",
    "still_solved_all_orders_reset_each",
];

#[derive(Clone, Copy)]
pub struct Xor {
    pub bias_input: bool,
    /// Reset the CTRNN state before each pattern during training.
    pub reset: bool,
}

impl Xor {
    /// The outputs for the patterns, presented in `order`; `res[k]` is the
    /// answer to pattern `k` whatever its position in `order`.
    pub fn outputs(&self, o: &mut Organism, order: &[usize], reset_each: bool) -> [f64; 4] {
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

    pub fn fit(res: &[f64; 4]) -> f64 {
        let d: f64 = PATTERNS
            .iter()
            .zip(res.iter())
            .map(|(p, r)| (p.2 - r).powi(2))
            .sum();
        16.0 / (1.0 + d)
    }

    /// Fitness with the patterns in `order`, starting from rest.
    fn fit_from_rest(&self, o: &mut Organism, order: &[usize], reset_each: bool) -> f64 {
        o.reset_state();
        Xor::fit(&self.outputs(o, order, reset_each))
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

/// The 24 orders in which the four patterns can be presented.
pub fn all_orders() -> Vec<[usize; 4]> {
    let mut orders = Vec::with_capacity(24);
    for a in 0..4 {
        for b in (0..4).filter(|&b| b != a) {
            for c in (0..4).filter(|&c| c != a && c != b) {
                orders.push([a, b, c, 6 - a - b - c]);
            }
        }
    }
    orders
}

/// Fitness of a champion under each check of `SUMMARY_CHECKS`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ChampionChecks {
    pub as_trained: f64,
    pub reset_each: f64,
    pub reversed_no_reset: f64,
    pub reversed_reset_each: f64,
    /// The lowest fitness over the 24 orders.
    pub all_orders_reset_each: f64,
}

pub fn champion_checks(env: &Xor, c: &mut Organism) -> ChampionChecks {
    const TRAINING: [usize; 4] = [0, 1, 2, 3];
    const REVERSED: [usize; 4] = [3, 2, 1, 0];
    ChampionChecks {
        as_trained: env.fit_from_rest(c, &TRAINING, env.reset),
        reset_each: env.fit_from_rest(c, &TRAINING, true),
        reversed_no_reset: env.fit_from_rest(c, &REVERSED, false),
        reversed_reset_each: env.fit_from_rest(c, &REVERSED, true),
        all_orders_reset_each: all_orders()
            .iter()
            .map(|order| env.fit_from_rest(c, order, true))
            .fold(f64::INFINITY, f64::min),
    }
}

#[derive(Debug, Clone, Default)]
pub struct RunResult {
    pub solved_gen: Option<usize>,
    pub pop_min: usize,
    pub pop_max: usize,
    pub pop_end: usize,
    pub species_max: usize,
    pub species_end: usize,
    /// Fewest / most species from generation 50 on (None if solved before).
    pub species_min_g50: Option<usize>,
    pub species_max_g50: Option<usize>,
    pub neurons: usize,
    pub genes: usize,
    pub enabled: usize,
    pub max_abs_w: f64,
    /// Fitness of the champion under each check (zero if unsolved).
    pub checks: ChampionChecks,
    pub champ_regress: usize,
    pub best_fit: f64,
    pub secs: f64,
}

pub struct Bench {
    pub runs: usize,
    pub max_gens: usize,
    pub mode: String,
    pub pop_size: usize,
    pub target: f64,
    pub seed: Option<u64>,
    pub cfg: MutationConfig,
    pub env: Xor,
    pub extra: Vec<String>,
}

pub fn parse_args(args: &[String]) -> Result<Bench, String> {
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
    let mut b = MutationConfig::builder();
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
            "bias" => b.bias_mutation_rate(v),
            "bpow" => b.bias_mutate_power(v),
            "wgene" => b.weight_gene_mutation_rate(v),
            "bgene" => b.bias_gene_mutation_rate(v),
            "wpert" => b.weight_perturbation_rate(v),
            "bpert" => b.bias_perturbation_rate(v),
            "wmax" => b.weight_max(v),
            "bmax" => b.bias_max(v),
            "tau" => b.tau(v),
            "dt" => b.dt(v),
            "taurate" => b.tau_mutation_rate(v),
            "taugene" => b.tau_gene_mutation_rate(v),
            "taupow" => b.tau_mutate_power(v),
            "taumin" => {
                let c = b.build();
                b.tau_range(v, c.tau_max)
            }
            "taumax" => {
                let c = b.build();
                b.tau_range(c.tau_min, v)
            }
            "ct" => b.compatibility_tau_coefficient(v),
            "dis" => b.disabled_gene_inheritance(v),
            "xavg" => b.crossover_weight_averaging_rate(v),
            "surv" => b.survival_threshold(v),
            // `v as usize` saturates: `champ=inf` is usize::MAX (no elitism).
            "champ" => b.champion_copy_min_offspring(v as usize),
            "splitw" | "splitb" => {
                let c = b.build();
                if k == "splitw" {
                    b.split_neuron(v, c.split_bias)
                } else {
                    b.split_neuron(c.split_input_weight, v)
                }
            }
            "step" => b.step_time(v),
            "integ" if v == 0.0 => b.integrator(Integrator::ExponentialEuler),
            "integ" if v == 1.0 => b.integrator(Integrator::Euler),
            "integ" => return Err(format!("bad value for integ: {} (0 or 1)", v)),
            "out" if v == 0.0 => b.output_activation(OutputActivation::Sigmoid),
            "out" if v == 1.0 => b.output_activation(OutputActivation::Linear),
            "out" => return Err(format!("bad value for out: {} (0 or 1)", v)),
            "stag" | "elit" => {
                let c = b.build();
                if k == "stag" {
                    b.species_stagnation(v as usize, c.species_elitism)
                } else {
                    b.species_stagnation(c.species_stagnation_generations, v as usize)
                }
            }
            "reeval" if v == 0.0 || v == 1.0 => b.reevaluate_elites(v == 1.0),
            "reeval" => return Err(format!("bad value for reeval: {} (0 or 1)", v)),
            "xsp" => b.interspecies_mate_rate(v),
            "c1" | "c2" | "c3" | "cb" => {
                let c = b.build();
                let mut k4 = [
                    c.compatibility_excess_coefficient,
                    c.compatibility_disjoint_coefficient,
                    c.compatibility_weight_coefficient,
                    c.compatibility_bias_coefficient,
                ];
                k4[["c1", "c2", "c3", "cb"]
                    .iter()
                    .position(|x| *x == k)
                    .unwrap()] = v;
                b.compatibility_coefficients(k4[0], k4[1], k4[2], k4[3])
            }
            "nsmall" => b.compatibility_small_genome_genes(v as usize),
            "tspec" if v == 0.0 => b.fixed_compatibility_threshold(),
            "tspec" => b.target_species(v as usize),
            "tstep" | "tmin" => {
                let c = b.build();
                if k == "tstep" {
                    b.compatibility_threshold_adjustment(v, c.compatibility_threshold_min)
                } else {
                    b.compatibility_threshold_adjustment(c.compatibility_threshold_step, v)
                }
            }
            "tmax" => b.compatibility_threshold_max(v),
            "assign" if v == 0.0 => b.species_assignment(SpeciesAssignment::FirstCompatible),
            "assign" if v == 1.0 => b.species_assignment(SpeciesAssignment::Closest),
            "assign" => return Err(format!("bad value for assign: {} (0 or 1)", v)),
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
        #[allow(deprecated)]
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

pub fn run_once(bench: &Bench, seed: Option<u64>) -> RunResult {
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
        pop.evaluate_in(&env)
            .expect("the environment must return a finite fitness");
        let n = pop.size();
        r.pop_min = r.pop_min.min(n);
        r.pop_max = r.pop_max.max(n);
        r.pop_end = n;
        r.species_end = pop.species.len();
        r.species_max = r.species_max.max(pop.species.len());
        if g + 1 >= 50 {
            let s = pop.species.len();
            r.species_min_g50 = Some(r.species_min_g50.map_or(s, |m| m.min(s)));
            r.species_max_g50 = Some(r.species_max_g50.map_or(s, |m| m.max(s)));
        }
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
        r.neurons = c.genome.neurons().len();
        r.genes = c.genome.total_genes();
        r.enabled = c.genome.get_genes().iter().filter(|g| g.enabled()).count();
        r.max_abs_w = c
            .genome
            .get_genes()
            .iter()
            .fold(0.0, |m, g| g.weight().abs().max(m));
        r.checks = champion_checks(&env, &mut c);
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

pub fn summary(bench: &Bench, rs: &[RunResult]) -> String {
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
    let check_fits: [fn(&RunResult) -> f64; 5] = [
        |r| r.checks.as_trained,
        |r| r.checks.reset_each,
        |r| r.checks.reversed_no_reset,
        |r| r.checks.reversed_reset_each,
        |r| r.checks.all_orders_reset_each,
    ];
    let checks = SUMMARY_CHECKS
        .iter()
        .zip(check_fits)
        .map(|(name, f)| format!("{}={}/{}", name, robust(f), solved))
        .collect::<Vec<_>>()
        .join(" ");
    format!(
        "SUMMARY mode={} args={:?} runs={} max_gens={} solved={} ({:.0}%) gens_median={} \
         gens_mean={:.1} gens_min={} gens_max={} | champion: neurons_avg={:.1} genes_avg={:.1} \
         enabled_avg={:.1} max|w|_avg={:.1} | {} | pop_min={} pop_max={} \
         species_max={} species_end_avg={:.1} | from_gen50: runs={} species_min={} species_max={} \
         runs_always_5_to_20={} | champ_regress_runs={} best_fit_unsolved_avg={:.2} \
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
        checks,
        rs.iter().map(|r| r.pop_min).min().unwrap_or(0),
        rs.iter().map(|r| r.pop_max).max().unwrap_or(0),
        rs.iter().map(|r| r.species_max).max().unwrap_or(0),
        mean(&rs.iter().map(|r| r.species_end as f64).collect::<Vec<_>>()),
        rs.iter().filter(|r| r.species_min_g50.is_some()).count(),
        rs.iter()
            .filter_map(|r| r.species_min_g50)
            .min()
            .unwrap_or(0),
        rs.iter()
            .filter_map(|r| r.species_max_g50)
            .max()
            .unwrap_or(0),
        rs.iter()
            .filter(|r| {
                r.species_min_g50.is_some_and(|m| m >= 5)
                    && r.species_max_g50.is_some_and(|m| m <= 20)
            })
            .count(),
        rs.iter().filter(|r| r.champ_regress > 0).count(),
        mean(&unsolved_best),
        rs.iter().map(|r| r.secs).sum::<f64>(),
    )
}
