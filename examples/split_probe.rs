//! Probe (issues #68, #107): how much does splitting a connection change the
//! output?
//!
//! For each input range [lo, hi] it evolves bench A populations (XOR with
//! the inputs 0 and 1 mapped to lo and hi, reset each pattern) for <gens>
//! generations, splits every enabled gene of every organism with several
//! rules (in->new weight, new bias, new->out weight, destination bias shift)
//! and reports the mean output change on the XOR patterns and on a grid of
//! [lo, hi]^2, and how often the fitness drops by more than 0.5.
//!
//! usage: split_probe <gens> <seeds...>
extern crate rustneat;
use rustneat::{Environment, Gene, Genome, MutationConfig, Organism, Population};

const PATTERNS: [(f64, f64, f64); 4] = [(0., 0., 0.), (0., 1., 1.), (1., 0., 1.), (1., 1., 0.)];
const WEIGHT_MAX: f64 = 30.0;

/// XOR whose inputs 0 and 1 are presented as `lo` and `hi`.
#[derive(Clone, Copy)]
struct Xor {
    lo: f64,
    hi: f64,
}

impl Xor {
    fn input(&self, x: f64) -> f64 {
        self.lo + (self.hi - self.lo) * x
    }
}

impl Environment for Xor {
    fn test(&self, o: &mut Organism) -> f64 {
        let d: f64 = PATTERNS
            .iter()
            .map(|&(a, b, t)| (t - out(o, self.input(a), self.input(b))).powi(2))
            .sum();
        16.0 / (1.0 + d)
    }
}

fn out(o: &mut Organism, a: f64, b: f64) -> f64 {
    o.reset_state();
    let mut r = vec![0f64];
    o.activate(vec![a, b], &mut r);
    r[0]
}

/// A split of `i -> o` (weight w): `i -> new` weight a, bias θ of the new
/// neuron, `new -> o` weight b and shift of the bias of o.
struct Split {
    a: f64,
    theta: f64,
    b: f64,
    dest_shift: f64,
}

/// The current rule fit to [0, 1]: σ(5·s - 2.5) ≈ s.
fn unit(w: f64) -> Split {
    Split {
        a: 5.0,
        theta: -2.5,
        b: w,
        dest_shift: 0.0,
    }
}

/// The current rule rescaled to [lo, hi]: u = (s - lo) / (hi - lo) is in
/// [0, 1] and s ≈ lo + (hi - lo)·σ(5·u - 2.5).
fn rescaled(w: f64, lo: f64, hi: f64) -> Split {
    let r = hi - lo;
    Split {
        a: 5.0 / r,
        theta: -2.5 - 5.0 * lo / r,
        b: w * r,
        dest_shift: w * lo,
    }
}

/// A rule: the split of a gene of weight w whose source is an input (true)
/// or a CTRNN neuron (false), for the input range [lo, hi].
type Rule = fn(f64, bool, f64, f64) -> Split;

fn split(g: &Genome, index: usize, rule: Rule, lo: f64, hi: f64) -> Genome {
    let inputs = g.input_neurons();
    let mut genes: Vec<Gene> = g.get_genes().clone();
    let old = genes[index];
    genes[index].set_disabled();
    let new_id = g.len();
    let w = old.weight();
    let s = rule(w, old.in_neuron_id() < inputs, lo, hi);
    genes.push(Gene::new(old.in_neuron_id(), new_id, s.a, true));
    genes.push(Gene::new(
        new_id,
        old.out_neuron_id(),
        s.b.clamp(-WEIGHT_MAX, WEIGHT_MAX),
        true,
    ));
    let mut n = Genome::from_genes_with_io(genes, new_id, inputs, g.output_neurons());
    for id in g.neurons() {
        n.set_bias(id, g.bias(id));
    }
    n.set_bias(new_id, s.theta);
    let d = old.out_neuron_id();
    n.set_bias(d, g.bias(d) + s.dest_shift);
    n
}

fn rules() -> Vec<(&'static str, Rule)> {
    vec![
        ("paper a=1 th=0 b=w", |w, _, _, _| Split {
            a: 1.0,
            theta: 0.0,
            b: w,
            dest_shift: 0.0,
        }),
        ("current a=5 th=-2.5 b=w (fit [0,1])", |w, _, _, _| unit(w)),
        ("[-1,1] fixed: a=2.5 th=0 b=2w dest-=w", |w, input, _, _| {
            if input {
                rescaled(w, -1.0, 1.0)
            } else {
                unit(w)
            }
        }),
        ("centered a=1 th=0 b=4w dest-=2w", |w, input, _, _| {
            if input {
                Split {
                    a: 1.0,
                    theta: 0.0,
                    b: 4.0 * w,
                    dest_shift: -2.0 * w,
                }
            } else {
                unit(w)
            }
        }),
        ("range-aware [lo,hi], no dest shift", |w, input, lo, hi| {
            if input {
                Split {
                    dest_shift: 0.0,
                    ..rescaled(w, lo, hi)
                }
            } else {
                unit(w)
            }
        }),
        (
            "range-aware [lo,hi] (split_input_range)",
            |w, input, lo, hi| {
                if input {
                    rescaled(w, lo, hi)
                } else {
                    unit(w)
                }
            },
        ),
    ]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: split_probe <gens> <seeds...>");
        std::process::exit(2);
    }
    let gens: usize = args[1].parse().unwrap();
    let seeds: Vec<u64> = args[2..].iter().map(|s| s.parse().unwrap()).collect();
    for &(lo, hi) in &[(0.0, 1.0), (-1.0, 1.0), (-3.0, 3.0), (-10.0, 10.0)] {
        let task = Xor { lo, hi };
        let mut organisms = vec![];
        let mut solved = 0;
        for &seed in &seeds {
            let mut p = Population::create_population_initialized_with_config(
                150,
                2,
                1,
                MutationConfig::default(),
            )
            .with_seed(seed);
            for _ in 0..gens {
                p.evaluate_in(&task).unwrap();
                p.evolve();
            }
            p.evaluate_in(&task).unwrap();
            if p.get_organisms().iter().any(|o| o.fitness > 15.0) {
                solved += 1;
            }
            organisms.extend(p.get_organisms());
        }
        println!(
            "inputs [{}, {}]: {} organisms, {} of {} seeds with fitness > 15",
            lo,
            hi,
            organisms.len(),
            solved,
            seeds.len()
        );
        let grid: Vec<(f64, f64)> = (0..=10)
            .flat_map(|i| (0..=10).map(move |j| (i as f64 / 10.0, j as f64 / 10.0)))
            .map(|(i, j)| (task.input(i), task.input(j)))
            .collect();
        for (label, rule) in rules() {
            let (mut n, mut sum_xor, mut sum_grid, mut worse) = (0usize, 0f64, 0f64, 0usize);
            let (mut n_in, mut s_in, mut n_hid, mut s_hid) = (0usize, 0f64, 0usize, 0f64);
            for o in &organisms {
                let inputs = o.genome.input_neurons();
                for (index, gene) in o.genome.get_genes().iter().enumerate() {
                    if !gene.enabled() {
                        continue;
                    }
                    let mut before = o.clone();
                    let mut after = o.clone();
                    after.genome = split(&o.genome, index, rule, lo, hi);
                    let mut dx = 0f64;
                    for &(a, b, _) in &PATTERNS {
                        let (a, b) = (task.input(a), task.input(b));
                        dx += (out(&mut before, a, b) - out(&mut after, a, b)).abs() / 4.0;
                    }
                    for &(a, b) in &grid {
                        sum_grid += (out(&mut before, a, b) - out(&mut after, a, b)).abs()
                            / grid.len() as f64;
                    }
                    if task.test(&mut after) < task.test(&mut before) - 0.5 {
                        worse += 1;
                    }
                    n += 1;
                    sum_xor += dx;
                    if gene.in_neuron_id() < inputs {
                        n_in += 1;
                        s_in += dx;
                    } else {
                        n_hid += 1;
                        s_hid += dx;
                    }
                }
            }
            println!(
                "  {:<42} splits={} mean|dout| xor={:.4} (from input {:.4} n={}, from neuron {:.4} n={}) grid={:.4} fit drop>0.5: {:.1}%",
                label,
                n,
                sum_xor / n as f64,
                s_in / n_in.max(1) as f64,
                n_in,
                s_hid / n_hid.max(1) as f64,
                n_hid,
                sum_grid / n as f64,
                100.0 * worse as f64 / n as f64
            );
        }
    }
}
