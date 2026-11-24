//! Probe (issue #68): how much does splitting a connection change the output?
//! Evolves bench A populations (XOR, reset each pattern) for <gens>
//! generations, splits every enabled gene of every organism with several
//! rules (in->new weight, new bias, new->out weight, destination bias shift)
//! and reports the mean output change and how often the fitness drops.
//! usage: split_probe <gens> <seeds...>
extern crate rustneat;
use rustneat::{Environment, Gene, Genome, MutationConfig, Organism, Population};

const PATTERNS: [(f64, f64, f64); 4] = [(0., 0., 0.), (0., 1., 1.), (1., 0., 1.), (1., 1., 0.)];

struct Xor;
impl Environment for Xor {
    fn test(&self, o: &mut Organism) -> f64 {
        let d: f64 = PATTERNS
            .iter()
            .map(|&(a, b, t)| (t - out(o, a, b)).powi(2))
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

/// (in->new weight, new bias, new->out weight factor, dest bias shift factor of w)
/// for a source that is an input and for a source that is a CTRNN neuron.
type Rule = ((f64, f64, f64, f64), (f64, f64, f64, f64));

fn split(g: &Genome, index: usize, rule: Rule, inputs: usize) -> Genome {
    let mut genes: Vec<Gene> = g.get_genes().clone();
    let old = genes[index];
    genes[index].set_disabled();
    let new_id = g.len();
    let (a, th, bf, shift) = if old.in_neuron_id() < inputs {
        rule.0
    } else {
        rule.1
    };
    let w = old.weight();
    genes.push(Gene::new(old.in_neuron_id(), new_id, a, true));
    genes.push(Gene::new(
        new_id,
        old.out_neuron_id(),
        (bf * w).clamp(-30.0, 30.0),
        true,
    ));
    let mut n = Genome::from_genes(genes, new_id);
    for id in g.neurons() {
        n.set_bias(id, g.bias(id));
    }
    n.set_bias(new_id, th);
    let d = old.out_neuron_id();
    n.set_bias(d, g.bias(d) + shift * w);
    n
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let gens: usize = args[1].parse().unwrap();
    let seeds: Vec<u64> = args[2..].iter().map(|s| s.parse().unwrap()).collect();
    let rules: Vec<(&str, Rule)> = vec![
        (
            "V0 paper a=1 th=0 b=w",
            ((1., 0., 1., 0.), (1., 0., 1., 0.)),
        ),
        ("V1 a=4 th=-2 b=w", ((4., -2., 1., 0.), (4., -2., 1., 0.))),
        (
            "V1b a=5 th=-2.5 b=w (rustneat)",
            ((5., -2.5, 1., 0.), (5., -2.5, 1., 0.)),
        ),
        (
            "V2 input:a=1 b=4w dest-=2w; neuron:a=4 th=-2",
            ((1., 0., 4., -2.), (4., -2., 1., 0.)),
        ),
    ];
    let mut organisms = vec![];
    for &seed in &seeds {
        let cfg = MutationConfig::default();
        let mut p =
            Population::create_population_initialized_with_config(150, 2, 1, cfg).with_seed(seed);
        for _ in 0..gens {
            p.evaluate_in(&Xor).unwrap();
            p.evolve();
        }
        p.evaluate_in(&Xor).unwrap();
        organisms.extend(p.get_organisms());
    }
    // uniform grid of inputs in [-1, 1]^2
    let grid: Vec<(f64, f64)> = (0..=10)
        .flat_map(|i| (0..=10).map(move |j| (i as f64 / 5.0 - 1.0, j as f64 / 5.0 - 1.0)))
        .collect();
    for (label, rule) in &rules {
        let (mut n, mut sum_xor, mut sum_grid, mut big, mut dfit, mut worse) =
            (0usize, 0f64, 0f64, 0usize, 0f64, 0usize);
        let (mut n_in, mut s_in, mut n_hid, mut s_hid) = (0usize, 0f64, 0usize, 0f64);
        for o in &organisms {
            let genes = o.genome.get_genes();
            for (index, gene) in genes.iter().enumerate() {
                if !gene.enabled() {
                    continue;
                }
                let mut before = o.clone();
                let mut after = o.clone();
                after.genome = split(&o.genome, index, *rule, 2);
                let mut dx = 0f64;
                let mut maxd = 0f64;
                for &(a, b, _) in &PATTERNS {
                    let d = (out(&mut before, a, b) - out(&mut after, a, b)).abs();
                    dx += d / 4.0;
                    maxd = maxd.max(d);
                }
                let mut dg = 0f64;
                for &(a, b) in &grid {
                    dg +=
                        (out(&mut before, a, b) - out(&mut after, a, b)).abs() / grid.len() as f64;
                }
                let fb = Xor.test(&mut before);
                let fa = Xor.test(&mut after);
                n += 1;
                sum_xor += dx;
                sum_grid += dg;
                if maxd > 0.1 {
                    big += 1;
                }
                dfit += (fa - fb).abs();
                if fa < fb - 0.5 {
                    worse += 1;
                }
                if gene.in_neuron_id() < 2 {
                    n_in += 1;
                    s_in += dx;
                } else {
                    n_hid += 1;
                    s_hid += dx;
                }
            }
        }
        println!(
            "{:<46} splits={} mean|dout|xor={:.4} (from input {:.4} n={}, from neuron {:.4} n={}) mean|dout|grid[-1,1]={:.4} max|dout|>0.1: {:.1}% mean|dfit|={:.3} fit drop>0.5: {:.1}%",
            label,
            n,
            sum_xor / n as f64,
            s_in / n_in.max(1) as f64,
            n_in,
            s_hid / n_hid.max(1) as f64,
            n_hid,
            sum_grid / n as f64,
            100.0 * big as f64 / n as f64,
            dfit / n as f64,
            100.0 * worse as f64 / n as f64
        );
    }
}
