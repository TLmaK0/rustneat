//! XOR benchmark: runs XOR many times and prints one `SUMMARY` line so every
//! change can be measured against the same yardstick.
//!
//! usage: `xor_bench <runs> <max_gens> <mode> [key=value ...]`; run it
//! without arguments for the modes, the keys and the meaning of every
//! `SUMMARY` field (`xor_bench_common::USAGE`).
//!
//! Reference ("bench A"): `xor_bench 50 500 init reset=1`.
extern crate rustneat;

mod xor_bench_common;

use std::sync::{Arc, Mutex};
use xor_bench_common::{parse_args, run_once, summary, RunResult};

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
