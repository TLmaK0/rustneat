//! The cart pole example (issue #95): the episode loop of
//! `examples/cart_pole_common` runs against a fake environment, so it is
//! tested without Python and without the `openai` feature.

#[path = "../examples/cart_pole_common/mod.rs"]
#[allow(dead_code)]
mod cart_pole_common;

use cart_pole_common::{action, run_episode, Step, INPUTS, OUTPUTS};
use rustneat::{Gene, Genome, Organism};

/// An organism whose single output follows the sign of the first input.
fn follows_first_input() -> Organism {
    let mut genome = Genome::new_unconnected(INPUTS, OUTPUTS);
    genome.add_gene(Gene::new(0, INPUTS, 10.0, true));
    Organism::new(genome)
}

fn observation(x: f64) -> Vec<f64> {
    vec![x, 0.0, 0.0, 0.0]
}

#[test]
fn action_pushes_right_only_above_one_half() {
    assert_eq!(action(0.9), 1);
    assert_eq!(action(0.51), 1);
    assert_eq!(action(0.5), 0);
    assert_eq!(action(0.1), 0);
}

#[test]
fn episode_acts_on_the_reset_observation_and_then_on_each_step_observation() {
    let mut organism = follows_first_input();
    let next = [-1.0, 1.0, -1.0];
    let mut actions = Vec::new();
    let total = run_episode(&mut organism, observation(1.0), |a| {
        actions.push(a);
        let i = actions.len() - 1;
        Ok::<_, ()>(Step {
            observation: observation(next[i]),
            reward: 1.0,
            terminated: i == 2,
            truncated: false,
        })
    })
    .unwrap();
    assert_eq!(actions, vec![1, 0, 1]);
    assert_eq!(total, 3.0);
}

#[test]
fn truncation_ends_the_episode() {
    let mut organism = follows_first_input();
    let mut steps = 0;
    let total = run_episode(&mut organism, observation(1.0), |_| {
        steps += 1;
        Ok::<_, ()>(Step {
            observation: observation(1.0),
            reward: 0.5,
            terminated: false,
            truncated: steps == 4,
        })
    })
    .unwrap();
    assert_eq!(steps, 4);
    assert_eq!(total, 2.0);
}

#[test]
fn an_environment_error_stops_the_episode() {
    let mut organism = follows_first_input();
    let mut steps = 0;
    let result = run_episode(&mut organism, observation(1.0), |_| {
        steps += 1;
        if steps == 2 {
            return Err("step failed");
        }
        Ok(Step {
            observation: observation(1.0),
            reward: 1.0,
            terminated: false,
            truncated: false,
        })
    });
    assert_eq!(result, Err("step failed"));
    assert_eq!(steps, 2);
}
