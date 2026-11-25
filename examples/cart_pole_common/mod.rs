//! Code of the cart pole example (`openai`) that does not need Python, so
//! `tests/cart_pole_tests.rs` can run it without the `openai` feature.

use rustneat::Organism;

/// Gymnasium id of the environment: +1 reward per step, episodes truncated
/// at 500 steps, solved at an average reward of 475.
pub const ENV_ID: &str = "CartPole-v1";
/// Cart position, cart velocity, pole angle, pole angular velocity.
pub const INPUTS: usize = 4;
pub const OUTPUTS: usize = 1;
/// Reward the champion must exceed.
pub const SOLVED_REWARD: f64 = 475.0;

/// One `env.step(action)` of the gymnasium API (its 5-tuple without `info`).
#[derive(Debug, Clone, PartialEq)]
pub struct Step {
    pub observation: Vec<f64>,
    pub reward: f64,
    pub terminated: bool,
    pub truncated: bool,
}

/// Action for the network output σ(y + θ), in (0, 1): push the cart right
/// (1) above one half, left (0) otherwise.
pub fn action(output: f64) -> i64 {
    if output > 0.5 {
        1
    } else {
        0
    }
}

/// Run one episode from the observation `env.reset()` returned and give its
/// total reward. `step` performs `env.step(action)`; the episode ends when it
/// reports `terminated` or `truncated`, or as soon as it fails.
pub fn run_episode<E, F>(
    organism: &mut Organism,
    initial_observation: Vec<f64>,
    mut step: F,
) -> Result<f64, E>
where
    F: FnMut(i64) -> Result<Step, E>,
{
    organism.reset_state();
    let mut observation = initial_observation;
    let mut output = vec![0f64; OUTPUTS];
    let mut total_reward = 0f64;
    loop {
        organism.activate(observation, &mut output);
        let result = step(action(output[0]))?;
        total_reward += result.reward;
        if result.terminated || result.truncated {
            return Ok(total_reward);
        }
        observation = result.observation;
    }
}
