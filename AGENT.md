# RustNEAT - Evolutionary Neural Networks

## Purpose
Rust implementation of the **NEAT** (NeuroEvolution of Augmenting Topologies) algorithm with **CTRNN** (Continuous-Time Recurrent Neural Networks) to evolve neural network topologies that solve control and classification problems.

## Current Plan
@[plan.md](plan.md) - Current feature development plan. This file is continuously updated as work progresses and is NOT versioned in git.

**Guidelines for plan.md:**
- For each step in "Next Steps", create a detailed implementation plan when you start working on it
- Break down each step into concrete sub-tasks with checkboxes
- Mark tasks as completed (✓) as you finish them
- Update the plan continuously as you discover new requirements or issues
- Move completed steps from "Next Steps" to "Implemented" section

## Main Architecture

### Core Components (`src/`)
- **Gene** (`gene.rs`): Connection between neurons (in_neuron, out_neuron, weight, enabled)
- **Genome** (`genome.rs`): Network topology (gene collection) + mutation operators
- **Organism** (`organism.rs`): Genome + fitness + CTRNN activation
- **Population** (`population.rs`): Species + evolution + selection
- **Specie** (`specie.rs`): Groups of similar genomes + fitness sharing
- **CTRNN** (`ctrnn.rs`): Continuous-time recurrent neural network

### Mutations
- Add connection (1%)
- Add neuron (1%)
- Modify weight (90%)
- Toggle expression (0.5%)
- Toggle bias (1%)

### Speciation
- Compatibility distance: δ = (c2 × D / N) + c3 × W
- Threshold: 1.3
- Fitness sharing within species

## CTRNN
- Equation: dy/dt = (-y + wji*σ(y+θ) + I) / τ
- τ = 0.01
- Euler integration: 10 steps of 0.01s
- Activation: sigmoid

## Project Status
**Branch**: `feature/openai_lunar_lander`
- Complete NEAT + Lunar Lander implementation
- Optimized threading for Python environments
- Species visualization and progress tracking
- Initialization with predefined topology
