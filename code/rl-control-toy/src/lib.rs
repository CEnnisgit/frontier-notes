//! Pendulum-on-cart toy for the Degrave-style RL-vs-classical-control
//! comparison. `env` owns the dynamics, `pid` owns the classical baseline.
//! Week 8 will add `rl` for the actor-critic variant.

pub mod env;
pub mod pid;
