//! Pendulum-on-cart toy with an electromagnetic-style actuator and classical
//! controllers (PID cascade + LQR on the linearized system). Week 7 setup for
//! the Degrave-style RL vs. classical head-to-head that lands in Week 8.
//!
//! The actuator is not an ideal force: a coil-current first-order lag maps
//! the commanded voltage `V` to the physical cart force. Observations carry
//! Gaussian noise; voltages saturate. These are the knobs the Week-8 RL
//! policy has to share with PID/LQR.

pub mod env;
pub mod pid;
