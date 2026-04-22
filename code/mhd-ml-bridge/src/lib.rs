//! Brio-Wu MLP surrogate — scaffolding for the ML method layer.
//!
//! This crate is a Rust/candle port of the Month 1 Python/JAX scaffold. The
//! goal is the same: prove the stack (classical solver → training data → MLP →
//! predictions) runs end-to-end. See `README.md` for the honest framing of
//! what this is and isn't.

pub mod data;
pub mod surrogate;
pub mod sweep;
