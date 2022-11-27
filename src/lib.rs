extern crate core;

pub mod index;
mod macros;
mod matrix;

pub use matrix::{LUSolve, Matrix, Vector};
