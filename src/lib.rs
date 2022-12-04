extern crate core;

pub mod index;
mod macros;
mod matrix;
mod util;

pub use matrix::{LUSolve, Matrix, Vector};
