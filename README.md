Vector-Victor ✈️
================

Vector-Victor is yet another crate that provides a matrix type for linear algebra

## What is it?
Vector-Victor is:
- In its Alpha stage
- Geared towards programmers familiar with linear algebra through computer graphics, or complete beginners
- Fully generic across both types and dimensions

## What is it not?
Vector-Victor does not:
- Provide variable-sized or sparse vectors
- Support complex numbers natively
- Directly replace/competete with [Nalgebra](https://www.nalgebra.org)

## Why does this exist? Why not use something else?

[![XKCD 927](https://imgs.xkcd.com/comics/standards_2x.png)](https://xkcd.com/927/)
I wrote Vector-Victor for myself, to reflect the way I think about matrices and vectors. I didn't want a library that
overcomplicated things with [Typenums](https://lib.rs/crates/typenum) or multiple backing datastructure options. I did 
want types that felt familiar to me from when I write shaders or game mods. 