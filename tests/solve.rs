#[macro_use]
mod common;

use common::Approx;
use generic_parameterize::parameterize;
use num_traits::real::Real;
use num_traits::Zero;
use std::fmt::Debug;
use std::iter::{zip, Product, Sum};
use vector_victor::solve::{LUDecomp, LUSolve};
use vector_victor::{Matrix, Vector};

#[parameterize(S = (f32, f64), M = [1,2,3,4])]
#[test]
/// The LU decomposition of the identity matrix should produce
/// the identity matrix with no permutations and parity 1
fn test_lu_identity<S: Default + Approx + Real + Debug + Product + Sum, const M: usize>() {
    // let a: Matrix<f32, 3, 3> = Matrix::<f32, 3, 3>::identity();
    let i = Matrix::<S, M, M>::identity();
    let ones = Vector::<S, M>::fill(S::one());
    let decomp = i.lu().expect("Singular matrix encountered");
    let LUDecomp { lu, idx, parity } = decomp;
    assert_eq!(lu, i, "Incorrect LU decomposition");
    assert!(
        (0..M).eq(idx.elements().cloned()),
        "Incorrect permutation matrix",
    );
    assert_approx!(parity, S::one(), "Incorrect permutation parity");

    // Check determinant calculation which uses LU decomposition
    assert_approx!(
        i.det(),
        S::one(),
        "Identity matrix should have determinant of 1"
    );

    // Check inverse calculation with uses LU decomposition
    assert_eq!(
        i.inverse(),
        Some(i),
        "Identity matrix should be its own inverse"
    );
    assert_eq!(
        i.solve(&ones),
        Some(ones),
        "Failed to solve using identity matrix"
    );

    // Check triangle separation
    assert_eq!(decomp.separate(), (i, i));
}

#[parameterize(S = (f32, f64), M = [2,3,4])]
#[test]
/// The LU decomposition of any singular matrix should be `None`
fn test_lu_singular<S: Default + Real + Debug + Product + Sum, const M: usize>() {
    // let a: Matrix<f32, 3, 3> = Matrix::<f32, 3, 3>::identity();
    let mut a = Matrix::<S, M, M>::zero();
    let ones = Vector::<S, M>::fill(S::one());
    a.set_row(0, &ones);

    assert_eq!(a.lu(), None, "Matrix should be singular");
    assert_eq!(
        a.det(),
        S::zero(),
        "Singular matrix should have determinant of zero"
    );
    assert_eq!(a.inverse(), None, "Singular matrix should have no inverse");
    assert_eq!(
        a.solve(&ones),
        None,
        "Singular matrix should not be solvable"
    )
}

#[test]
fn test_lu_2x2() {
    let a = Matrix::new([[1.0, 2.0], [3.0, 0.0]]);
    let decomp = a.lu().expect("Singular matrix encountered");
    // the decomposition is non-unique, due to the combination of lu and idx.
    // Instead of checking the exact value, we only check the results.
    // Also check if they produce the same results with both methods, since the
    // Matrix<> methods use shortcuts the decomposition methods don't

    let (l, u) = decomp.separate();
    assert_approx!(l.mmul(&u), a.permute_rows(&decomp.idx));

    assert_approx!(a.det(), -6.0);
    assert_approx!(a.det(), decomp.det());

    assert_approx!(
        a.inverse().unwrap(),
        Matrix::new([[0.0, 2.0], [3.0, -1.0]]) * (1.0 / 6.0)
    );
    assert_approx!(a.inverse().unwrap(), decomp.inverse());
    assert_approx!(a.inverse().unwrap().inverse().unwrap(), a)
}

#[test]
fn test_lu_3x3() {
    let a = Matrix::new([[1.0, -5.0, 8.0], [1.0, -2.0, 1.0], [2.0, -1.0, -4.0]]);
    let decomp = a.lu().expect("Singular matrix encountered");

    let (l, u) = decomp.separate();
    assert_approx!(l.mmul(&u), a.permute_rows(&decomp.idx));

    assert_approx!(a.det(), 3.0);
    assert_approx!(a.det(), decomp.det());

    assert_approx!(
        a.inverse().unwrap(),
        Matrix::new([[9.0, -28.0, 11.0], [6.0, -20.0, 7.0], [3.0, -9.0, 3.0]]) * (1.0 / 3.0)
    );
    assert_approx!(a.inverse().unwrap(), decomp.inverse());
    assert_approx!(a.inverse().unwrap().inverse().unwrap(), a)
}
