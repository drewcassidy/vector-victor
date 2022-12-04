use generic_parameterize::parameterize;
use num_traits::real::Real;
use num_traits::Zero;
use std::fmt::Debug;
use std::iter::{Product, Sum};
use std::ops;
use vector_victor::{LUSolve, Matrix, Vector};

#[parameterize(S = (i32, f32, u32), M = [1,4], N = [1,4])]
#[test]
fn test_add<S: Copy + From<u16> + PartialEq + Debug, const M: usize, const N: usize>()
where
    Matrix<S, M, N>: ops::Add<Output = Matrix<S, M, N>>,
{
    let a = Matrix::<S, M, N>::fill(S::from(1));
    let b = Matrix::<S, M, N>::fill(S::from(3));
    let c: Matrix<S, M, N> = a + b;
    for (i, ci) in c.elements().enumerate() {
        assert_eq!(*ci, S::from(4));
    }
}

#[parameterize(S = (f32, f64), M = [1,2,3,4])]
#[test]
fn test_lu_identity<S: Default + Real + Debug + Product + Sum, const M: usize>() {
    // let a: Matrix<f32, 3, 3> = Matrix::<f32, 3, 3>::identity();
    let i = Matrix::<S, M, M>::identity();
    let ones = Vector::<S, M>::fill(S::one());
    let decomp = i.lu().expect("Singular matrix encountered");
    let (lu, idx, d) = decomp;
    assert_eq!(lu, i, "Incorrect LU decomposition");
    assert!(
        (0..M).eq(idx.elements().cloned()),
        "Incorrect permutation matrix",
    );
    assert_eq!(d, S::one(), "Incorrect permutation parity");
    assert_eq!(i.det(), S::one());
    assert_eq!(i.inverse(), Some(i));
    assert_eq!(i.solve(&ones), Some(ones));
    assert_eq!(decomp.separate(), (i, i));
}

#[parameterize(S = (f32, f64), M = [2,3,4])]
#[test]
fn test_lu_singular<S: Default + Real + Debug + Product + Sum, const M: usize>() {
    // let a: Matrix<f32, 3, 3> = Matrix::<f32, 3, 3>::identity();
    let mut a = Matrix::<S, M, M>::zero();
    let ones = Vector::<S, M>::fill(S::one());
    a.set_row(0, &ones);

    assert_eq!(a.lu(), None, "Matrix should be singular");
    assert_eq!(a.det(), S::zero());
    assert_eq!(a.inverse(), None);
    assert_eq!(a.solve(&ones), None)
}

#[test]
fn test_lu_2x2() {
    let a = Matrix::new([[1.0, 2.0], [3.0, 0.0]]);
    let decomp = a.lu().expect("Singular matrix encountered");
    let (lu, idx, d) = decomp;
    // the decomposition is non-unique, due to the combination of lu and idx.
    // Instead of checking the exact value, we only check the results.
    // Also check if they produce the same results with both methods, since the
    // Matrix<> methods use shortcuts the decomposition methods don't

    let (l, u) = decomp.separate();
    assert_eq!(l.mmul(&u), a.permute_rows(&idx));

    assert_eq!(a.det(), -6.0);
    assert_eq!(a.det(), decomp.det());

    assert_eq!(
        a.inverse(),
        Some(Matrix::new([[0.0, 2.0], [3.0, -1.0]]) * (1.0 / 6.0))
    );
    assert_eq!(a.inverse(), Some(decomp.inverse()));
    assert_eq!(a.inverse().unwrap().inverse().unwrap(), a)
}
