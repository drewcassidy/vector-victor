use generic_parameterize::parameterize;
use num_traits::real::Real;
use std::fmt::Debug;
use std::iter::{Product, Sum};
use std::ops;
use vector_victor::{Matrix, Vector};

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
    let (lu, idx, d) = i.lu().expect("Singular matrix encountered");
    assert_eq!(
        lu,
        i,
        "Incorrect LU decomposition matrix for {m}x{m} identity matrix",
        m = M
    );
    assert!(
        (0..M).eq(idx.elements().cloned()),
        "Incorrect permutation matrix result for {m}x{m} identity matrix",
        m = M
    );
    assert_eq!(
        d,
        S::one(),
        "Incorrect permutation parity for {m}x{m} identity matrix",
        m = M
    );
    assert_eq!(
        i.det(),
        S::one(),
        "Incorrect determinant for {m}x{m} identity matrix",
        m = M
    );
    assert_eq!(
        i.inverse(),
        Some(i),
        "Incorrect inverse for {m}x{m} identity matrix",
        m = M
    );
    assert_eq!(
        i.solve(&ones),
        Some(ones),
        "Incorrect solve result for {m}x{m} identity matrix",
        m = M
    )
}
