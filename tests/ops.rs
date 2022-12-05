use generic_parameterize::parameterize;
use num_traits::real::Real;
use num_traits::Zero;
use std::fmt::Debug;
use std::iter::{zip, Product, Sum};
use std::ops;
use vector_victor::{LUSolve, Matrix, Vector};

macro_rules! scalar_eq {
    ($left:expr, $right:expr $(,)?) => {
        match (&$left, &$right) {
            (_left_val, _right_val) => {
                scalar_eq!($left, $right, "Difference is less than epsilon")
            }
        }
    };
    ($left:expr, $right:expr, $($arg:tt)+) => {
        match (&$left, &$right) {
            (left_val, right_val) => {
                let epsilon = f32::epsilon() as f64;
                let lf : f64 = (*left_val).into();
                let rf : f64 = (*right_val).into();
                let diff : f64 = (lf - rf).abs();
                if diff >= epsilon {
                    assert_eq!(left_val, right_val, $($arg)+) // done this way to get nice errors
                }
            }
        }
    };
}

macro_rules! matrix_eq {
    ($left:expr, $right:expr $(,)?) => {
        match (&$left, &$right) {
            (_left_val, _right_val) => {
                matrix_eq!($left, $right, "Difference is less than epsilon")
            }
        }
    };
    ($left:expr, $right:expr, $($arg:tt)+) => {
        match (&$left, &$right) {
            (left_val, right_val) => {
                let epsilon = f32::epsilon() as f64;
                for (l, r) in zip(left_val.elements(), right_val.elements()) {
                    let lf : f64 = (*l).into();
                    let rf : f64 = (*r).into();
                    let diff : f64 = (lf - rf).abs();
                    if diff >= epsilon {
                        assert_eq!($left, $right, $($arg)+) // done this way to get nice errors
                    }
                }

            }
        }
    };
}

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
fn test_lu_identity<S: Default + Real + Debug + Product + Sum + Into<f64>, const M: usize>() {
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
    scalar_eq!(d, S::one(), "Incorrect permutation parity");
    scalar_eq!(i.det(), S::one());
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
    let (_lu, idx, _d) = decomp;
    // the decomposition is non-unique, due to the combination of lu and idx.
    // Instead of checking the exact value, we only check the results.
    // Also check if they produce the same results with both methods, since the
    // Matrix<> methods use shortcuts the decomposition methods don't

    let (l, u) = decomp.separate();
    matrix_eq!(l.mmul(&u), a.permute_rows(&idx));

    scalar_eq!(a.det(), -6.0);
    scalar_eq!(a.det(), decomp.det());

    matrix_eq!(
        a.inverse().unwrap(),
        Matrix::new([[0.0, 2.0], [3.0, -1.0]]) * (1.0 / 6.0)
    );
    matrix_eq!(a.inverse().unwrap(), decomp.inverse());
    matrix_eq!(a.inverse().unwrap().inverse().unwrap(), a)
}

#[test]
fn test_lu_3x3() {
    let a = Matrix::new([[1.0, -5.0, 8.0], [1.0, -2.0, 1.0], [2.0, -1.0, -4.0]]);
    let decomp = a.lu().expect("Singular matrix encountered");
    let (_lu, idx, _d) = decomp;

    let (l, u) = decomp.separate();
    matrix_eq!(l.mmul(&u), a.permute_rows(&idx));

    scalar_eq!(a.det(), 3.0);
    scalar_eq!(a.det(), decomp.det());

    matrix_eq!(
        a.inverse().unwrap(),
        Matrix::new([[9.0, -28.0, 11.0], [6.0, -20.0, 7.0], [3.0, -9.0, 3.0]]) * (1.0 / 3.0)
    );
    matrix_eq!(a.inverse().unwrap(), decomp.inverse());
    matrix_eq!(a.inverse().unwrap().inverse().unwrap(), a)
}
