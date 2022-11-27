use generic_parameterize::parameterize;
use std::convert::identity;
use std::fmt::Debug;
use std::ops;
use std::thread::sleep;
use std::time::Duration;
use vector_victor::Matrix;

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

#[test]
fn test_lu() {
    // let a: Matrix<f32, 3, 3> = Matrix::<f32, 3, 3>::identity();
    let a = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    let (lu, _idx, _d) = a.lu().expect("What");
    println!("{:?}", lu);
}
