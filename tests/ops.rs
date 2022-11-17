use generic_parameterize::parameterize;
use std::fmt::Debug;
use std::ops;
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
