use crate::impl_matrix_op;
use crate::index::Index2D;
use std::iter::{zip, Enumerate, Flatten};
use std::ops::{Add, Deref, DerefMut, Index, IndexMut, Range};
use std::option::IntoIter;

pub trait Get2D {
    type Scalar: Sized + Copy;
    const HEIGHT: usize;
    const WIDTH: usize;

    fn get<I: Index2D>(&self, i: I) -> Option<&Self::Scalar>;
}

pub trait Get2DMut: Get2D {
    fn get_mut<I: Index2D>(&mut self, i: I) -> Option<&mut Self::Scalar>;
}

trait Scalar: Copy + 'static {}
macro_rules! multi_impl { ($name:ident for $($t:ty),*) => ($( impl $name for $t {} )*) }
multi_impl!(Scalar for i8, i16, i32, i64, isize, u8, u16, u32, u64, usize, f32, f64);
impl<T> Scalar for &'static T where T: Scalar {}

#[derive(Debug, Copy, Clone)]
struct Matrix<T, const M: usize, const N: usize>
where
    T: Scalar,
{
    data: [[T; N]; M],
}

type Vector<T, const N: usize> = Matrix<T, N, 1>;

impl<T: Scalar, const M: usize, const N: usize> Matrix<T, M, N> {
    fn new(data: [[T; N]; M]) -> Self {
        return Matrix::<T, M, N> { data };
    }

    fn from_rows<I>(iter: &I) -> Self
    where
        Self: Default,
        I: Iterator<Item = Vector<T, N>> + Copy,
    {
        let mut result = Self::default();
        for (m, row) in iter.enumerate().filter(|(m, _)| *m <= M) {
            result.set_row(m, &row)
        }
        result
    }

    fn from_cols<I>(iter: &I) -> Self
    where
        Self: Default,
        I: Iterator<Item = Vector<T, M>> + Copy,
    {
        let mut result = Self::default();
        for (n, col) in iter.enumerate().filter(|(n, _)| *n <= N) {
            result.set_col(n, &col)
        }
        result
    }

    fn elements<'a>(&'a self) -> impl Iterator<Item = &T> + 'a {
        self.data.iter().flatten()
    }

    fn elements_mut<'a>(&'a mut self) -> impl Iterator<Item = &mut T> + 'a {
        self.data.iter_mut().flatten()
    }

    fn get(&self, index: impl Index2D) -> Option<&T> {
        let (m, n) = index.to_2d(M, N)?;
        Some(&self.data[m][n])
    }

    fn get_mut(&mut self, index: impl Index2D) -> Option<&mut T> {
        let (m, n) = index.to_2d(M, N)?;
        Some(&mut self.data[m][n])
    }

    fn row(&self, m: usize) -> Option<Vector<T, N>> {
        if m < M {
            Some(Vector::<T, N>::new_vector(self.data[m]))
        } else {
            None
        }
    }

    fn set_row(&mut self, m: usize, val: &Vector<T, N>) {
        assert!(
            m < M,
            "Row index {} out of bounds for {}x{} matrix",
            m,
            M,
            N
        );
        for (n, v) in val.elements().enumerate() {
            self.data[m][n] = *v;
        }
    }

    fn col(&self, n: usize) -> Option<Vector<T, M>> {
        if n < N {
            Some(Vector::<T, M>::new_vector(self.data.map(|r| r[n])))
        } else {
            None
        }
    }

    fn set_col(&mut self, n: usize, val: &Vector<T, M>) {
        assert!(
            n < N,
            "Column index {} out of bounds for {}x{} matrix",
            n,
            M,
            N
        );

        for (m, v) in val.elements().enumerate() {
            self.data[m][n] = *v;
        }
    }

    fn rows<'a>(&'a self) -> impl Iterator<Item = Vector<T, N>> + 'a {
        (0..M).map(|m| self.row(m).expect("invalid row reached while iterating"))
    }

    fn cols<'a>(&'a self) -> impl Iterator<Item = Vector<T, M>> + 'a {
        (0..N).map(|n| self.col(n).expect("invalid column reached while iterating"))
    }
}

// constructor for column vectors
impl<T: Scalar, const N: usize> Vector<T, N> {
    fn new_vector(data: [T; N]) -> Self {
        return Vector::<T, N> {
            data: data.map(|e| [e]),
        };
    }
}

// default constructor
impl<T, const M: usize, const N: usize> Default for Matrix<T, M, N>
where
    [[T; N]; M]: Default,
    T: Scalar,
{
    fn default() -> Self {
        Matrix {
            data: Default::default(),
        }
    }
}

// deref 1x1 matrices to a scalar automatically
impl<T: Scalar> Deref for Matrix<T, 1, 1> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.data[0][0]
    }
}

// deref 1x1 matrices to a mutable scalar automatically
impl<T: Scalar> DerefMut for Matrix<T, 1, 1> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data[0][0]
    }
}

impl<T: Scalar, const M: usize, const N: usize> IntoIterator for Matrix<T, M, N> {
    type Item = T;
    type IntoIter = Flatten<std::array::IntoIter<[T; N], M>>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter().flatten()
    }
}

impl_matrix_op!(neg, |l: L| { -l });
impl_matrix_op!(!, |l: L| { !l });
impl_matrix_op!(+, |l,r| {l + r});
impl_matrix_op!(-, |l,r| {l - r});
impl_matrix_op!(*, |l,r| {l * r});
impl_matrix_op!(/, |l,r| {l / r});
impl_matrix_op!(%, |l,r| {l % r});
impl_matrix_op!(&, |l,r| {l & r});
impl_matrix_op!(|, |l,r| {l | r});
impl_matrix_op!(^, |l,r| {l ^ r});
impl_matrix_op!(<<, |l,r| {l << r});
impl_matrix_op!(>>, |l,r| {l >> r});
