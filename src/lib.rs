extern crate core;

use index::Index2D;
use std::cmp::min;
use std::fmt::Debug;
use std::iter::{zip, Flatten};
use std::ops::{Index, IndexMut};

pub mod decompose;
mod identities;
pub mod index;
mod math;
mod ops;

mod util;

/// A 2D array of values which can be operated upon.
///
/// Matrices have a fixed size known at compile time
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Matrix<T, const M: usize, const N: usize>
where
    T: Copy,
{
    data: [[T; N]; M], // Row-Major order
}

/// An alias for a [Matrix] with a single column
pub type Vector<T, const N: usize> = Matrix<T, N, 1>;

// CONSTRUCTORS

// Default
impl<T: Copy + Default, const M: usize, const N: usize> Default for Matrix<T, M, N> {
    fn default() -> Self {
        Matrix::fill(T::default())
    }
}

// Matrix constructors
impl<T: Copy, const M: usize, const N: usize> Matrix<T, M, N> {
    /// Generate a new matrix from a 2D Array
    ///
    /// # Arguments
    ///
    /// * `data`: A 2D array of elements to copy into the new matrix
    ///
    /// returns: Matrix<T, M, N>
    ///
    /// # Examples
    ///
    /// ```
    /// # use vector_victor::Matrix;
    /// let a = Matrix::mat([[1,2,3,4];4]);
    /// ```
    #[must_use]
    pub fn mat(data: [[T; N]; M]) -> Self {
        assert!(M > 0, "Matrix must have at least 1 row");
        assert!(N > 0, "Matrix must have at least 1 column");
        Matrix::<T, M, N> { data }
    }

    /// Generate a new matrix from a single scalar
    ///
    /// # Arguments
    ///
    /// * `scalar`: Scalar value to copy into the new matrix.
    ///
    /// returns: Matrix<T, M, N>
    ///
    /// # Examples
    ///
    /// ```
    /// # use vector_victor::Matrix;
    /// let my_matrix = Matrix::<i32,4,4>::fill(5);
    /// // is equivalent to
    /// assert_eq!(my_matrix, Matrix::mat([[5;4];4]))
    /// ```
    #[must_use]
    pub fn fill(scalar: T) -> Matrix<T, M, N> {
        assert!(M > 0, "Matrix must have at least 1 row");
        assert!(N > 0, "Matrix must have at least 1 column");
        Matrix::<T, M, N> {
            data: [[scalar; N]; M],
        }
    }

    /// Create a matrix from an iterator of vectors
    ///
    /// # Arguments
    ///
    /// * `iter`: iterator of vectors to copy into rows
    ///
    /// returns: Matrix<T, M, N>
    ///
    /// # Examples
    ///
    /// ```
    /// # use vector_victor::Matrix;
    /// let my_matrix = Matrix::mat([[1,2,3],[4,5,6]]);
    /// let transpose : Matrix<_,3,2>= Matrix::from_rows(my_matrix.cols());
    /// assert_eq!(transpose, Matrix::mat([[1,4],[2,5],[3,6]]))
    /// ```
    #[must_use]
    pub fn from_rows<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = Vector<T, N>>,
        Self: Default,
    {
        let mut result = Self::default();
        for (m, row) in iter.into_iter().enumerate().take(M) {
            result.set_row(m, &row)
        }
        result
    }

    /// Create a matrix from an iterator of vectors
    ///
    /// # Arguments
    ///
    /// * `iter`: iterator of vectors to copy into columns
    ///
    /// returns: Matrix<T, M, N>
    ///
    /// # Examples
    ///
    /// ```
    /// # use vector_victor::Matrix;
    /// let my_matrix = Matrix::mat([[1,2,3],[4,5,6]]);
    /// let transpose : Matrix<_,3,2>= Matrix::from_cols(my_matrix.rows());
    /// assert_eq!(transpose, Matrix::mat([[1,4],[2,5],[3,6]]))
    /// ```
    #[must_use]
    pub fn from_cols<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = Vector<T, M>>,
        Self: Default,
    {
        let mut result = Self::default();
        for (n, col) in iter.into_iter().enumerate().take(N) {
            result.set_col(n, &col)
        }
        result
    }
}

// Vector constructor
impl<T: Copy, const N: usize> Vector<T, N> {
    /// Create a vector from a 1D array.
    /// Note that vectors are always column vectors unless explicitly instantiated as row vectors
    ///
    /// # Examples
    /// ```
    /// # use vector_victor::{Matrix, Vector};
    /// let my_vector = Vector::vec([1,2,3,4]);
    /// // is equivalent to
    /// assert_eq!(my_vector, Matrix::mat([[1],[2],[3],[4]]));
    /// ```
    pub fn vec(data: [T; N]) -> Self {
        assert!(N > 0, "Vector must have at least 1 element");
        return Vector::<T, N> {
            data: data.map(|e| [e]),
        };
    }
}

// ACCESSORS AND MUTATORS
impl<T: Copy, const M: usize, const N: usize> Matrix<T, M, N> {
    /// Returns an iterator over the elements of the matrix in row-major order.
    ///
    /// # Examples
    /// ```
    /// # use vector_victor::Matrix;
    /// let my_matrix = Matrix::mat([[1,2],[3,4]]);
    /// assert!(vec![1,2,3,4].iter().eq(my_matrix.elements()))
    /// ```
    #[must_use]
    pub fn elements<'a>(&'a self) -> impl Iterator<Item = &'a T> + 'a {
        self.data.iter().flatten()
    }

    /// Returns a mutable iterator over the elements of the matrix in row-major order.
    #[must_use]
    pub fn elements_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut T> + 'a {
        self.data.iter_mut().flatten()
    }

    /// returns an iterator over the elements along the diagonal of a matrix
    #[must_use]
    pub fn diagonals<'s>(&'s self) -> impl Iterator<Item = T> + 's {
        (0..min(N, M)).map(|n| self[(n, n)])
    }

    /// Returns an iterator over the elements directly below the diagonal of a matrix
    #[must_use]
    pub fn subdiagonals<'s>(&'s self) -> impl Iterator<Item = T> + 's {
        (0..min(N, M) - 1).map(|n| self[(n, n + 1)])
    }

    /// Returns a reference to the element at that position in the matrix, or `None` if out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vector_victor::Matrix;
    /// let my_matrix = Matrix::mat([[1,2],[3,4]]);
    ///
    /// // element at index 2 is the same as the element at (row 1, column 0).
    /// assert_eq!(my_matrix.get(2), my_matrix.get((1,0)));
    ///
    /// // my_matrix.get() is equivalent to my_matrix[],
    /// // but returns an Option instead of panicking
    /// assert_eq!(my_matrix.get(2), Some(&my_matrix[2]));
    ///
    /// // index 4 is out of range, so get(4) returns None.
    /// assert_eq!(my_matrix.get(4), None);
    /// ```
    #[inline]
    #[must_use]
    pub fn get(&self, index: impl Index2D) -> Option<&T> {
        let (m, n) = index.to_2d(M, N)?;
        Some(&self.data[m][n])
    }

    /// Returns a mutable reference to the element at that position in the matrix, or `None` if out of bounds.
    #[inline]
    #[must_use]
    pub fn get_mut(&mut self, index: impl Index2D) -> Option<&mut T> {
        let (m, n) = index.to_2d(M, N)?;
        Some(&mut self.data[m][n])
    }

    /// Returns a row of the matrix. or [None] if index is out of bounds
    ///
    /// # Examples
    ///
    /// ```
    /// # use vector_victor::{Matrix, Vector};
    /// let my_matrix = Matrix::mat([[1,2],[3,4]]);
    ///
    /// // row at index 1
    /// assert_eq!(my_matrix.row(1), Vector::vec([3,4]));
    /// ```
    #[inline]
    #[must_use]
    pub fn row(&self, m: usize) -> Vector<T, N> {
        assert!(
            m < M,
            "Row index {} out of bounds for {}x{} matrix",
            m,
            M,
            N
        );
        Vector::<T, N>::vec(self.data[m])
    }

    #[inline]
    pub fn set_row(&mut self, m: usize, val: &Vector<T, N>) {
        assert!(
            m < M,
            "Row index {} out of bounds for {}x{} matrix",
            m,
            M,
            N
        );
        for n in 0..N {
            self.data[m][n] = val.data[n][0];
        }
    }

    pub fn pivot_row(&mut self, m1: usize, m2: usize) {
        let tmp = self.row(m2);
        self.set_row(m2, &self.row(m1));
        self.set_row(m1, &tmp);
    }

    #[inline]
    #[must_use]
    pub fn col(&self, n: usize) -> Vector<T, M> {
        assert!(
            n < N,
            "Column index {} out of bounds for {}x{} matrix",
            n,
            M,
            N
        );
        Vector::<T, M>::vec(self.data.map(|r| r[n]))
    }

    #[inline]
    pub fn set_col(&mut self, n: usize, val: &Vector<T, M>) {
        assert!(
            n < N,
            "Column index {} out of bounds for {}x{} matrix",
            n,
            M,
            N
        );

        for m in 0..M {
            self.data[m][n] = val.data[m][0];
        }
    }

    pub fn pivot_col(&mut self, n1: usize, n2: usize) {
        let tmp = self.col(n2);
        self.set_col(n2, &self.col(n1));
        self.set_col(n1, &tmp);
    }

    #[must_use]
    pub fn rows<'a>(&'a self) -> impl Iterator<Item = Vector<T, N>> + 'a {
        (0..M).map(|m| self.row(m))
    }

    #[must_use]
    pub fn cols<'a>(&'a self) -> impl Iterator<Item = Vector<T, M>> + 'a {
        (0..N).map(|n| self.col(n))
    }

    #[must_use]
    pub fn permute_rows(&self, ms: &Vector<usize, M>) -> Self
    where
        T: Default,
    {
        Self::from_rows(ms.elements().map(|&m| self.row(m)))
    }

    #[must_use]
    pub fn permute_cols(&self, ns: &Vector<usize, N>) -> Self
    where
        T: Default,
    {
        Self::from_cols(ns.elements().map(|&n| self.col(n)))
    }

    pub fn transpose(&self) -> Matrix<T, N, M>
    where
        Matrix<T, N, M>: Default,
    {
        Matrix::<T, N, M>::from_rows(self.cols())
    }
}

// Index
impl<I, T, const M: usize, const N: usize> Index<I> for Matrix<T, M, N>
where
    I: Index2D,
    T: Copy,
{
    type Output = T;

    #[inline(always)]
    fn index(&self, index: I) -> &Self::Output {
        self.get(index).expect(&*format!(
            "index {:?} out of range for {}x{} Matrix",
            index, M, N
        ))
    }
}

// IndexMut
impl<I, T, const M: usize, const N: usize> IndexMut<I> for Matrix<T, M, N>
where
    I: Index2D,
    T: Copy,
{
    #[inline(always)]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        self.get_mut(index).expect(&*format!(
            "index {:?} out of range for {}x{} Matrix",
            index, M, N
        ))
    }
}

// CONVERSIONS

// Convert from 2D Array (equivalent to new)
impl<T: Copy, const M: usize, const N: usize> From<[[T; N]; M]> for Matrix<T, M, N> {
    fn from(data: [[T; N]; M]) -> Self {
        Self::mat(data)
    }
}

// Convert from 1D Array (equivalent to vec)
impl<T: Copy, const M: usize> From<[T; M]> for Vector<T, M> {
    fn from(data: [T; M]) -> Self {
        Self::vec(data)
    }
}

// Convert from scalar (equivalent to fill)
impl<T: Copy, const M: usize, const N: usize> From<T> for Matrix<T, M, N> {
    fn from(scalar: T) -> Self {
        Self::fill(scalar)
    }
}

// IntoIter
impl<T: Copy, const M: usize, const N: usize> IntoIterator for Matrix<T, M, N> {
    type Item = T;
    type IntoIter = Flatten<std::array::IntoIter<[T; N], M>>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter().flatten()
    }
}

// FromIterator
impl<T: Copy, const M: usize, const N: usize> FromIterator<T> for Matrix<T, M, N>
where
    Self: Default,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut result: Self = Default::default();
        for (l, r) in zip(result.elements_mut(), iter) {
            *l = r;
        }
        result
    }
}
