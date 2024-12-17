// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

extern crate core;

use index::MatrixIndex;
use itertools::Itertools;
use num_traits::{Bounded, One, Zero};
use std::cmp::min;
use std::fmt::Debug;
use std::iter::{zip, Flatten};
use std::ops::{Add, Index, IndexMut};

pub mod decompose;
pub mod index;
pub mod math;
pub mod ops;

pub mod swizzle;
pub mod util;

/** A 2D array of values which can be operated upon.

Matrices have a fixed size known at compile time */
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

// Zero
impl<T: Copy + Zero, const M: usize, const N: usize> Zero for Matrix<T, M, N> {
    fn zero() -> Self {
        Matrix::fill(T::zero())
    }

    fn is_zero(&self) -> bool {
        self.elements().all(|e| e.is_zero())
    }
}

// One
impl<T: Copy + One, const M: usize, const N: usize> One for Matrix<T, M, N> {
    fn one() -> Self {
        Matrix::fill(T::one())
    }
}

// min_value and max_value
// LowerBounded and UpperBounded are automatically implemented from this
impl<T: Copy + Bounded, const N: usize, const M: usize> Bounded for Matrix<T, N, M> {
    fn min_value() -> Self {
        Self::fill(T::min_value())
    }

    fn max_value() -> Self {
        Self::fill(T::max_value())
    }
}

// Identity
impl<T: Copy + Zero + One, const N: usize> Matrix<T, N, N> {
    /** Create an identity matrix, a square matrix where the diagonals are 1 and
    all other elements are 0.

    for example,

    $bbI = \[\[1,0,0],\[0,1,0],\[0,0,1]]$

    Matrix multiplication between a matrix and the identity matrix always results in itself

    $bbA xx bbI = bbA$

    # Examples
    ```
    # use vector_victor::Matrix;
    let i = Matrix::<i32,3,3>::identity();
    assert_eq!(i, Matrix::mat([[1, 0, 0],
                               [0, 1, 0],
                               [0, 0, 1]]))
    ```

    Note that the identity only exists for matrices that are square, so this doesnt work:
    ```compile_fail
    # use vector_victor::Matrix;
    let i = Matrix::<i32,4,2>::identity();
    ``` */
    #[must_use]
    pub fn identity() -> Self {
        let mut result = Self::zero();
        for i in 0..N {
            result[(i, i)] = T::one();
        }
        return result;
    }
}

// Matrix constructors
impl<T: Copy, const M: usize, const N: usize> Matrix<T, M, N> {
    /** Generate a new matrix from a 2D Array

    # Arguments

    * `data`: A 2D array of elements to copy into the new matrix

    # Examples

    ```
    # use vector_victor::Matrix;
    let a = Matrix::mat([[1,2,3,4];4]);
    ``` */
    #[must_use]
    pub fn mat(data: [[T; N]; M]) -> Self {
        assert!(M > 0, "Matrix must have at least 1 row");
        assert!(N > 0, "Matrix must have at least 1 column");
        Matrix::<T, M, N> { data }
    }

    /** Generate a new matrix from a single scalar

    # Arguments

    * `scalar`: Scalar value to copy into the new matrix.

    # Examples

    ```
    # use vector_victor::Matrix;
    // these are equivalent
    assert_eq!(Matrix::<i32,4,4>::fill(5), Matrix::mat([[5;4];4]))
    ``` */
    #[must_use]
    pub fn fill(scalar: T) -> Matrix<T, M, N> {
        assert!(M > 0, "Matrix must have at least 1 row");
        assert!(N > 0, "Matrix must have at least 1 column");
        Matrix::<T, M, N> {
            data: [[scalar; N]; M],
        }
    }

    /** Create a matrix from an iterator of vectors

    # Arguments

    * `iter`: iterator of vectors to copy into rows

    # Examples

    The following is another way of performing [`Matrix::transpose()`]
    ```
    # use vector_victor::Matrix;
    let my_matrix = Matrix::mat([[1, 2, 3],
                                 [4, 5, 6]]);

    let transpose : Matrix<_,3,2>= Matrix::from_rows(my_matrix.cols());

    assert_eq!(transpose, Matrix::mat([[1, 4],
                                       [2, 5],
                                       [3, 6]]))
    ``` */
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

    /** Create a matrix from an iterator of vectors

    # Arguments

    * `iter`: iterator of vectors to copy into columns

    # Examples

    The following is another way of performing [`Matrix::transpose()`]
    ```
    # use vector_victor::Matrix;
    let my_matrix = Matrix::mat([[1, 2, 3],
                                 [4, 5, 6]]);

    let transpose : Matrix<_,3,2>= Matrix::from_cols(my_matrix.rows());

    assert_eq!(transpose, Matrix::mat([[1, 4],
                                       [2, 5],
                                       [3, 6]]))
    ``` */
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
    /** Create a vector from a 1D array.
    Note that vectors are always column vectors unless explicitly instantiated as row vectors

    # Examples
    ```
    # use vector_victor::{Matrix, Vector};
    // these are equivalent
    assert_eq!(Vector::vec([1,2,3,4]), Matrix::mat([[1],[2],[3],[4]]));
    ``` */
    pub fn vec(data: [T; N]) -> Self {
        assert!(N > 0, "Vector must have at least 1 element");
        return Vector::<T, N> {
            data: data.map(|e| [e]),
        };
    }
}

// ACCESSORS AND MUTATORS
impl<T: Copy, const M: usize, const N: usize> Matrix<T, M, N> {
    /** Returns an iterator over the elements of the matrix in row-major order.

    This is identical to the behavior of [`IntoIterator`](#associatedtype.IntoIter)

    # Examples
    ```
    # use vector_victor::Matrix;
    let my_matrix = Matrix::mat([[1, 2],
                                 [3, 4]]);

    itertools::assert_equal(my_matrix.elements(), [1,2,3,4].iter())
    ``` */
    #[must_use]
    pub fn elements<'s>(&'s self) -> impl Iterator<Item = &'s T> + 's {
        self.data.iter().flatten()
    }

    /** Returns a mutable iterator over the elements of the matrix in row-major order.

    # Examples
    ```
    # use vector_victor::Matrix;
    let mut my_matrix = Matrix::mat([[1, 2],
                                     [3, 4]]);

    for elem in my_matrix.elements_mut() {*elem += 2;}
    itertools::assert_equal(my_matrix.elements(), [3,4,5,6].iter())
    ``` */
    #[must_use]
    pub fn elements_mut<'s>(&'s mut self) -> impl Iterator<Item = &'s mut T> + 's {
        self.data.iter_mut().flatten()
    }

    /** returns an iterator over the elements along the diagonal of a matrix

    # Examples
    ```
    # use vector_victor::Matrix;
    let my_matrix = Matrix::mat([[1, 2, 3],
                                 [4, 5, 6],
                                 [7, 8, 9],
                                 [10,11,12]]);

    itertools::assert_equal(my_matrix.diagonals(), [1,5,9].iter())
    ``` */
    #[must_use]
    pub fn diagonals<'s>(&'s self) -> impl Iterator<Item = &'s T> + 's {
        (0..min(N, M)).map(|n| &self[(n, n)])
    }

    /** Returns an iterator over the elements directly below the diagonal of a matrix

    # Examples
    ```
    # use vector_victor::Matrix;
    let my_matrix = Matrix::mat([[1, 2, 3],
                                 [4, 5, 6],
                                 [7, 8, 9],
                                 [10,11,12]]);

    itertools::assert_equal(my_matrix.subdiagonals(), [4,8,12].iter());
    ``` */
    #[must_use]
    pub fn subdiagonals<'s>(&'s self) -> impl Iterator<Item = &'s T> + 's {
        (0..min(N, M - 1)).map(|n| &self[(n + 1, n)])
    }

    /** Returns a reference to the element at that position in the matrix, or `None` if out of bounds.

    [`Index`](#impl-Index%3CI%3E-for-Matrix%3CT,+M,+N%3E) behaves similarly,
    but will panic if the index is out of bounds instead of returning an option

    # Arguments

    * `index`: a 1D or 2D index into the matrix. See [MatrixIndex] for more information on matrix indexing.

    # Examples

    ```
    # use vector_victor::Matrix;
    let my_matrix = Matrix::mat([[1, 2],
                                 [3, 4]]);

    // element at index 2 is the same as the element at row 1, column 0.
    assert_eq!(my_matrix.get(2), my_matrix.get((1,0)));

    // my_matrix.get() is equivalent to my_matrix[],
    // but returns an Option instead of panicking
    assert_eq!(my_matrix.get(2), Some(&my_matrix[2]));

    // index 4 is out of range, so get(4) returns None.
    assert_eq!(my_matrix.get(4), None);
    ``` */
    #[inline]
    #[must_use]
    pub fn get(&self, index: impl MatrixIndex) -> Option<&T> {
        let (m, n) = index.to_2d(M, N)?;
        Some(&self.data[m][n])
    }

    /** Returns a mutable reference to the element at that position in the matrix,
    or `None` if out of bounds.

    [`IndexMut`](#impl-IndexMut%3CI%3E-for-Matrix%3CT,+M,+N%3E) behaves similarly,
    but will panic if the index is out of bounds instead of returning an option

    # Arguments

    * `index`: a 1D or 2D index into the matrix. See [MatrixIndex] for more information
    on matrix indexing.

    # Examples

    ```
    # use vector_victor::Matrix;
    let mut my_matrix = Matrix::mat([[1, 2],
                                     [3, 4]]);

    match my_matrix.get_mut(2) {
        Some(t) => *t = 5,
        None => panic!()};
    assert_eq!(my_matrix, Matrix::mat([[1,2],[5,4]]))
    ``` */
    #[inline]
    #[must_use]
    pub fn get_mut(&mut self, index: impl MatrixIndex) -> Option<&mut T> {
        let (m, n) = index.to_2d(M, N)?;
        Some(&mut self.data[m][n])
    }

    /** Returns a row of the matrix.

    # Panics

    Panics if row index `m` is out of bounds.

    # Examples

    ```
    # use vector_victor::{Matrix, Vector};
    let my_matrix = Matrix::mat([[1, 2],
                                 [3, 4]]);

    // row at index 1
    assert_eq!(my_matrix.row(1), Vector::vec([3,4]));
    ``` */
    #[inline]
    #[must_use]
    pub fn row(&self, m: usize) -> Vector<T, N> {
        assert!(
            m < M,
            "Row index {} out of bounds for {}×{} matrix",
            m,
            M,
            N
        );
        Vector::<T, N>::vec(self.data[m])
    }

    /** Sets a row of the matrix.

    # Panics

    Panics if row index `m` is out of bounds.

    # Examples

    ```
    # use vector_victor::{Matrix, Vector};
    let mut my_matrix = Matrix::mat([[1, 2],
                                     [3, 4]]);
    // row at index 1
    my_matrix.set_row(1, &Vector::vec([5,6]));
    assert_eq!(my_matrix, Matrix::mat([[1,2],[5,6]]));
    ``` */
    #[inline]
    pub fn set_row(&mut self, m: usize, val: &Vector<T, N>) {
        assert!(
            m < M,
            "Row index {} out of bounds for {}×{} matrix",
            m,
            M,
            N
        );
        for n in 0..N {
            self.data[m][n] = val.data[n][0];
        }
    }

    /** Returns a column of the matrix.

    # Panics

    Panics if column index `n` is out of bounds.

    # Examples

    ```
    # use vector_victor::{Matrix, Vector};
    let my_matrix = Matrix::mat([[1, 2],
                                 [3, 4]]);

    // column at index 1
    assert_eq!(my_matrix.col(1), Vector::vec([2,4]));
    ``` */
    #[inline]
    #[must_use]
    pub fn col(&self, n: usize) -> Vector<T, M> {
        assert!(
            n < N,
            "Column index {} out of bounds for {}×{} matrix",
            n,
            M,
            N
        );
        Vector::<T, M>::vec(self.data.map(|r| r[n]))
    }

    /** Sets a column of the matrix.

    # Panics

    Panics if column index `n` is out of bounds.

    # Examples

    ```
    # use vector_victor::{Matrix, Vector};
    let mut my_matrix = Matrix::mat([[1, 2],
                                     [3, 4]]);
    // column at index 1
    my_matrix.set_col(1, &Vector::vec([5,6]));
    assert_eq!(my_matrix, Matrix::mat([[1,5],[3,6]]));
    ``` */
    #[inline]
    pub fn set_col(&mut self, n: usize, val: &Vector<T, M>) {
        assert!(
            n < N,
            "Column index {} out of bounds for {}×{} matrix",
            n,
            M,
            N
        );

        for m in 0..M {
            self.data[m][n] = val.data[m][0];
        }
    }

    /// Returns an iterator over the rows of the matrix, returning them as column vectors.
    #[must_use]
    pub fn rows<'a>(&'a self) -> impl Iterator<Item = Vector<T, N>> + 'a {
        (0..M).map(|m| self.row(m))
    }

    /// Returns an iterator over the columns of the matrix, returning them as column vectors.
    #[must_use]
    pub fn cols<'a>(&'a self) -> impl Iterator<Item = Vector<T, M>> + 'a {
        (0..N).map(|n| self.col(n))
    }

    /** Interchange two rows

    # Panics

    Panics if row index `m1` or `m2` are out of bounds */
    pub fn pivot_row(&mut self, m1: usize, m2: usize) {
        let tmp = self.row(m2);
        self.set_row(m2, &self.row(m1));
        self.set_row(m1, &tmp);
    }

    /** Interchange two columns

    # Panics

    Panics if column index `n1` or `n2` are out of bounds */
    pub fn pivot_col(&mut self, n1: usize, n2: usize) {
        let tmp = self.col(n2);
        self.set_col(n2, &self.col(n1));
        self.set_col(n1, &tmp);
    }

    /** Apply a permutation matrix to the rows of a matrix

    # Arguments

    * `ms`: a [`Vector`] of [`usize`] of length P. Each entry is the index of the row that will
    appear in the result

    Returns: a P×N matrix

    # Panics

    Panics if any of the row indices in `ms` is out of bounds

    # Examples

    ```
    # use vector_victor::{Matrix, Vector};
    let my_matrix = Matrix::mat([[1, 2, 3],
                                 [4, 5, 6],
                                 [7, 8, 9]]);

    let permuted = my_matrix.permute_rows(&Vector::vec([1, 0, 2]));
    assert_eq!(permuted, Matrix::mat([[4, 5, 6],
                                      [1, 2, 3],
                                      [7, 8, 9]]))
    ``` */
    #[must_use]
    pub fn permute_rows<const P: usize>(&self, ms: &Vector<usize, P>) -> Matrix<T, P, N>
    where
        T: Default,
    {
        Matrix::<T, P, N>::from_rows(ms.elements().map(|&m| self.row(m)))
    }

    /** Apply a permutation matrix to the columns of a matrix

    # Arguments

    * `ns`: a [`Vector`] of [`usize`] of length P. Each entry is the index of the column that will
    appear in the result

    Returns: a P×N matrix

    # Panics

    Panics if any of the column indices in `ns` is out of bounds */
    #[must_use]
    pub fn permute_cols<const P: usize>(&self, ns: &Vector<usize, P>) -> Matrix<T, M, P>
    where
        T: Default,
    {
        Matrix::<T, M, P>::from_cols(ns.elements().map(|&n| self.col(n)))
    }

    /** Returns the transpose $M^T$ of the matrix, or the matrix flipped across its diagonal.

    # Examples
    ```
    # use vector_victor::Matrix;
    let my_matrix = Matrix::mat([[1, 2, 3],
                                 [4, 5, 6]]);

    assert_eq!(
        my_matrix.transpose(),
        Matrix::mat([[1, 4],
                     [2, 5],
                     [3, 6]]))
    ``` */
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
    I: MatrixIndex,
    T: Copy,
{
    type Output = T;

    #[inline(always)]
    fn index(&self, index: I) -> &Self::Output {
        self.get(index).expect(&*format!(
            "index {:?} out of range for {}×{} Matrix",
            index, M, N
        ))
    }
}

// IndexMut
impl<I, T, const M: usize, const N: usize> IndexMut<I> for Matrix<T, M, N>
where
    I: MatrixIndex,
    T: Copy,
{
    #[inline(always)]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        self.get_mut(index).expect(&*format!(
            "index {:?} out of range for {}×{} Matrix",
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

// Convert to 2D Array
impl<T: Copy + Debug, const M: usize, const N: usize> Into<[[T; N]; M]> for Matrix<T, M, N> {
    fn into(self) -> [[T; N]; M] {
        self.rows()
            .map(|row| row.into())
            .collect_vec()
            .try_into()
            .unwrap()
    }
}

// convert to 1D Array
impl<T: Copy + Debug, const M: usize> Into<[T; M]> for Vector<T, M> {
    fn into(self) -> [T; M] {
        self.elements().cloned().collect_vec().try_into().unwrap()
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
