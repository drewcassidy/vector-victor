// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

mod index;
pub mod legacy;
mod math;
mod ops;
pub mod splat;

extern crate core;

use num_traits::{One, Zero};
use std::fmt::Debug;
use std::iter::zip;
use std::ops::{Add, Deref, DerefMut, Index, IndexMut, Mul};

pub use legacy::{Matrix, Vector};
pub use splat::{Scalar, Splat};

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct Col<T: Copy, const N: usize> {
    data: [T; N],
}

pub type Mat<T, const H: usize, const W: usize> = Col<Col<T, W>, H>;

impl<T: Copy, const N: usize> From<[T; N]> for Col<T, N> {
    fn from(value: [T; N]) -> Self {
        Self { data: value }
    }
}

impl<T: Copy, const N: usize> Deref for Col<T, N> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T: Copy, const N: usize> DerefMut for Col<T, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<T: Copy, const H: usize, const W: usize> From<[[T; W]; H]> for Mat<T, H, W> {
    fn from(value: [[T; W]; H]) -> Self {
        Self::from(value.map(Col::<T, W>::from))
    }
}

impl<T: Copy + Default, const N: usize> Default for Col<T, N> {
    fn default() -> Self {
        Self::from([T::default(); N])
    }
}

impl<T: Copy + One + Add<T, Output = T>, const N: usize> One for Col<T, N>
where
    Col<T, N>: Mul<Col<T, N>, Output = Col<T, N>>,
{
    fn one() -> Self {
        Self::from([T::one(); N])
    }
}

impl<T: Copy + Zero, const N: usize> Zero for Col<T, N>
where
    Col<T, N>: Add<Col<T, N>, Output = Col<T, N>>,
{
    fn zero() -> Self {
        Self::from([T::zero(); N])
    }

    fn is_zero(&self) -> bool {
        self.rows().all(|r| r.is_zero())
    }
}

impl<T: Copy, const N: usize> IntoIterator for Col<T, N> {
    type Item = T;
    type IntoIter = <[T; N] as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<T: Copy, const N: usize> Col<T, N> {
    /// Returns a vector with the same size as `self`, but with function `F` applied to each row
    #[must_use]
    pub fn map<F, U>(self, f: F) -> Col<U, N>
    where
        F: FnMut(T) -> U,
        U: Copy,
    {
        Col::<U, N> {
            data: self.data.map(f),
        }
    }

    /// Returns a vector with the same size as `self`, but with function `F` applied to the corresponding elements of `self` and `rhs`
    #[must_use]
    pub fn zip<F, U, R>(self, rhs: Col<R, N>, mut f: F) -> Col<U, N>
    where
        F: FnMut(T, R) -> U,
        U: Copy + Splat<U>,
        R: Copy,
    {
        Col::<U, N>::try_from_rows(zip(self.rows(), rhs.rows()).map(|(l, r)| f(l, r))).unwrap()
    }

    /** Returns one row of the column.

    # Panics

    Panics if row index `n` is out of bounds.

    # Examples

    Getting a row of a column:

    ```
    # use vector_victor::{Mat, Col};
    let my_vector = Col::from([5,6,7,8]);

    // row at index 1
    assert_eq!(my_vector.row(2),7);
    ```

    Getting a row of a matrix:

    ```
    # use vector_victor::{Mat, Col};
    let my_matrix = Mat::from([[1, 2], [3, 4]]);

    // row at index 1
    assert_eq!(my_matrix.row(1), Col::from([3,4]));
    ``` */
    #[inline]
    #[must_use]
    pub fn row(&self, n: usize) -> T {
        assert!(n < N, "Row index {n} out of bounds for {N}-element column",);
        self[n]
    }

    /** Sets one row of a column.

    # Panics

    Panics if row index `n` is out of bounds.

    # Examples

    Setting a row of a column:
    ```
    # use vector_victor::Col;
    let mut my_vector = Col::from([5,6,7,8]);

    // row at index 2
    my_vector.set_row(2, 9);
    assert_eq!(my_vector, Col::from([5,6,9,8]));
    ```

    Setting a row of a matrix:
    ```
    # use vector_victor::{Mat, Col};
    let mut my_matrix = Mat::from([[1, 2], [3, 4]]);

    // row at index 1
    my_matrix.set_row(1, Col::from([5,6]));
    assert_eq!(my_matrix, Mat::from([[1,2],[5,6]]));
    ```

    Setting a row of a matrix to a scalar:
    ```
    # use vector_victor::{Mat, Col};
    let mut my_matrix = Mat::from([[1, 2], [3, 4]]);

    // row at index 1
    my_matrix.set_row(1, 7);
    assert_eq!(my_matrix, Mat::from([[1,2],[7,7]]));
    ```
    */
    #[inline]
    pub fn set_row<V: Splat<T>>(&mut self, n: usize, val: V) {
        assert!(n < N, "Row index {n} out of bounds for {N}-element column",);
        self[n] = val.splat()
    }

    /// Returns an iterator over the rows of the column
    #[must_use]
    pub fn rows(self) -> <Self as IntoIterator>::IntoIter {
        self.into_iter()
    }

    /** Create a column from an iterator of rows.
    Returns `None` if the iterator does not have exactly `N` items.

    # Arguments

    * `iter`: iterator of items to copy into rows. Item type must be splattable to `T`.
    **/
    #[must_use]
    pub fn try_from_rows(iter: impl IntoIterator<Item: Splat<T>>) -> Option<Self> {
        Some(Self::from(
            <[T; N]>::try_from(iter.into_iter().map(Splat::splat).collect::<Vec<_>>()).ok()?,
        ))
    }

    /** Create a column from an iterator of rows.

    Iterator items beyond the size of the column will be ignored.
    Column items not assigned to will be `T::default()`

    # Arguments

    * `iter`: iterator of items to copy into rows. Item type must be splattable to `T`

    # Examples

    ```
    # use vector_victor::Col;
    let increment =  Col::<_,8>::from_rows(0..5);

    assert_eq!(increment, Col::from([0,1,2,3,4,0,0,0]));
    ``` */
    pub fn from_rows<R: Splat<T>>(iter: impl IntoIterator<Item = R>) -> Self
    where
        Self: Default,
    {
        let mut ret = Self::default();
        for (n, row) in zip(0..N, iter.into_iter()) {
            ret.set_row(n, row)
        }
        ret
    }
}

impl<T: Copy, const H: usize, const W: usize> Mat<T, H, W> {
    /** Returns one column of a matrix.

    # Panics

    Panics if column index `n` is out of bounds.

    # Examples

    ```
    # use vector_victor::{Mat, Col};
    let my_matrix = Mat::from([[1, 2], [3, 4]]);

    // column at index 1
    assert_eq!(my_matrix.column(1), Col::from([2,4]));
    ``` */
    #[inline]
    #[must_use]
    pub fn column(&self, n: usize) -> Col<T, H> {
        assert!(n < W, "Column index {n} out of bounds for {H}×{W} matrix",);
        self.map(|r| r[n])
    }

    /** Sets one column of a matrix.

    # Panics

    Panics if column index `n` is out of bounds.

    # Examples

    ```
    # use vector_victor::{Mat, Col};
    let mut my_matrix = Mat::from([[1, 2], [3, 4]]);

    // column at index 1
    my_matrix.set_column(1, Col::from([5,6]));
    assert_eq!(my_matrix, Mat::from([[1,5],[3,6]]));
    ``` */
    #[inline]
    pub fn set_column<V: Splat<Col<T, H>>>(&mut self, n: usize, val: V) {
        assert!(n < W, "Column index {n} out of bounds for {H}×{W} matrix",);
        let val = val.splat();
        for m in 0..H {
            self[m][n] = val[m]
        }
    }

    /// Returns an iterator over the columns of a matrix
    pub fn columns<'a>(&'a self) -> impl Iterator<Item = Col<T, H>> + 'a {
        (0..H).map(|n| self.column(n))
    }

    /** Create a matrix from an iterator of columns.

    Iterator items beyond the size of the column will be ignored.
    Columns not assigned to will be `T::default()`

    # Arguments

    * `iter`: iterator of items to copy into columns. Item type must be splattable to `Col<T,H>`

    # Examples

    ```
    # use vector_victor::{Col,Mat};
    let increment =  Mat::<_,4,4>::from_columns(
            (0..4).map(|n| Col::<i32,4>::from_rows(n..)));

    assert_eq!(increment, Mat::from([[0,1,2,3],
                                     [1,2,3,4],
                                     [2,3,4,5],
                                     [3,4,5,6]]));
    ``` */
    pub fn from_columns<R: Splat<Col<T, H>>>(iter: impl IntoIterator<Item = R>) -> Self
    where
        Self: Default,
    {
        let mut ret = Self::default();
        for (n, col) in zip(0..W, iter.into_iter()) {
            ret.set_column(n, col)
        }
        ret
    }
}
