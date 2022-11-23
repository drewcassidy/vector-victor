use crate::impl_matrix_op;
use crate::index::Index2D;
use itertools::Itertools;
use num_traits::{Num, One, Zero};
use std::fmt::Debug;
use std::iter::{zip, Flatten, Product, Sum};
use std::ops::{Add, AddAssign, Deref, DerefMut, Index, IndexMut, Mul, MulAssign, Neg, Sub};
/// A Scalar that a [Matrix] can be made up of.
///
/// This trait has no associated functions and can be implemented on any type that is [Default] and
/// [Copy] and has a static lifetime.
// pub trait Scalar: Default + Copy + 'static {}
// macro_rules! multi_impl { ($name:ident for $($t:ty),*) => ($( impl $name for $t {} )*) }
// multi_impl!(Scalar for i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f32, f64);
// impl<T> Scalar for &'static T
// where
//     T: Scalar,
//     &'static T: Default,
// {
// }

/// A 2D array of values which can be operated upon.
///
/// Matrices have a fixed size known at compile time, and must be made up of types that implement
/// the [Scalar] trait.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Matrix<T, const M: usize, const N: usize>
where
    T: Copy,
{
    data: [[T; N]; M], // Row-Major order
}

/// An alias for a [Matrix] with a single column
pub type Vector<T, const N: usize> = Matrix<T, N, 1>;

pub trait MatrixDepth {
    const DEPTH: usize = 1;
}

pub trait Dot<R> {
    type Output;
    #[must_use]
    fn dot(&self, rhs: &R) -> Self::Output;
}

pub trait Cross<R> {
    #[must_use]
    fn cross_r(&self, rhs: &R) -> Self;
    #[must_use]
    fn cross_l(&self, rhs: &R) -> Self;
}

pub trait MMul<R> {
    type Output;
    #[must_use]
    fn mmul(&self, rhs: &R) -> Self::Output;
}

pub trait Identity {
    #[must_use]
    fn identity() -> Self;
}

pub trait Determinant {
    type Output;
    #[must_use]
    fn determinant(&self) -> Self::Output;
}

// Simple access functions that only require T be copyable
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
    /// let a = Matrix::new([[1,2,3,4];4]);
    /// ```
    #[must_use]
    pub fn new(data: [[T; N]; M]) -> Self {
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
    /// assert_eq!(my_matrix, Matrix::new([[5;4];4]))
    /// ```
    #[must_use]
    pub fn fill(scalar: T) -> Matrix<T, M, N> {
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
    /// let my_matrix = Matrix::new([[1,2,3],[4,5,6]]);
    /// let transpose : Matrix<_,3,2>= Matrix::from_rows(my_matrix.cols());
    /// assert_eq!(transpose, Matrix::new([[1,4],[2,5],[3,6]]))
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
    /// let my_matrix = Matrix::new([[1,2,3],[4,5,6]]);
    /// let transpose : Matrix<_,3,2>= Matrix::from_cols(my_matrix.rows());
    /// assert_eq!(transpose, Matrix::new([[1,4],[2,5],[3,6]]))
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

    /// Returns an iterator over the elements of the matrix in row-major order.
    ///
    /// # Examples
    /// ```
    /// # use vector_victor::Matrix;
    /// let my_matrix = Matrix::new([[1,2],[3,4]]);
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

    /// Returns a reference to the element at that position in the matrix, or `None` if out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vector_victor::Matrix;
    /// let my_matrix = Matrix::new([[1,2],[3,4]]);
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

    /// Returns a row of the matrix. panics if index is out of bounds
    ///
    /// # Examples
    ///
    /// ```
    /// # use vector_victor::{Matrix, Vector};
    /// let my_matrix = Matrix::new([[1,2],[3,4]]);
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
        for (n, v) in val.elements().enumerate() {
            self.data[m][n] = *v;
        }
    }

    #[inline]
    #[must_use]
    pub fn col(&self, n: usize) -> Option<Vector<T, M>> {
        if n < N {
            Some(Vector::<T, M>::vec(self.data.map(|r| r[n])))
        } else {
            None
        }
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

        for (m, v) in val.elements().enumerate() {
            self.data[m][n] = *v;
        }
    }

    #[must_use]
    pub fn rows<'a>(&'a self) -> impl Iterator<Item = Vector<T, N>> + 'a {
        (0..M).map(|m| self.row(m))
    }

    #[must_use]
    pub fn cols<'a>(&'a self) -> impl Iterator<Item = Vector<T, M>> + 'a {
        (0..N).map(|n| self.col(n).expect("invalid column reached while iterating"))
    }

    pub fn transpose(&self) -> Matrix<T, N, M>
    where
        Matrix<T, N, M>: Default,
    {
        Matrix::<T, N, M>::from_rows(self.cols())
    }

    // pub fn mmul<const P: usize, R, O>(&self, rhs: &Matrix<R, P, N>) -> Matrix<T, P, M>
    // where
    //     R: Num,
    //     T: Scalar + Mul<R, Output = T>,
    //     Vector<T, N>: Dot<Vector<R, M>, Output = T>,
    // {
    //     let mut result: Matrix<T, P, M> = Zero::zero();
    //
    //     for (m, a) in self.rows().enumerate() {
    //         for (n, b) in rhs.cols().enumerate() {
    //             // result[(m, n)] = a.dot(b)
    //         }
    //     }
    //
    //     return result;
    // }
}

// 1D vector implementations
impl<T: Copy, const M: usize> Vector<T, M> {
    /// Create a vector from a 1D array.
    /// Note that vectors are always column vectors unless explicitly instantiated as row vectors
    ///
    /// # Arguments
    ///
    /// * `data`: A 1D array of elements to copy into the new vector
    ///
    /// returns: Vector<T, M>
    ///
    /// # Examples
    ///
    /// ```
    /// # use vector_victor::{Matrix, Vector};
    /// let my_vector = Vector::vec([1,2,3,4]);
    /// // is equivalent to
    /// assert_eq!(my_vector, Matrix::new([[1],[2],[3],[4]]));
    /// ```
    pub fn vec(data: [T; M]) -> Self {
        return Vector::<T, M> {
            data: data.map(|e| [e]),
        };
    }
}

impl<T: Copy, R: Copy, const M: usize> Dot<Vector<R, M>> for Vector<T, M>
where
    for<'a> T: Sum<&'a T>,
    for<'b> &'b Self: Mul<&'b Vector<R, M>, Output = Self>,
{
    type Output = T;
    fn dot(&self, rhs: &Matrix<R, M, 1>) -> Self::Output {
        (self * rhs).elements().sum::<Self::Output>()
    }
}

impl<T: Copy, R: Copy> Cross<Vector<R, 3>> for Vector<T, 3>
where
    T: Mul<R, Output = T> + Sub<T, Output = T>,
    Self: Neg<Output = Self>,
{
    fn cross_r(&self, rhs: &Vector<R, 3>) -> Self {
        Self::vec([
            (self[1] * rhs[2]) - (self[2] * rhs[1]),
            (self[2] * rhs[0]) - (self[0] * rhs[2]),
            (self[0] * rhs[1]) - (self[1] * rhs[0]),
        ])
    }

    fn cross_l(&self, rhs: &Vector<R, 3>) -> Self {
        -self.cross_r(rhs)
    }
}

//Matrix Multiplication
impl<T: Copy, R: Copy, const M: usize, const N: usize, const P: usize> MMul<Matrix<R, N, P>>
    for Matrix<T, M, N>
where
    T: Default,
    Vector<T, N>: Dot<Vector<R, N>, Output = T>,
{
    type Output = Matrix<T, M, P>;

    fn mmul(&self, rhs: &Matrix<R, N, P>) -> Self::Output {
        let mut result = Self::Output::default();

        for (m, a) in self.rows().enumerate() {
            for (n, b) in rhs.cols().enumerate() {
                result[(m, n)] = a.dot(&b)
            }
        }

        return result;
    }
}

// Identity
impl<T: Copy + Zero + One, const M: usize> Identity for Matrix<T, M, M> {
    fn identity() -> Self {
        let mut result = Self::zero();
        for i in 0..M {
            result[(i, i)] = T::one();
        }
        return result;
    }
}

// Determinant
impl<T: Copy, const M: usize> Determinant for Matrix<T, M, M>
where
    for<'a> T: Product<T> + Sum<T> + Mul<&'a i32, Output = T>,
{
    type Output = T;

    fn determinant(&self) -> Self::Output {
        // Leibniz formula

        // alternating 1,-1,1,-1...
        let signs = [1, -1].iter().cycle();
        // all permutations of 0..M
        let column_permutations = (0..M).permutations(M);

        // Calculating the determinant is done by summing up M steps,
        // each with a different permutation of columns and alternating sign
        // Each step involves a product of the components being operated on
        let summand = |(columns, sign)| -> T {
            zip(0..M, columns).map(|(r, c)| self[(r, c)]).product::<T>() * sign
        };

        // Now sum up all steps
        zip(column_permutations, signs).map(summand).sum()
    }
}

// Index
impl<I, T, const M: usize, const N: usize> Index<I> for Matrix<T, M, N>
where
    I: Index2D,
    T: Copy,
{
    type Output = T;

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
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        self.get_mut(index).expect(&*format!(
            "index {:?} out of range for {}x{} Matrix",
            index, M, N
        ))
    }
}
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

impl<T: Copy, const M: usize, const N: usize> From<[[T; N]; M]> for Matrix<T, M, N> {
    fn from(data: [[T; N]; M]) -> Self {
        Self::new(data)
    }
}

impl<T: Copy, const M: usize> From<[T; M]> for Vector<T, M> {
    fn from(data: [T; M]) -> Self {
        Self::vec(data)
    }
}

impl<T: Copy, const M: usize, const N: usize> From<T> for Matrix<T, M, N> {
    fn from(scalar: T) -> Self {
        Self::fill(scalar)
    }
}

// deref 1x1 matrices to a scalar automatically
impl<T: Copy> Deref for Matrix<T, 1, 1> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.data[0][0]
    }
}

// deref 1x1 matrices to a mutable scalar automatically
impl<T: Copy> DerefMut for Matrix<T, 1, 1> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data[0][0]
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

impl<T: Copy + AddAssign, const M: usize, const N: usize> Sum for Matrix<T, M, N>
where
    Self: Zero + AddAssign,
{
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut sum = Self::zero();

        for m in iter {
            sum += m;
        }

        sum
    }
}

impl<T: Copy + MulAssign, const M: usize, const N: usize> Product for Matrix<T, M, N>
where
    Self: One + MulAssign,
{
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut prod = Self::one();

        for m in iter {
            prod *= m;
        }

        prod
    }
}

impl_matrix_op!(neg);
impl_matrix_op!(!);
impl_matrix_op!(+);
impl_matrix_op!(-);
impl_matrix_op!(*);
impl_matrix_op!(/);
impl_matrix_op!(%);
impl_matrix_op!(&);
impl_matrix_op!(|);
impl_matrix_op!(^);
impl_matrix_op!(<<);
impl_matrix_op!(>>);
