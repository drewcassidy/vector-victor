use crate::impl_matrix_op;
use crate::index::Index2D;

use num_traits::real::Real;
use num_traits::{Num, NumOps, One, Signed, Zero};
use std::fmt::Debug;
use std::iter::{zip, Flatten, Product, Sum};

use std::ops::{Add, AddAssign, Deref, DerefMut, Index, IndexMut, Mul, MulAssign, Neg, Sub};

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

pub trait MatrixLike {
    type Scalar: Copy;
    const WIDTH: usize;
    const HEIGHT: usize;
}
impl<T: Copy, const M: usize, const N: usize> MatrixLike for Matrix<T, M, N> {
    type Scalar = T;
    const WIDTH: usize = N;
    const HEIGHT: usize = M;
}

pub trait SquareMatrix {}
impl<T: Copy, const M: usize> SquareMatrix for Matrix<T, M, M> {}

// pub trait Dot<R>: MatrixLike {
//     #[must_use]
//     fn dot(&self, rhs: &R) -> <Self as MatrixLike>::Scalar;
// }
//
// pub trait Cross<R> {
//     #[must_use]
//     fn cross_r(&self, rhs: &R) -> Self;
//     #[must_use]
//     fn cross_l(&self, rhs: &R) -> Self;
// }
//
// pub trait MMul<R> {
//     type Output;
//     #[must_use]
//     fn mmul(&self, rhs: &R) -> Self::Output;
// }

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
    pub fn row(&self, m: usize) -> Option<Vector<T, N>> {
        if m < M {
            Some(Vector::<T, N>::vec(self.data[m]))
        } else {
            None
        }
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

    pub fn pivot_row(&mut self, m1: usize, m2: usize) {
        let tmp = self.row(m2).expect("Invalid row index");
        self.set_row(m2, &self.row(m1).expect("Invalid row index"));
        self.set_row(m1, &tmp);
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

    pub fn pivot_col(&mut self, n1: usize, n2: usize) {
        let tmp = self.col(n2).expect("Invalid column index");
        self.set_col(n2, &self.col(n1).expect("Invalid column index"));
        self.set_col(n1, &tmp);
    }

    #[must_use]
    pub fn rows<'a>(&'a self) -> impl Iterator<Item = Vector<T, N>> + 'a {
        (0..M).map(|m| self.row(m).expect("invalid row reached while iterating"))
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

    pub fn abs(&self) -> Self
    where
        T: Default + PartialOrd + Zero + Neg<Output = T>,
    {
        self.elements()
            .map(|&x| match x > T::zero() {
                true => x,
                false => -x,
            })
            .collect()
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

    pub fn dot<R>(&self, rhs: &R) -> T
    where
        for<'s> &'s Self: Mul<&'s R, Output = Self>,
        T: Sum<T>,
    {
        (self * rhs).elements().cloned().sum()
    }
}

// Cross Product
impl<T: Copy> Vector<T, 3> {
    pub fn cross_r<R: Copy>(&self, rhs: &Vector<R, 3>) -> Self
    where
        T: NumOps<R> + NumOps,
    {
        Self::vec([
            (self[1] * rhs[2]) - (self[2] * rhs[1]),
            (self[2] * rhs[0]) - (self[0] * rhs[2]),
            (self[0] * rhs[1]) - (self[1] * rhs[0]),
        ])
    }

    pub fn cross_l<R: Copy>(&self, rhs: &Vector<R, 3>) -> Self
    where
        T: NumOps<R> + NumOps + Neg<Output = T>,
    {
        -self.cross_r(rhs)
    }
}

//Matrix Multiplication
impl<T: Copy, const M: usize, const N: usize> Matrix<T, M, N> {
    pub fn mmul<R: Copy, const P: usize>(&self, rhs: &Matrix<R, N, P>) -> Matrix<T, M, P>
    where
        T: Default + NumOps<R> + Sum,
    {
        let mut result: Matrix<T, M, P> = Default::default();

        for (m, a) in self.rows().enumerate() {
            for (n, b) in rhs.cols().enumerate() {
                result[(m, n)] = a.dot(&b)
            }
        }

        return result;
    }
}

// Square matrix impls
impl<T: Copy, const N: usize> Matrix<T, N, N> {
    #[must_use]
    pub fn identity() -> Self
    where
        T: Zero + One,
    {
        let mut result = Self::zero();
        for i in 0..N {
            result[(i, i)] = T::one();
        }
        return result;
    }

    #[must_use]
    pub fn diagonals<'s>(&'s self) -> impl Iterator<Item = T> + 's {
        (0..N).map(|n| self[(n, n)])
    }

    #[must_use]
    pub fn subdiagonals<'s>(&'s self) -> impl Iterator<Item = T> + 's {
        (0..N - 1).map(|n| self[(n, n + 1)])
    }

    #[must_use]
    ///
    /// <math xmlns="http://www.w3.org/1998/Math/MathML" alttext="a^{3}" display="block">
    ///   <msup>
    ///     <mi>a</mi>
    ///     <mn>3</mn>
    ///   </msup>
    /// </math>
    pub fn lu(&self) -> Option<(Self, Vector<usize, N>, T)>
    where
        T: Real + Default,
    {
        // Implementation from Numerical Recipes §2.3
        let mut lu = self.clone();
        let mut idx: Vector<usize, N> = Default::default();
        let mut d = T::one();

        let mut vv: Vector<T, N> = self
            .rows()
            .map(|row| {
                let m = row.elements().cloned().reduce(|acc, x| acc.max(x.abs()))?;
                match m < T::epsilon() {
                    true => None,
                    false => Some(T::one() / m),
                }
            })
            .collect::<Option<_>>()?; // get the inverse maxabs value in each row

        for k in 0..N {
            // search for the pivot element and its index
            let (ipivot, _) = (lu.col(k)? * vv)
                .abs()
                .elements()
                .enumerate()
                .skip(k) // below the diagonal
                .reduce(|(imax, xmax), (i, x)| match x > xmax {
                    // Is the figure of merit for the pivot better than the best so far?
                    true => (i, x),
                    false => (imax, xmax),
                })?;

            // do we need to interchange rows?
            if k != ipivot {
                lu.pivot_row(ipivot, k); // yes, we do
                d = -d; // change parity of d
                vv[ipivot] = vv[k] //interchange scale factor
            }

            idx[k] = ipivot;
            let pivot = lu[(k, k)];
            if pivot.abs() < T::epsilon() {
                // if the pivot is zero, the matrix is singular
                return None;
            };

            for i in (k + 1)..N {
                // divide by the pivot element
                let dpivot = lu[(i, k)] / pivot;
                lu[(i, k)] = dpivot;
                for j in (k + 1)..N {
                    // reduce remaining submatrix
                    lu[(i, j)] = lu[(i, j)] - (dpivot * lu[(k, j)]);
                }
            }
        }

        return Some((lu, idx, d));
    }

    #[must_use]
    pub fn inverse(&self) -> Option<Self>
    where
        T: Real + Default + Sum,
    {
        self.solve(&Self::identity())
    }

    #[must_use]
    pub fn det(&self) -> T
    where
        T: Real + Default + Product + Sum,
    {
        match N {
            0 => T::zero(),
            1 => self[0],
            2 => (self[(0, 0)] * self[(1, 1)]) - (self[(0, 1)] * self[(1, 0)]),
            3 => {
                // use rule of Sarrus
                (0..N) // starting column
                    .map(|i| {
                        let dn = (0..N)
                            .map(|j| -> T { self[(j, (j + i) % N)] })
                            .product::<T>();
                        let up = (0..N)
                            .map(|j| -> T { self[(N - j - 1, (j + i) % N)] })
                            .product::<T>();
                        dn - up
                    })
                    .sum::<T>()
            }
            _ => {
                // use LU decomposition
                if let Some((lu, _, d)) = self.lu() {
                    d * lu.diagonals().product()
                } else {
                    T::zero()
                }
            }
        }
    }

    #[must_use]
    pub fn solve<const M: usize>(&self, b: &Matrix<T, N, M>) -> Option<Matrix<T, N, M>>
    where
        T: Real + Default + Sum,
    {
        Some(self.lu()?.solve(b))
    }
}

pub trait LUSolve<R>: Copy {
    fn solve(&self, rhs: &R) -> R;
}

impl<T: Copy, const N: usize, const M: usize> LUSolve<Matrix<T, N, M>>
    for (Matrix<T, N, N>, Vector<usize, N>, T)
where
    for<'t> T: Real + Default + Sum,
{
    #[must_use]
    fn solve(&self, b: &Matrix<T, N, M>) -> Matrix<T, N, M> {
        let (lu, idx, _) = self;
        Matrix::<T, N, M>::from_cols(b.cols().map(|mut x| {
            // Implementation from Numerical Recipes §2.3

            // When ii is set to a positive value,
            // it will become the index of the first nonvanishing element of b
            let mut ii = 0usize;
            for i in 0..N {
                // forward substitution
                let ip = idx[i]; // i permuted
                let sum = x[ip];
                x[ip] = x[i]; // unscramble as we go
                if ii > 0 {
                    x[i] = sum
                        - (lu.row(i).expect("Invalid row reached") * x)
                            .elements()
                            .skip(ii - 1)
                            .cloned()
                            .sum()
                } else {
                    x[i] = sum;
                    if sum.abs() > T::epsilon() {
                        ii = i + 1;
                    }
                }
            }
            for i in (0..(N - 1)).rev() {
                // back substitution
                let sum = x[i]
                    - (lu.row(i).expect("Invalid row reached") * x)
                        .elements()
                        .skip(ii - 1)
                        .cloned()
                        .sum();

                x[i] = sum / lu[(i, i)]
            }
            x
        }))
    }
}

// Square matrices
impl<T: Copy, const N: usize> Matrix<T, N, N> {}

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
    Self: Zero + Add<Output = Self>,
{
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), Self::add)
    }
}

impl<T: Copy + MulAssign, const M: usize, const N: usize> Product for Matrix<T, M, N>
where
    Self: One + Mul<Output = Self>,
{
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), Self::mul)
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
