use crate::Matrix;
use num_traits::{Bounded, One, Zero};

// Identity
impl<T: Copy + Zero + One, const N: usize> Matrix<T, N, N> {
    /// Create an identity matrix, a square matrix where the diagonals are 1 and all other elements
    /// are 0.
    /// for example,
    ///
    /// $bbI = [[1,0,0],[0,1,0],[0,0,1]]$
    ///
    /// Matrix multiplication between a matrix and the identity matrix always results in itself
    ///
    /// $bbA xx bbI = bbA$
    ///
    /// # Examples
    /// ```
    /// # use vector_victor::Matrix;
    /// let i = Matrix::<i32,3,3>::identity();
    /// assert_eq!(i, Matrix::mat([[1,0,0],[0,1,0],[0,0,1]]))
    /// ```
    ///
    /// Note that the identity only exists for matrices that are square, so this doesnt work:
    /// ```compile_fail
    /// # use vector_victor::Matrix;
    /// let i = Matrix::<i32,4,2>::identity();
    /// ```
    #[must_use]
    pub fn identity() -> Self {
        let mut result = Self::zero();
        for i in 0..N {
            result[(i, i)] = T::one();
        }
        return result;
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
