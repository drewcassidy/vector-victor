use crate::legacy::util::checked_inv;
use crate::{Col, Mat, Matrix, Splat, Vector};
use num_traits::real::Real;
use num_traits::{Num, Signed};
use std::iter::{Product, Sum};
use std::mem::swap;
use std::ops::{Mul, Neg, Not};

/// The parity of an [LU decomposition](LUDecomposition). In other words, how many times the
/// source matrix has to have rows swapped before the decomposition takes place
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Parity {
    Even,
    Odd,
}

impl<T> Mul<T> for Parity
where
    T: Neg<Output = T>,
{
    type Output = T;

    fn mul(self, rhs: T) -> Self::Output {
        match self {
            Parity::Even => rhs,
            Parity::Odd => -rhs,
        }
    }
}

impl Not for Parity {
    type Output = Parity;

    fn not(self) -> Self::Output {
        match self {
            Parity::Even => Parity::Odd,
            Parity::Odd => Parity::Even,
        }
    }
}

/// The result of the [LU decomposition](LUDecompose::lu) of a matrix.
///
/// This struct provides a convenient way to reuse one LU decomposition to solve multiple
/// matrix equations. You likely do not need to worry about its contents.
///
/// See [LU decomposition](https://en.wikipedia.org/wiki/LU_decomposition)
/// on wikipedia for more information
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LUDecomposition<T: Copy, const N: usize> {
    /// The $bbL$ and $bbU$ matrices combined into one
    ///
    /// for example if
    ///
    /// $ bbU = [[u_{11}, u_{12}, cdots,  u_{1n} ],
    ///          [0,      u_{22}, cdots,  u_{2n} ],
    ///          [vdots,  vdots,  ddots,  vdots  ],
    ///          [0,      0,      cdots,  u_{mn} ]] $
    /// and
    /// $ bbL = [[1,      0,      cdots,  0      ],
    ///          [l_{21}, 1,      cdots,  0      ],
    ///          [vdots,  vdots,  ddots,  vdots  ],
    ///          [l_{m1}, l_{m2}, cdots,  1      ]] $,
    /// then
    /// $ bb{LU} = [[u_{11}, u_{12}, cdots,  u_{1n} ],
    ///             [l_{21}, u_{22}, cdots,  u_{2n} ],
    ///             [vdots,  vdots,  ddots,  vdots  ],
    ///             [l_{m1}, l_{m2}, cdots,  u_{mn} ]] $
    ///
    /// note that the diagonals of the $bbL$ matrix are always 1, so no information is lost
    pub lu: Mat<T, N, N>,

    /// The indices of the permutation matrix $bbP$, such that $bbP xx bbA$ = $bbL xx bbU$
    ///
    /// The permutation matrix rearranges the rows of the original matrix in order to produce
    /// the LU decomposition. This makes calculation simpler, but makes the result
    /// (known as an LUP decomposition) no longer unique
    pub idx: Col<usize, N>,

    /// The parity of the decomposition.
    pub parity: Parity,
}

impl<T: Copy + Default + Real, const H: usize> LUDecomposition<T, H> {
    /// Solve for $x$ in $bbM xx x = b$, where $bbM$ is the original matrix this is a decomposition of.
    ///
    /// This is equivalent to [`LUDecompose::solve`] while allowing the LU decomposition
    /// to be reused
    #[must_use]
    pub fn solve<R, const W: usize>(&self, b: R) -> Mat<T, H, W>
    where
        R: Splat<Mat<T, H, W>>,
    {
        let b_permuted = b.splat().permute_rows(&self.idx);

        Mat::from_columns(b_permuted.columns().map(|mut x| {
            // Implementation from Numerical Recipes §2.3
            // When ii is set to a positive value,
            // it will become the index of the first non-vanishing element of b
            let mut ii = 0usize;
            for i in 0..H {
                // forward substitution using L
                let mut sum = x[i];
                ii.ne(0)
                    .then(|mask| {
                        for j in (ii - 1)..i {
                            mask.set(sum, sum - (self.lu[i][j] * x[j]))
                        }
                    })
                    .elif(sum.abs().gt(T::epsilon()))
                    .set(ii, i + 1);

                if ii != 0 {
                    for j in (ii - 1)..i {
                        sum = sum - (self.lu[i][j] * x[j]);
                    }
                } else if sum.abs() > T::epsilon() {
                    ii = i + 1;
                }
                x[i] = sum;
            }
            for i in (0..H).rev() {
                // back substitution using U
                let mut sum = x[i];
                for j in (i + 1)..H {
                    sum = sum - (self.lu[i][j] * x[j]);
                }
                x[i] = sum / self.lu[i][i]
            }
            x
        }))
    }

    /// Calculate the determinant $|M|$ of the matrix $M$.
    /// If the matrix is singular, the determinant is 0.
    ///
    /// This is equivalent to [`LUDecompose::det`] while allowing the LU decomposition
    /// to be reused
    #[must_use]
    pub fn det(&self) -> T {
        self.parity * self.lu.diagonals().fold(T::one(), |l, r| l * r)
    }

    /// Calculate the inverse of the original matrix, such that $bbM xx bbM^{-1} = bbI$
    ///
    /// This is equivalent to [`Matrix::inv`] while allowing the LU decomposition to be reused
    #[must_use]
    pub fn inv(&self) -> Mat<T, H, H> {
        self.solve(Mat::<T, H, H>::identity())
    }

    /// Separate the $L$ and $U$ sides of the $LU$ matrix.
    /// See [the `lu` field](LUDecomposition::lu) for more information
    #[must_use]
    pub fn separate(&self) -> (Mat<T, H, H>, Mat<T, H, H>) {
        let mut l = Mat::<T, H, H>::identity();
        let mut u = self.lu; // lu

        for m in 1..H {
            for n in 0..m {
                // iterate over lower diagonal
                l[m][n] = u[m][n];
                u[m][n] = T::zero();
            }
        }

        (l, u)
    }
}

/// A Matrix that can be decomposed into an upper and lower diagonal matrix,
/// known as an [LU Decomposition](LUDecomposition).
///
/// See [LU decomposition](https://en.wikipedia.org/wiki/LU_decomposition)
/// on wikipedia for more information
pub trait LUDecompose<T: Copy, const N: usize> {
    /// return this matrix's [`crate::legacy::decompose::LUDecomposition`], or [`None`] if the matrix is singular.
    /// This can be used to solve for multiple results
    ///
    /// ```
    /// # use vector_victor::decompose::LUDecompose;
    /// # use vector_victor::{Mat, Col};
    /// let m = Mat::from([[1.0,3.0],[2.0,4.0]]);
    /// let lu = m.lu().expect("Cannot decompose a signular matrix");
    ///
    /// let b = Col::from([7.0,10.0]);
    /// assert_eq!(lu.solve(b), Mat::from([[1.0],[2.0]]));
    ///
    /// let c = Col::from([10.0, 14.0]);
    /// assert_eq!(lu.solve(c), Mat::from([[1.0],[3.0]]));
    ///
    /// ```
    #[must_use]
    fn lu(&self) -> Option<LUDecomposition<T, N>>;

    /// Calculate the inverse of the matrix, such that $bbMxxbbM^{-1} = bbI$,
    /// or [`None`] if the matrix is singular.
    ///
    /// ```
    /// # use vector_victor::decompose::LUDecompose;
    /// # use vector_victor::Matrix;
    /// let m = Matrix::mat([[1.0,3.0],[2.0,4.0]]);
    /// let mi = m.inv().expect("Cannot invert a singular matrix");
    ///
    /// assert_eq!(mi, Matrix::mat([[-2.0, 1.5],[1.0, -0.5]]), "unexpected inverse matrix");
    ///
    /// // multiplying a matrix by its inverse yields the identity matrix
    /// assert_eq!(m.mmul(&mi), Matrix::identity())
    /// ```
    #[must_use]
    fn inv(&self) -> Option<Mat<T, N, N>>;

    /// Calculate the determinant $|M|$ of the matrix $M$.
    /// If the matrix is singular, the determinant is 0
    #[must_use]
    fn det(&self) -> T;

    /// Solve for $x$ in $bbM xx x = b$
    ///
    /// ```
    /// # use vector_victor::decompose::LUDecompose;
    /// # use vector_victor::{Matrix, Vector};
    ///
    /// let m = Matrix::mat([[1.0,3.0],[2.0,4.0]]);
    /// let b = Vector::vec([7.0,10.0]);
    /// let x = m.solve(&b).expect("Cannot solve a singular matrix");
    ///
    /// assert_eq!(x, Vector::vec([1.0,2.0]), "x = [1,2]");
    /// assert_eq!(m.mmul(&x), b, "Mx = b");
    /// ```
    ///
    /// $x$ does not need to be a column-vector, it can also be a 2D matrix. For example,
    /// the following is another way to calculate the [inverse](crate::legacy::decompose::LUDecompose::inv()) by solving for the identity matrix $I$.
    ///
    /// ```
    /// # use vector_victor::decompose::LUDecompose;
    /// # use vector_victor::{Matrix, Vector};
    ///
    /// let m = Matrix::mat([[1.0,3.0],[2.0,4.0]]);
    /// let i = Matrix::<f64,2,2>::identity();
    /// let mi = m.solve(&i).expect("Cannot solve a singular matrix");
    ///
    /// assert_eq!(mi, Matrix::mat([[-2.0, 1.5],[1.0, -0.5]]));
    /// assert_eq!(m.mmul(&mi), i, "M x M^-1 = I");
    /// ```
    #[must_use]
    fn solve<B, const M: usize>(&self, b: B) -> Option<Mat<T, N, M>>
    where
        B: Splat<Mat<T, N, M>>;
}

impl<T, const N: usize> LUDecompose<T, N> for Mat<T, N, N>
where
    T: Copy + Default + Real + Sum + Product + Signed + Splat<T> + Splat<Mat<T, N, N>>,
{
    fn lu(&self) -> Option<LUDecomposition<T, N>> {
        // Implementation from Numerical Recipes §2.3
        let mut lu = self.clone();
        let mut idx: Col<usize, N> = Col::<usize, N>::from_rows(0..N);
        let mut parity = Parity::Even;

        let mut vv: Col<T, N> = self
            .rows()
            .map(|row| {
                let m = row.iter().cloned().reduce(|acc, x| acc.max(x.abs()))?;
                checked_inv(m)
            })
            .collect::<Option<_>>()?; // get the inverse max abs value in each row

        // for each column in the matrix...
        for k in 0..N {
            // search for the pivot element and its index
            let (ipivot, _) = (lu.column(k) * vv)
                .abs()
                .iter()
                .enumerate()
                .skip(k) // below the diagonal
                .reduce(|(imax, xmax), (i, x)| match x > xmax {
                    // Is the figure of merit for the pivot better than the best so far?
                    true => (i, x),
                    false => (imax, xmax),
                })?;

            // do we need to interchange rows?
            if k != ipivot {
                lu.pivot_rows(ipivot, k);
                idx.pivot_rows(ipivot, k);
                parity = !parity; // swap parity
                vv[ipivot] = vv[k] // interchange scale factor
            }

            // select our pivot, which is now on the diagonal
            let pivot = lu[k][k];
            if pivot.abs() < T::epsilon() {
                // if the pivot is zero, the matrix is singular
                return None;
            };

            // for each element in the column k below the diagonal...
            // this is called outer product Gaussian elimination
            for i in (k + 1)..N {
                // divide by the pivot element
                lu[i][k] = lu[i][k] / pivot;

                // for each element in the column k below the diagonal...
                for j in (k + 1)..N {
                    // reduce remaining submatrix
                    lu[i][j] = lu[i][j] - (lu[i][k] * lu[k][j]);
                }
            }
        }

        return Some(LUDecomposition { lu, idx, parity });
    }

    fn inv(&self) -> Option<Mat<T, N, N>> {
        match N {
            1 => Some(checked_inv(self[0][0])?.splat()),
            2 => {
                let mut result = Self::default();
                result[0][0] = self[1][1];
                result[1][1] = self[0][0];
                result[1][0] = -self[1][0];
                result[0][1] = -self[0][1];
                Some(result * checked_inv(self.det())?)
            }
            _ => Some(self.lu()?.inv()),
        }
    }

    fn det(&self) -> T {
        match N {
            1 => self[0][0],
            2 => (self[0][0] * self[1][1]) - (self[0][1] * self[1][0]),
            3 => {
                // use rule of Sarrus
                (0..N) // starting column
                    .map(|i| {
                        let dn = (0..N).map(|j| -> T { self[j][(j + i) % N] }).product::<T>();
                        let up = (0..N)
                            .map(|j| -> T { self[N - j - 1][(j + i) % N] })
                            .product::<T>();
                        dn - up
                    })
                    .sum::<T>()
            }
            _ => {
                // use LU decomposition
                self.lu().map_or(T::zero(), |lu| lu.det())
            }
        }
    }

    fn solve<B, const M: usize>(&self, b: B) -> Option<Mat<T, N, M>>
    where
        B: Splat<Mat<T, N, M>>,
    {
        Some(self.lu()?.solve(b))
    }
}
