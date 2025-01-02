use crate::{Col, Mat, Scalar, Splat};
use std::ops::Not;

pub fn masked_if<M: Mask, F: FnMut(M), const N: usize>(mask: M, f: F) -> M {
    mask.do_(f)
}

pub trait Mask: Copy + Not<Output = Self> {
    fn and<R: Splat<Self>>(self, rhs: R) -> Self;
    fn or<R: Splat<Self>>(self, rhs: R) -> Self;

    fn do_<F: FnMut(Self)>(self, mut f: F) -> Self {
        f(self);
        self
    }

    fn else_<F: FnMut(Self)>(self, f: F) -> Self {
        self.not().do_(f)
    }

    fn else_if<F: FnMut(Self), R: Splat<Self>>(self, rhs: R, f: F) -> Self {
        self.not().and(rhs).do_(f)
    }

    fn select<S: Copy, T: Copy + Maskable<S, Mask = Self>, R: Splat<T>>(self, t: T, f: R) -> T {
        let mut result = t;
        result.set_masked(!self, f.splat());
        result
    }
}

// impl Mask for bool {
//     fn and<R: Splat<Self>>(self, rhs: R) -> Self {
//         self && rhs.splat()
//     }
//
//     fn or<R: Splat<Self>>(self, rhs: R) -> Self {
//         self || rhs.splat()
//     }
// }

impl<const N: usize> Mask for Col<bool, N> {
    fn and<R: Splat<Self>>(self, rhs: R) -> Self {
        self.zip(&rhs.splat(), |l, r| l && r)
    }

    fn or<R: Splat<Self>>(self, rhs: R) -> Self {
        self.zip(&rhs.splat(), |l, r| l && r)
    }
}

impl<M: Mask, const N: usize> Mask for Col<M, N> {
    fn and<R: Splat<Self>>(self, rhs: R) -> Self {
        self.zip(&rhs.splat(), |l, r| l.and(r))
    }

    fn or<R: Splat<Self>>(self, rhs: R) -> Self {
        self.zip(&rhs.splat(), |l, r| l.or(r))
    }
}

pub trait Maskable<T>: Copy {
    type Mask: Mask;

    fn set_masked<R: Splat<Self>>(&mut self, mask: Self::Mask, value: R);
}

impl<T, const N: usize> Maskable<T> for Col<T, N>
where
    T: Copy,
{
    type Mask = Col<bool, N>;

    fn set_masked<R: Splat<Self>>(&mut self, mask: Self::Mask, value: R) {
        let value: Self = value.splat();
        for (i, &m) in mask.iter().enumerate() {
            if m {
                self[i] = value[i]
            }
        }
    }
}

impl<S: Copy, T: Copy, const H: usize, const W: usize> Maskable<T> for Mat<S, W, H>
where
    Col<S, W>: Copy + Maskable<T>,
{
    type Mask = Col<Col<S, W>::Mask, H>;

    fn set_masked<R: Splat<Self>>(&mut self, mask: Self::Mask, value: R) {
        let value: Self = value.splat();
        for (i, &m) in mask.iter().enumerate() {
            self[i].set_masked(m, value[i]);
        }
    }
}

pub trait MaskEq<T>: Copy + Maskable<T> {
    fn where_eq<R: Splat<Self>>(&self, rhs: R) -> Self::Mask;

    fn where_ne<R: Splat<Self>>(&self, rhs: R) -> Self::Mask {
        !self.where_eq(rhs)
    }
}

impl<T, const N: usize> MaskEq<T> for Col<T, N>
where
    Col<T, N>: Copy,
    T: Copy + PartialEq<T>,
{
    fn where_eq<R: Splat<Self>>(&self, rhs: R) -> Self::Mask {
        self.zip(&rhs.splat(), |l, r| l.eq(r))
    }
}

impl<T, const N: usize> MaskEq<T> for Col<T, N>
where
    Col<T, N>: Copy,
    T: Copy + MaskEq<T>,
{
    fn where_eq<R: Splat<Self>>(&self, rhs: R) -> Self::Mask {
        self.zip(&rhs.splat(), |l, r| l.where_eq(r))
    }
}

pub trait MaskOrd<T>: Copy + Maskable<T> {
    fn where_gt<R: Splat<Self>>(&self, rhs: R) -> Self::Mask;
    fn where_ge<R: Splat<Self>>(&self, rhs: R) -> Self::Mask;
    fn where_lt<R: Splat<Self>>(&self, rhs: R) -> Self::Mask;
    fn where_le<R: Splat<Self>>(&self, rhs: R) -> Self::Mask;
}

impl<T> MaskOrd<T> for T
where
    T: Copy + PartialOrd,
{
    fn where_gt<R: Splat<Self>>(&self, rhs: R) -> Self::Mask {
        self > &rhs.splat()
    }

    fn where_ge<R: Splat<Self>>(&self, rhs: R) -> Self::Mask {
        self >= &rhs.splat()
    }

    fn where_lt<R: Splat<Self>>(&self, rhs: R) -> Self::Mask {
        self < &rhs.splat()
    }

    fn where_le<R: Splat<Self>>(&self, rhs: R) -> Self::Mask {
        self <= &rhs.splat()
    }
}

impl<T, const N: usize> MaskOrd<T> for Col<T, N>
where
    T: Copy + MaskOrd<T>,
{
    fn where_gt<R: Splat<Self>>(&self, rhs: R) -> Self::Mask {
        self.zip(&rhs.splat(), |l, r| l.where_gt(r))
    }

    fn where_ge<R: Splat<Self>>(&self, rhs: R) -> Self::Mask {
        self.zip(&rhs.splat(), |l, r| l.where_ge(r))
    }

    fn where_lt<R: Splat<Self>>(&self, rhs: R) -> Self::Mask {
        self.zip(&rhs.splat(), |l, r| l.where_lt(r))
    }

    fn where_le<R: Splat<Self>>(&self, rhs: R) -> Self::Mask {
        self.zip(&rhs.splat(), |l, r| l.where_le(r))
    }
}
