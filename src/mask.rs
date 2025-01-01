use crate::{Col, Scalar, Splat};
use std::iter::zip;
use std::ops::{BitAnd, Not};

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

    fn select<T: Copy + Maskable<Mask = Self>, R: Splat<T>>(self, t: T, f: R) -> T {
        let mut result = t.clone();
        result.set_masked(!self, f.splat());
        result
    }
}

impl Mask for bool {
    fn and<R: Splat<Self>>(self, rhs: R) -> Self {
        self && rhs.splat()
    }

    fn or<R: Splat<Self>>(self, rhs: R) -> Self {
        self || rhs.splat()
    }
}

impl<M: Mask, const H: usize> Mask for Col<M, H> {
    fn and<R: Splat<Self>>(self, rhs: R) -> Self {
        self.zip(&rhs.splat(), |l, r| l.and(r))
    }

    fn or<R: Splat<Self>>(self, rhs: R) -> Self {
        self.zip(&rhs.splat(), |l, r| l.or(r))
    }
}

pub trait Maskable: Copy {
    type Mask: Mask;

    fn set_masked<R: Splat<Self>>(&mut self, mask: Self::Mask, value: R);
}

impl<T> Maskable for T
where
    T: Copy + Scalar,
{
    type Mask = bool;

    fn set_masked<R: Splat<Self>>(&mut self, mask: Self::Mask, value: R) {
        if mask {
            *self = value.splat();
        }
    }
}

impl<T, const N: usize> Maskable for Col<T, N>
where
    T: Copy + Maskable,
{
    type Mask = Col<T::Mask, N>;

    fn set_masked<R: Splat<Self>>(&mut self, mask: Self::Mask, value: R) {
        let value: Self = value.splat();
        for (i, &m) in mask.iter().enumerate() {
            self[i].set_masked(m, value[i]);
        }
    }
}

pub trait MaskEq: Copy + Maskable {
    fn where_eq<R: Splat<Self>>(&self, rhs: R) -> Self::Mask;

    fn where_ne<R: Splat<Self>>(&self, rhs: R) -> Self::Mask {
        !self.where_eq(rhs)
    }
}

impl<T> MaskEq for T
where
    T: Copy + Scalar + PartialEq,
{
    fn where_eq<R: Splat<Self>>(&self, rhs: R) -> Self::Mask {
        self.eq(&rhs.splat())
    }
}

impl<T, const N: usize> MaskEq for Col<T, N>
where
    Col<T, N>: Copy,
    T: Copy + MaskEq,
{
    fn where_eq<R: Splat<Self>>(&self, rhs: R) -> Self::Mask {
        self.zip(&rhs.splat(), |l, r| l.where_eq(r))
    }
}

pub trait MaskOrd: Copy + Maskable {
    fn where_gt<R: Splat<Self>>(&self, rhs: R) -> Self::Mask;
    fn where_ge<R: Splat<Self>>(&self, rhs: R) -> Self::Mask;
    fn where_lt<R: Splat<Self>>(&self, rhs: R) -> Self::Mask;
    fn where_le<R: Splat<Self>>(&self, rhs: R) -> Self::Mask;
}

impl<T> MaskOrd for T
where
    T: Copy + Scalar + PartialOrd,
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

impl<T, const N: usize> MaskOrd for Col<T, N>
where
    T: Copy + MaskOrd,
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
