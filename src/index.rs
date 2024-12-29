use crate::{Col, Mat};
use std::ops::{Index, IndexMut};
// impl<T: Copy, I, const N: usize> Index<I> for Col<T, N>
// where
//     [T; N]: Index<I>,
// {
//     type Output = <[T; N] as Index<I>>::Output;
//
//     fn index(&self, index: I) -> &Self::Output {
//         &self.data[index]
//     }
// }
//
// impl<T: Copy, I, const N: usize> IndexMut<I> for Col<T, N>
// where
//     [T; N]: IndexMut<I>,
// {
//     fn index_mut(&mut self, index: I) -> &mut Self::Output {
//         &mut self.data[index]
//     }
// }

pub trait ColGet<T: Copy, const N: usize>: Sized + Copy {
    type Output: ?Sized;
    fn get_unchecked(self, from: &Col<T, N>) -> &Self::Output;
    fn get_mut_unchecked(self, from: &mut Col<T, N>) -> &mut Self::Output;

    fn check_bounds(self) -> Option<()>;

    fn get(self, from: &Col<T, N>) -> Option<&Self::Output> {
        self.check_bounds().map(|()| self.get_unchecked(from))
    }
    fn get_mut(self, from: &mut Col<T, N>) -> Option<&mut Self::Output> {
        self.check_bounds().map(|()| self.get_mut_unchecked(from))
    }
}

impl<T: Copy, const N: usize> ColGet<T, N> for usize {
    type Output = T;

    fn get_unchecked(self, from: &Col<T, N>) -> &Self::Output {
        &from[self]
    }

    fn get_mut_unchecked(self, from: &mut Col<T, N>) -> &mut Self::Output {
        &mut from[self]
    }

    fn check_bounds(self) -> Option<()> {
        (self < N).then_some(())
    }
}

impl<T: Copy, const W: usize, const H: usize> ColGet<Col<T, W>, H> for (usize, usize) {
    type Output = T;

    fn get_unchecked(self, from: &Mat<T, H, W>) -> &Self::Output {
        &from[self.0][self.1]
    }

    fn get_mut_unchecked(self, from: &mut Mat<T, H, W>) -> &mut Self::Output {
        &mut from[self.0][self.1]
    }

    fn check_bounds(self) -> Option<()> {
        (self.0 < H && self.1 < W).then_some(())
    }
}

impl<T: Copy, const D: usize, const W: usize, const H: usize> ColGet<Col<Col<T, D>, W>, H>
    for (usize, usize, usize)
{
    type Output = T;

    fn get_unchecked(self, from: &Mat<Col<T, D>, H, W>) -> &Self::Output {
        &from[self.0][self.1][self.2]
    }

    fn get_mut_unchecked(self, from: &mut Mat<Col<T, D>, H, W>) -> &mut Self::Output {
        &mut from[self.0][self.1][self.2]
    }

    fn check_bounds(self) -> Option<()> {
        (self.0 < H && self.1 < W && self.2 < D).then_some(())
    }
}

impl<T: Copy, const N: usize> Col<T, N> {
    pub fn get<I>(&self, index: I) -> Option<&<I as ColGet<T, N>>::Output>
    where
        I: ColGet<T, N>,
    {
        index.get(self)
    }

    pub fn get_mut<I>(&mut self, index: I) -> Option<&mut <I as ColGet<T, N>>::Output>
    where
        I: ColGet<T, N>,
    {
        index.get_mut(self)
    }
}
