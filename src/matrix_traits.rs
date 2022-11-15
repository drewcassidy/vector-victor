use num_traits::Pow;

pub trait Dot<RHS> {
    type Output;
    fn dot(&self, other: &RHS) -> <Self as Dot<RHS>>::Output;
}

pub trait Cross<RHS> {
    type Output;
    fn cross(&self, other: &RHS) -> <Self as Cross<RHS>>::Output;
}

pub trait Mult<RHS> {
    type Output;
    fn mult(&self, other: &RHS) -> <Self as Mult<RHS>>::Output;
}

pub trait Magnitude<T: Pow<f32>> {
    fn sqrmag(&self) -> T;
    fn mag(&self) -> <T as Pow<f32>>::Output;
    fn norm(&self) -> Self;
}
