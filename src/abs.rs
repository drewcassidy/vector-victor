use num_traits::Signed;

pub trait Abs {
    fn abs(&self) -> Self;
}

impl<T> Abs for T
where
    T: Signed,
{
    fn abs(&self) -> Self {
        Signed::abs(self)
    }
}
