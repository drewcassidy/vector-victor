pub trait Index2D: Copy {
    fn to_1d(self, height: usize, width: usize) -> Option<usize> {
        let (r, c) = self.to_2d(height, width)?;
        Some(r * width + c)
    }

    fn to_2d(self, height: usize, width: usize) -> Option<(usize, usize)>;
}

impl Index2D for usize {
    fn to_2d(self, height: usize, width: usize) -> Option<(usize, usize)> {
        match self < (height * width) {
            true => Some((self / width, self % width)),
            false => None,
        }
    }
}

impl Index2D for (usize, usize) {
    fn to_2d(self, height: usize, width: usize) -> Option<(usize, usize)> {
        match self.0 < height && self.1 < width {
            true => Some(self),
            false => None,
        }
    }
}
