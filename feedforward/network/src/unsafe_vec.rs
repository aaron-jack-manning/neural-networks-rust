use std::ops;
use std::fmt;
use std::iter;

#[derive(Clone)]
pub (crate) struct UnsafeVec<T>(Vec<T>);

impl<T> UnsafeVec<T> {
    pub fn new() -> UnsafeVec<T> {
        UnsafeVec(Vec::new())
    }

    pub fn with_capacity(capacity : usize) -> UnsafeVec<T> {
        UnsafeVec(Vec::with_capacity(capacity))
    }
}

impl<T> iter::FromIterator<T> for UnsafeVec<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter : I) -> Self {
        let mut vec = UnsafeVec::new();

        for i in iter {
            vec.push(i)
        }

        vec
    }
}

impl<T> fmt::Debug for UnsafeVec<T> 
    where T : fmt::Debug {
    fn fmt(&self, f : &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl<T> ops::Deref for UnsafeVec<T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> ops::DerefMut for UnsafeVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T> ops::Index<usize> for UnsafeVec<T> {
    type Output = T;

    fn index(&self, index : usize) -> &Self::Output {
        debug_assert!(index < self.len());
        unsafe { self.get_unchecked(index) }
    }
}

impl<T> ops::Index<ops::Range<usize>> for UnsafeVec<T> {
    type Output = [T];

    fn index(&self, range : ops::Range<usize>) -> &Self::Output {
        &self.0[range]
    }
}

impl<T> ops::IndexMut<usize> for UnsafeVec<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        debug_assert!(index < self.len());
        unsafe { self.get_unchecked_mut(index) }
    }
}


