use std::ops;
use std::fmt;

#[derive(Clone)]
pub (crate) struct Vector(pub Vec<f64>);

impl fmt::Debug for Vector {
    fn fmt(&self, f : &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Vector {{ Length: {}, Values: {:?} }}", self.0.len(), self.0)
    }
}

#[derive(Clone)]
pub struct Matrix {
    rows : usize,
    cols : usize,
    values : Vec<f64>,
}

impl fmt::Debug for Matrix {
    fn fmt(&self, f : &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut matrix_strings = String::new();
    
        for i in 0..self.rows {
            let slice = &self.values[(i * self.cols)..((i + 1) * self.cols)];
            matrix_strings.push_str(
                &if i == self.rows - 1 {
                    format!("{:?}", slice)
                }
                else {
                    format!("{:?}, ", slice)
                }
            );
        }

        write!(f, "Matrix {{ Rows: {}, Columns {}, Values: [{}] }}", self.rows, self.cols, matrix_strings)
    }
}

impl Vector {
   
    /// Creates a new vector from a Vec<f64>.
    pub (crate) fn new(values : Vec<f64>) -> Vector {
        Vector(values)
    }

    /// Returns the length of the vector.
    pub (crate) fn len(&self) -> usize {
        self.0.len()
    }

    /// Indexes the vector.
    fn index(&self, index : usize) -> f64 {
        self.0[index]
    }

    /// Creates an iterator from the vector.
    pub (crate) fn iter(&self) -> impl Iterator<Item = &f64> {
        self.0.iter()
    }
    
    pub (crate) fn map<F>(&self, mut mapping : F) -> Vector
        where F : FnMut(f64) -> f64 {
        Vector::new(
            self
            .iter()
            .map(|a| { mapping(*a) }) 
            .collect()
        )
    }
    
    pub (crate) fn into_matrix(self) -> Matrix {
        Matrix::new(self.len(), 1, self.0)
    }

    fn component_wise<F>(first : &Vector, second : &Vector, mut operation : F) -> Vector
        where F : FnMut(f64, f64) -> f64 {
        #[cfg(debug_assertions)]
        {
            if first.len() != second.len() {
                panic!("Attempt to perform component wise operation on vectors of different lengths.");
            }
        }
        Vector::new(
            first
            .0
            .iter()
            .zip(second.0.iter())
            .map(|(a, b)| operation(*a, *b))
            .collect()
        )
    } 

    pub (crate) fn outer_product(first : &Vector, second : &Vector) -> Matrix {
        let mut values = Vec::with_capacity(first.len() * second.len());

        for i in 0..first.len() {
            for j in 0..second.len() {
                values.push(
                    first.index(i) * second.index(j)
                );
            }
        }

        Matrix::new(first.len(), second.len(), values)
    }
}    

impl ops::Add<&Vector> for &Vector {
    type Output = Vector;

    fn add(self, other : &Vector) -> Vector {
        Vector::component_wise(self, other, |a, b| a + b)
    }
}

impl ops::Sub<&Vector> for &Vector {
    type Output = Vector;

    fn sub(self, other : &Vector) -> Vector {
        Vector::component_wise(self, other, |a, b| a - b)
    }
}

impl ops::Mul<&Vector> for f64 {
    type Output = Vector;

    fn mul(self, other : &Vector) -> Self::Output {
        other.map(|a| self * a)
    }
}

impl Matrix {
    pub (crate) fn new(rows : usize, cols : usize, values : Vec<f64>) -> Matrix {
        #[cfg(debug_assertions)]
        {
            if values.len() != rows * cols {
                panic!("Attempt to create a matrix with incorrect dimensions.")
            }
        }

        Matrix {
            rows,
            cols,
            values
        }
    }

    pub (crate) fn index(&self, row : usize, col : usize) -> f64 {
        #[cfg(debug_assertions)]
        {
            if row >= self.rows || col >= self.cols {
                panic!("Attempt to index a matrix out of bounds.")
            }
        }

        self.values[row * self.cols + col]
    }

    pub (crate) fn multiply(first : &Matrix, second : &Matrix) -> Matrix {
        #[cfg(debug_assertions)]
        {
            if first.cols != second.rows {
                panic!("Attempt to multiply matrices with incompatible dimensions.")
            } 
        }

        let mut unravelled = Vec::with_capacity(first.rows * second.cols);

        for i in 0..first.rows {
            for j in 0..second.cols {
                
                let mut new_val = 0.0; 

                for k in 0..first.cols { // Equivalent to other.rows
                    new_val += first.index(i, k) * second.index(k, j);
                }

                unravelled.push(new_val);
            }
        }

        Matrix::new(first.rows, second.cols, unravelled)
    }

    pub (crate) fn transpose(&self) -> Matrix {
        let mut values = Vec::with_capacity(self.rows * self.cols);

        for row in 0..self.rows {
            for col in 0..self.cols {
                values.push(self.index(row, col));
            }
        }

        Matrix::new(self.cols, self.rows, values)
    }

    pub (crate) fn diagonal(vector : &Vector) -> Matrix {
        let mut values = Vec::with_capacity(vector.len() * vector.len());

        for row in 0..vector.len() {
            for col in 0..vector.len() {
                values.push(
                    if row == col { vector.index(row) } else { 0.0 }
                );
            }
        }

        Matrix::new(vector.len(), vector.len(), values)
    }

    fn component_wise<F>(first : &Matrix, second : &Matrix, mut operation : F) -> Matrix
        where F : FnMut(f64, f64) -> f64 {
        #[cfg(debug_assertions)]
        {
            if first.rows != second.rows || first.cols != second.cols {
                panic!("Attempt to perform component wise operation on matrices of inconsistent dimensions.");
            }
        }

        Matrix::new(
            first.rows,
            first.cols,
            first 
            .values
            .iter()
            .zip(second.values.iter())
            .map(|(a, b)| operation(*a, *b))
            .collect()
        )
    } 

    pub (crate) fn into_vector(self) -> Vector {
        #[cfg(debug_assertions)]
        {
            if self.rows != 1 && self.cols != 1 {
                panic!("Attempt to convert matrix which is not a single row or column into a vector.")
            }
        }

        Vector::new(self.values)
    }
}

impl ops::Add<&Matrix> for &Matrix {
    type Output = Matrix;

    fn add(self, other : &Matrix) -> Self::Output {
        Matrix::component_wise(self, other, |a, b| a + b)
    }
}

impl ops::Sub<&Matrix> for &Matrix {
    type Output = Matrix;

    fn sub(self, other : &Matrix) -> Self::Output {
        Matrix::component_wise(self, other, |a, b| a - b)
    }
}

impl ops::Mul<&Matrix> for f64 {
    type Output = Matrix;

    fn mul(self, other : &Matrix) -> Self::Output {
        Matrix::new(
            other.rows,
            other.cols,
            other 
            .values
            .iter()
            .map(|a| self * a)
            .collect()
        )
    }
}

impl ops::Mul<&Matrix> for &Matrix {
    type Output = Matrix;

    fn mul(self, other : &Matrix) -> Self::Output {
        Matrix::multiply(self, other)
    }
}


impl ops::Mul<&Vector> for &Matrix {
    type Output = Vector;

    fn mul(self, other : &Vector) -> Self::Output {
        (self * &other.clone().into_matrix()).into_vector()
    }
}
