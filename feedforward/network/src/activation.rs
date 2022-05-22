#[inline]
pub fn sigmoid(x : f64) -> f64 {
    let y = (-x).exp();
    if y.is_infinite() {
        0.0
    }
    else {
        1.0 / (1.0 + (-x).exp())
    }
}

#[inline]
pub fn sigmoid_derivative(x : f64) -> f64 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

#[inline]
pub fn swish(x : f64) -> f64 {
    x * sigmoid(x) 
}

#[inline]
pub fn swish_derivative(x : f64) -> f64 {
    sigmoid(x) + x * sigmoid_derivative(x)
}
