use rand::prelude::*;

static mut RNG : Option<ThreadRng> = None;

pub fn init() {
    unsafe {
        RNG = Some(rand::thread_rng());
    }
}

fn uniform() -> f64 {
    unsafe {
        match &mut RNG {
            None => panic!("Cannot generate a random number without first calling init()."),
            Some(rng) => rng.gen(),
        }
    }
}

fn normal_variable() -> f64 {
    f64::sqrt(-2.0 * f64::ln(uniform())) * f64::cos(2.0 * std::f64::consts::PI * uniform())
}

/// Generates a normal random variable with the standard deviation determined by the left of the
/// left and right layers.
pub fn normal(left : usize, right : usize) -> f64 {
    normal_variable() * f64::sqrt(2.0/((left + right) as f64))
}
