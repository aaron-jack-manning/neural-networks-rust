mod algebra;
pub mod data;
pub mod weights_gen;
pub mod activation;
pub mod network;

use crate::algebra::{Vector, Matrix};

#[derive(Clone)]
pub struct DataSet(Vec<Vector>);

#[derive(Debug, Clone)]
pub struct Network {
    structure : Vec<usize>,
    weights : Vec<Matrix>,
    biases : Vec<Vector>, 
    activ : fn(f64) -> f64,
    activ_diff : fn(f64) -> f64
}
