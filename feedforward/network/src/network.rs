use std::collections::VecDeque;

use crate::algebra::{Vector, Matrix};
use crate::{DataSet, Network};

#[allow(dead_code)]
#[derive(Debug)]
struct FeedForwardResult {
    after_weights : Vector,
    after_biases : Vector,
    pub after_activ : Vector,
}

impl Network {
    /// Creates a new feed forward neural network.
    pub fn new(
        structure : Vec<usize>,
        weights_init : fn(usize, usize) -> f64,
        biases_init : fn(usize) -> f64, 
        activ : fn(f64) -> f64,
        activ_diff : fn(f64) -> f64)
            -> Network {

        // Initialising all the biases.
        let mut biases = Vec::with_capacity(structure.len() - 1);
        for layer_size in structure.iter().skip(1) {
            let mut layer = Vec::with_capacity(*layer_size);
            for _neuron_no in 0..*layer_size {
                layer.push((biases_init)(*layer_size));
            }
            biases.push(Vector::new(layer));
        }

        // Initialising all the weights.
        let mut weights : Vec<Matrix> = Vec::with_capacity(structure.len() - 1);
        for layer_no in 0..(structure.len() - 1) {
            let mut weights_set = Vec::with_capacity(structure[layer_no + 1] * structure[layer_no]);
            for _weight_no in 0..(structure[layer_no + 1] * structure[layer_no]) {
                weights_set.push((weights_init)(structure[layer_no], structure[layer_no + 1]));
            }
            weights.push(
                Matrix::new(
                    structure[layer_no + 1],
                    structure[layer_no],
                    weights_set
                )
            );
        }

        Network {
            structure,
            weights,
            biases,
            activ,
            activ_diff
        }
    }
}

impl Network {
    
    /// Returns the number of layers that the network has.
    fn num_layers(&self) -> usize {
        self.structure.len()
    }

    /// Output is in the form of (input, each layer result)
    fn feed_forward(&self, input : &Vector) -> (Vector, Vec<FeedForwardResult>) {
        let mut result : Vec<FeedForwardResult> = Vec::with_capacity(self.num_layers());

        for output_layer_no in 1..self.num_layers() {

            let after_weights = if output_layer_no == 1 {
                &self.weights[output_layer_no - 1] * input
            } else {
                &self.weights[output_layer_no - 1] * &result.last().unwrap().after_activ
            };

            let after_biases = &after_weights + &self.biases[output_layer_no - 1];

            let after_activ = after_biases.map(self.activ);

            result.push(FeedForwardResult { after_weights, after_biases, after_activ });
        }

        (input.clone(), result)
    }

    /// Feeds forward the provided data set.
    pub fn test(&self, input : &DataSet) -> DataSet {
        let mut result = Vec::with_capacity(input.quantity());

        for i in 0..input.quantity() {
            result.push(
                self.feed_forward(input.internal_get(i)).1.last().unwrap().after_activ.clone()
            );
        }

        DataSet(result)
    }
    
    /// Calculates the cost for the network for a given input.
    pub fn cost(&self, output : &DataSet, expected : &DataSet) -> Vec<f64> {

        if output.quantity() != expected.quantity() {
            panic!("Attempt to calculate cost for a neural network with a different number of output data sets as expected output data sets.")
        }
        if self.structure.last().unwrap() != &output.entries_per_set() || self.structure.last().unwrap() != &expected.entries_per_set() {
            panic!("Attempt to calculate cost for a neural network with an output or expected output which had data sets with length not matching the number of output neurons in the network.")
        }

        fn cost_singular(output : &Vector, expected : &Vector) -> f64 {
            output
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum()
        }

        output.0
        .iter()
        .zip(expected.0.iter())
        .map(|(out, exp)| cost_singular(out, exp))
        .collect()
    }

    /// Backpropagates the network and updates the weights and biases stochastically for a batch of inputs input.
    pub fn train_batch(&mut self, weights_lr : f64, biases_lr : f64, input : &DataSet, expected : &DataSet) {

        if input.quantity() != expected.quantity() {
            panic!("Attempt to train a neural network with a different number of input data sets as output data sets.")
        }
        if self.structure[0] != input.entries_per_set() {
            panic!("Attempt to train a neural network with inputs of the innapropriate size given the number of neurons in the first layer.")
        }
        
        for (input, expected) in input.0.iter().zip(expected.0.iter()) {
            self.train_singular(weights_lr, biases_lr, input, expected);
        }
    }

    /// Backpropagates the network and updates the weights and biases for a single input.
    fn train_singular(&mut self, weights_lr : f64, biases_lr : f64, input : &Vector, expected : &Vector) {
        let feed_forward_results = self.feed_forward(input);

        // Calculate the difference to the weights and biases for all layers.
        let activation_input_diff : VecDeque<Matrix> =
            self.activation_input_diff(&feed_forward_results, expected, 1);
        let biases_diff = &activation_input_diff;
        let weights_diff = self.weight_diff(&activation_input_diff, &feed_forward_results); 

       
        for param_set in 0..(self.num_layers() - 1) {
            self.weights[param_set] = &self.weights[param_set] - &(weights_lr * &weights_diff[param_set]);
            self.biases[param_set] = &self.biases[param_set] - &(biases_lr * &biases_diff[param_set].clone().into_vector());
        }
    }

    /// Calculates the derivative of the network cost with respect to the input of the activation
    /// function for each layer. The resulting VecDeque is indexed from 0 starting at the second
    /// layer in the network. This should be called with an initial value of 1.
    fn activation_input_diff(&self, feed_forward_results : &(Vector, Vec<FeedForwardResult>), expected : &Vector, layer_no : usize) -> VecDeque<Matrix> {

        // Last layer in the network.
        if layer_no == self.num_layers() - 1 {
            let cost_diff =
                (2.0 * &(&feed_forward_results.1[layer_no - 1].after_activ - expected))
                .into_matrix()
                .transpose();

            let activation_derivative =
                Matrix::diagonal(
                    &feed_forward_results.1[layer_no - 1].after_biases.map(self.activ_diff)
                );
            
            let activation_input_diff =
                &cost_diff * &activation_derivative;

            let mut diffs = VecDeque::with_capacity(self.num_layers() - 1); 
            diffs.push_front(activation_input_diff);
            diffs
        }
        else {
            let mut proceeding_layers = self.activation_input_diff(feed_forward_results, expected, layer_no + 1);

            let activation_derivative =
                Matrix::diagonal(
                    &feed_forward_results.1[layer_no - 1].after_biases.map(self.activ_diff)
                );

            let diff = proceeding_layers.front().unwrap() * &(&self.weights[layer_no] * &activation_derivative);
            proceeding_layers.push_front(diff);
            proceeding_layers
        }
    }

    /// Calculates the derivative of the weights with respect to the input to the activation
    /// function of the following layer.
    fn weight_diff(&self, activation_input_diff : &VecDeque<Matrix>, feed_forward_results : &(Vector, Vec<FeedForwardResult>)) -> Vec<Matrix> {
        let mut diffs = Vec::with_capacity(self.num_layers() - 1);
        
        for weight_set in 0..(self.num_layers() - 1) {
            diffs.push(
                Vector::outer_product(
                    &activation_input_diff[weight_set].clone().into_vector(),
                    if weight_set == 0 {
                        &feed_forward_results.0
                    }
                    else {
                        &feed_forward_results.1[weight_set - 1].after_activ
                    }
                )
            );
        }

        diffs
    }
}
