// TODO: Algebra module has new vector indexing functions, change all indexing over.

use std::path;

extern crate network;
use network::{DataSet, activation, weights_gen, Network};

fn max_index(vector : &[f64]) -> usize {
    vector
    .iter()
    .enumerate()
    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    .map(|(index, _)| index)
    .unwrap()
}

fn main() {
    weights_gen::init();
    
    let train_input = DataSet::from_csv(path::PathBuf::from("../mnist-datasets/numberstraininput.csv"), false).unwrap();
    let train_expected = DataSet::from_csv(path::PathBuf::from("../mnist-datasets/numberstrainoutput.csv"), false).unwrap();
    let test_input = DataSet::from_csv(path::PathBuf::from("../mnist-datasets/numberstestinput.csv"), false).unwrap();
    let test_expected = DataSet::from_csv(path::PathBuf::from("../mnist-datasets/numberstestoutput.csv"), false).unwrap();
    
    let mut network =
        Network::new(
            vec![784, 20, 20, 10],
            weights_gen::normal,
            |_size| 0.1,
            activation::swish,
            activation::swish_derivative
        );

    let biases_lr = 0.1;
    let weights_lr = 0.01;
   
    let epochs = 5;
    for _epoch in 0..epochs {
        network.train_batch(weights_lr, biases_lr, &train_input, &train_expected)
    }

    let testing_output = network.test(&test_input);

    let mut correct_count : u32 = 0;
    for i in 0..testing_output.quantity() {
        let guess = max_index(testing_output.get(i));
        let actual = max_index(test_expected.get(i));

        if guess == actual {
            correct_count += 1;
        }
    }

    //DataSet::save(&testing_output, path::PathBuf::from("/home/user/Downloads/output.csv")).unwrap();

    let percentage = 
        correct_count as f64 / testing_output.quantity() as f64 * 100.0;
    println!("{} correct out of {} ({:.2}%)", correct_count, testing_output.quantity(), percentage);


    let cost_sum : f64 =
        network.cost(&testing_output, &test_expected)
        .iter()
        .sum();

    println!("{}", cost_sum / testing_output.quantity() as f64);
}
