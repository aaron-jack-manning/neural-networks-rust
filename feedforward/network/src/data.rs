use std::path;

use crate::algebra::Vector;
use crate::DataSet;

use std::vec::Vec as AlgVec;
//use crate::unsafe_vec::UnsafeVec as AlgVec;

impl DataSet {
    /// Reads a data set from a CSV file, where each row is a set of inputs and each column
    /// corresponds with an input neuron.
    pub fn from_csv(path : path::PathBuf, has_headers : bool) -> Result<DataSet, String> {
        let mut data : Vec<Vector> = Vec::new();
        let mut reader = match csv::ReaderBuilder::new()
            .has_headers(has_headers)
            .from_path(path) {
                Ok(reader) => reader,
                Err(error) => return Err(error.to_string())
            };

        let mut length = 0;

        for result in reader.records() {
            let mut row = AlgVec::new();
            let row_record = match result {
                Ok(row_record) => row_record,
                Err(error) => return Err(error.to_string())
            };
            for value in row_record.iter() {
                row.push(value.parse::<f64>().unwrap())
            }
            if length == 0 { length = row.len() }
            else if length != row.len() { return Err(String::from("Input data set does not have consistent length rows.")) }
            data.push(Vector::new(row));
        }

        Ok(DataSet(data))
    }

    /// Returns the number of input sets.
    pub fn quantity(&self) -> usize {
        self.0.len()
    }

    /// Returns the number of entries within in each input set.
    pub fn entries_per_set(&self) -> usize {
        match self.0.len() {
            0 => 0, 
            _ => self.0[0].len()
        }
    }

    /// Returns a reference to the data set at the specified index.
    pub fn get(&self, index : usize) -> &Vec<f64> {
        &self.0[index].0
    }

    /// Returns a reference to the data set at the specified index.
    pub (crate) fn internal_get(&self, index : usize) -> &Vector {
        &self.0[index]
    }

    /// Saves the data set to a CSV file.
    pub fn save(&self, path : path::PathBuf) -> Result<(), String> {
        let mut writer = match csv::Writer::from_path(&path) {
                Ok(writer) => Ok(writer),
                Err(error) => Err(String::from(error.to_string()))
            }?;

        for set in &self.0 {
            let string_array = set.iter().map(|x| x.to_string());
            writer.write_record(string_array).unwrap();
        }
        writer.flush().unwrap();

        Ok(())
    }
}
