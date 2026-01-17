use std::error::Error;
use std::fs::read;
use npyz;

use crate::math;

pub fn load_numpy_files() -> Result<(Vec<u8>, Vec<u8>), Box<dyn Error>> {
    let bytes_x = read("./mnist_data/src/data/digits_X.npy")?;
    let bytes_y = read("./mnist_data/src/data/digits_y.npy")?;
    
    Ok((bytes_x, bytes_y))
}

pub fn convert_numpy_data(bytes_x: &[u8], bytes_y: &[u8]) -> Result<(Vec<Vec<f32>>, Vec<u8>), Box<dyn Error>> {
    let npy_x = npyz::NpyFile::new(bytes_x)?;
    let npy_y = npyz::NpyFile::new(bytes_y)?;
    
    let mut data: math::Matrix = vec![];
    let mut target: Vec<u8> = vec![];
    
    let mut current_row: Vec<f32> = Vec::with_capacity(64);
    for number in npy_x.data::<f32>()? {
        let number = number?;
        current_row.push(number);
        if current_row.len() == 64 {
            data.push(current_row);
            current_row = Vec::with_capacity(64);
        }
    }
    
    for target_value in npy_y.data::<u8>()? {
        target.push(target_value?);
    }
    
    Ok((data, target))
}


