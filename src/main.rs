mod load_data;
mod cnn;
mod math;

use std::error::Error;

use itertools::izip;


fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>());
}


fn main() -> Result<(), Box<dyn Error>> {
    let (bytes_x, bytes_y) = load_data::load_numpy_files()?;
    let (data, target) = load_data::convert_numpy_data(&bytes_x, &bytes_y)?;
    let weights: math::Matrix = math::create_matrix(9, 10);
    let fc_bias: Vec<f32> = vec![0.0; 10];
    let kernel_size: usize = 2;
    let pool_size: usize = 2;
    let stride: usize = 2;

    // to be updated during training
    let bias: f32 = 0.5;

    // to be updated during training
    let kernel: math::Matrix = math::create_kernel(kernel_size);

    for (i, sample) in data.iter().enumerate() {
        let inp_mat: math::Matrix = math::reshape(sample, 8);
        let conv_out: math::Matrix = cnn::forward(&inp_mat, &kernel, &bias);
        let pooled: math::Matrix = cnn::max_pool_layer(&conv_out, stride, pool_size);
        let flattened: Vec<f32> = math::flatten(&pooled);
        
        let output: Vec<f32> = cnn::fc_layer(&flattened, &weights, &fc_bias);    

    }

    Ok(())
}




