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
    let kernel_size: usize = 2;
    let pool_size: usize = 2;
    let stride: usize = 2;
    
    // to be updated during training
    let bias: f32 = 0.5;

    // to be updated during training
    let kernel: math::Matrix = math::create_kernel(kernel_size);

    for (i, sample) in data.iter().enumerate() {
        let inp_mat: math::Matrix = math::reshape(sample, 8);
        let out_mat: math::Matrix = cnn::forward(&inp_mat, &kernel, &bias);
        let mpl: math::Matrix = cnn::max_pool_layer(&out_mat, stride, pool_size);
        
        if i == 0 {
            println!("{:?}", mpl);
        }
    }

    Ok(())
}




