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

    let mat = vec![
        vec![1.37, 2.13],
        vec![3.2999, 2.11],
    ];

    let tp: math::Matrix = math::transpose(&mat);
    let kernel: math::Matrix = math::create_kernel(2);
    println!("{:?}", kernel);

    Ok(())
}




