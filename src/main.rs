mod load_data;

use std::error::Error;

fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>());
}


fn main() -> Result<(), Box<dyn Error>> {
    let (bytes_x, bytes_y) = load_data::load_numpy_files()?;
    let (data, target) = load_data::convert_numpy_data(&bytes_x, &bytes_y)?;

    Ok(())
}


