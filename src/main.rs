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

    let train_size = 1600;
    let train_data = &data[0..train_size];
    let train_target = &target[0..train_size];
    let test_data = &data[train_size..];
    let test_target = &target[train_size..];

    let mut weights: math::Matrix = math::create_matrix(9, 10);
    let mut fc_bias: Vec<f32> = vec![0.0; 10];
    let kernel_size: usize = 2;
    let pool_size: usize = 2;
    let stride: usize = 2;
    let lr: f32 = 0.007;
    let epochs = 100;

    let mut bias: f32 = 0.2;

    let mut kernel: math::Matrix = math::create_kernel(kernel_size);

    for epoch in 0..epochs {
        for (i, sample) in train_data.iter().enumerate() {

            // forward pass
            let inp_mat: math::Matrix = math::reshape(sample, 8);
            let pre_relu: math::Matrix = cnn::forward(&inp_mat, &kernel, &bias);
            let conv_out: math::Matrix = cnn::relu(&pre_relu);
            let pooled: math::Matrix = cnn::max_pool_layer(&conv_out, stride, pool_size);
            let flattened: Vec<f32> = math::flatten(&pooled);
            let output: Vec<f32> = cnn::fc_layer(&flattened, &weights, &fc_bias);
            let probabilities: Vec<f32> = math::softmax(&output);

            let loss: f32 = cnn::cross_entropy(&probabilities, train_target[i] as usize);

            // backward pass
            let y_grad: Vec<f32> = cnn::y_gradient(&probabilities, target[i] as usize);
            let (fc_bias_grad, x_grad, w_grad) = cnn::fc_backward(&y_grad, &flattened, &weights);
            let pooled_grad: math::Matrix = math::reshape(&x_grad, 3);
            let maxpool_grad: math::Matrix = cnn::max_pool_backward(&pooled_grad, &conv_out, stride, pool_size);
            let relu_grad: math::Matrix = cnn::relu_backward(&maxpool_grad, &pre_relu);
            let (kernel_grad, conv_bias_grad) = cnn::conv_backward(&relu_grad, &inp_mat, &kernel);

            cnn::update_weights(
                &mut weights,
                &mut fc_bias,
                &mut kernel,
                &mut bias,
                &w_grad,
                &fc_bias_grad,
                &kernel_grad,
                conv_bias_grad,
                lr,
            );
        }
    }

    // evaluation on test set
    let mut correct = 0;
    for (i, sample) in test_data.iter().enumerate() {
        let inp_mat = math::reshape(sample, 8);
        let pre_relu = cnn::forward(&inp_mat, &kernel, &bias);
        let conv_out = cnn::relu(&pre_relu);
        let pooled = cnn::max_pool_layer(&conv_out, stride, pool_size);
        let flattened = math::flatten(&pooled);
        let output = cnn::fc_layer(&flattened, &weights, &fc_bias);
        let probabilities = math::softmax(&output);
        let pred = math::extract_prediction(&probabilities);
        
        if pred == test_target[i] as usize {
            correct += 1;
        }
    }
    println!("Test accuracy: {}/{}", correct, test_data.len());

    Ok(())
}




