use crate::math;


pub fn forward(input: &math::Matrix, kernel: &math::Matrix, bias: &f32) -> math::Matrix {
    let mut output = math::cross_correlation(&input, &kernel);
    output = math::add_bias(&output, &bias);
    output
}


pub fn fc_layer(
    input: &Vec<f32>, 
    weights: &math::Matrix, 
    bias: &Vec<f32>
) -> Vec<f32> {

    let mut output = vec![];

    for col_idx in 0..weights[0].len() {
        let mut intermediate_vec = vec![];
        for row in weights {
            intermediate_vec.push(row[col_idx]);
        }
        let score = math::dot(input, &intermediate_vec) + bias[col_idx];
        output.push(score);
    }
    output
}


pub fn fc_backward(
    y_grad: &Vec<f32>,
    x: &Vec<f32>,
    weights: &math::Matrix
) -> (Vec<f32>, Vec<f32>, math::Matrix) {

    let b_grad = y_grad.clone();
    
    let mut x_grad = vec![];
    for j in 0..x.len() {
        let mut sum: f32 = 0.0;
        for i in 0..y_grad.len() {
            sum += y_grad[i] * weights[j][i];
        }
        x_grad.push(sum);
    }
    
    let mut w_grad = vec![];
    for j in 0..x.len() {
        let mut row = vec![];
        for i in 0..y_grad.len() {
            row.push(y_grad[i] * x[j]);
        }
        w_grad.push(row);
    }
    
    (b_grad, x_grad, w_grad)
}


pub fn max_pool(patch: &math::Matrix) -> f32 {
    let flat_lst = math::flatten(patch);
    let mut max_val = flat_lst[0];
    for val in &flat_lst {
        if *val > max_val {
            max_val = *val;
        }
    }
    max_val
}


pub fn max_pool_backward(
    pooled_grad: &math::Matrix, 
    conv_out: &math::Matrix, 
    stride: usize, 
    pool_size: usize
) -> math::Matrix {

    let mut out_grad = vec![vec![0.0; conv_out[0].len()]; conv_out.len()];
    
    for row_idx in 0..pooled_grad.len() {
        for col_idx in 0..pooled_grad[0].len() {
            let start_row = row_idx * stride;
            let start_col = col_idx * stride;
            
            let mut max_val = conv_out[start_row][start_col];
            let mut max_row = start_row;
            let mut max_col = start_col;
            
            for i in 0..pool_size {
                for j in 0..pool_size {
                    if conv_out[start_row + i][start_col + j] > max_val {
                        max_val = conv_out[start_row + i][start_col + j];
                        max_row = start_row + i;
                        max_col = start_col + j;
                    }
                }
            }
            
            out_grad[max_row][max_col] = pooled_grad[row_idx][col_idx];
        }
    }
    out_grad
}


pub fn relu_backward(grad: &math::Matrix, pre_relu: &math::Matrix) -> math::Matrix {
    let mut out_grad = vec![];

    for i in 0..grad.len() {
        let mut row = vec![];
        for j in 0..grad[0].len() {
            if pre_relu[i][j] > 0.0 {
                row.push(grad[i][j]);
            } else {
                row.push(0.0);
            }
        }
        out_grad.push(row);
    }
    out_grad
}


pub fn relu(mat: &math::Matrix) -> math::Matrix {
    let mut upd_mat = vec![];

    for row_idx in 0..mat.len() {
        let mut intermediate_row = vec![];
        for col_idx in 0..mat[0].len() {
            intermediate_row.push(mat[row_idx][col_idx].max(0.0));
        }
        upd_mat.push(intermediate_row);
    }
    upd_mat
}


pub fn max_pool_layer(mat: &math::Matrix, stride: usize, pool_size: usize) -> math::Matrix {
    let mut out_mat = vec![]; 

    for row_idx in (0..(mat[0].len() - pool_size + 1)).step_by(stride) {
        let mut intermediate_row = vec![];
        for col_idx in (0..(mat.len() - pool_size + 1)).step_by(stride) {
            let patch = math::extract_patch(mat, row_idx, col_idx, pool_size);
            let maxp = max_pool(&patch);
            intermediate_row.push(maxp);
        }
        out_mat.push(intermediate_row);
    }
    out_mat
}


pub fn conv_backward(
    grad: &math::Matrix,
    input: &math::Matrix,
    kernel: &math::Matrix,
) -> (math::Matrix, f32) {

    let mut bias_grad = 0.0;
    for i in 0..grad.len() {
        for j in 0..grad[0].len() {
            bias_grad += grad[i][j];
        }
    }
    
    let mut kernel_grad = vec![vec![0.0; kernel[0].len()]; kernel.len()];

    for ki in 0..kernel.len() {
        for kj in 0..kernel[0].len() {
            for i in 0..grad.len() {
                for j in 0..grad[0].len() {
                    kernel_grad[ki][kj] += grad[i][j] * input[i + ki][j + kj];
                }
            }
        }
    }
    (kernel_grad, bias_grad)
}


pub fn cross_entropy(probabilities: &Vec<f32>, target: usize) -> f32 {
    -probabilities[target].ln()
} 


pub fn y_gradient(probabilities: &Vec<f32>, target: usize) -> Vec<f32> {
    let mut gradient_vector: Vec<f32> = vec![];
    for (i, proba) in probabilities.iter().enumerate() {
        if i == target {
            gradient_vector.push(*proba - 1.0);
        } else {
            gradient_vector.push(*proba);
        }
    }
    gradient_vector
}


pub fn update_weights(
    weights: &mut math::Matrix,
    fc_bias: &mut Vec<f32>,
    kernel: &mut math::Matrix,
    bias: &mut f32,
    w_grad: &math::Matrix,
    fc_bias_grad: &Vec<f32>,
    kernel_grad: &math::Matrix,
    conv_bias_grad: f32,
    lr: f32
) {
    *bias = *bias - lr * conv_bias_grad;

    for i in 0..fc_bias.len() {
        fc_bias[i] = fc_bias[i] - lr * fc_bias_grad[i];
    }

    for i in 0..kernel.len() {
        for j in 0..kernel[0].len() {
            kernel[i][j] = kernel[i][j] - lr * kernel_grad[i][j];
        }
    }

    for i in 0..weights.len() {
        for j in 0..weights[0].len() {
            weights[i][j] = weights[i][j] - lr * w_grad[i][j];
        }
    }
}