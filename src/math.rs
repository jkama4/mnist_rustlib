use itertools::izip;
use rand::Rng;
use std::error::Error;


pub type Matrix = Vec<Vec<f32>>;


/// Calculate the element-wise product of two matrices
/// :mat_1 (Matrix): the first matrix - likely a patch of shape (kernel_size, kernel_size)
/// :mat_2 (Matrix): the second matrix - likely a patch of shape (kernel_size, kernel_size)
/// :kernel_size (usize): the size of the kernel (e.g., kernel_size = 2, then the kernel is 2x2)
/// :return (Matrix): the results matrix of the element-wise product between mat_1 and mat_2
pub fn product(
    mat_1: &Matrix, 
    mat_2: &Matrix, 
    kernel_size: usize,
) -> f32 {

    let mut res: f32 = 0.0;
    for (row_mat_1, row_mat_2) in izip!(&*mat_1, &*mat_2) {
        for idx in 0..kernel_size {
            res += row_mat_1[idx] * row_mat_2[idx];
        }
    }
    res
}


/// Convert a matrix into its transpose
/// :mat (Matrix): a matrix
/// :return (Matrix): transpose of the matrix, A^T
pub fn transpose(mat: &Matrix) -> Matrix {
    let mut tp = vec![];
    for j in 0..mat[0].len() {
        let mut intermediate_row = vec![];
        for i in 0..mat.len() {
            intermediate_row.push(mat[i][j]);
        }
        tp.push(intermediate_row);
    }
    tp
}


/// Generate random kernel for initialisation
/// :kernel_size (usize): the size of the kernel
/// :return (Matrix): generated kernel
pub fn create_kernel(kernel_size: usize) -> Matrix {
    let mut rng = rand::rng();
    let mut kernel = vec![];

    for i in 0..kernel_size {
        let mut intermediate_row = vec![];
        for j in 0..kernel_size {
            intermediate_row.push(
                rng.random_range(-0.5..0.5)
            );
        }
        kernel.push(intermediate_row);
    }
    kernel
}


/// Extract a patch from a matrix
/// :mat (Matrix): a matrix
/// :row_start (usize): row index to start from
/// :col_start (usize): column index to start from
/// :size (usize): size of the patch extracted
pub fn extract_patch(
    mat: &Matrix, 
    row_start: usize, 
    col_start: usize, 
    size: usize,
) -> Matrix {

    let mut patch = vec![];
    for i in row_start..(row_start + size) {
        let mut intermediate_row = vec![];
        for j in col_start..(col_start + size) {
            intermediate_row.push(mat[i][j]);
        }
        patch.push(intermediate_row);
    }
    patch
}


/// Perform cross-correlation on a matrix
/// :inp_mat (Matrix): the input matrix (8x8 image)
/// :kernel (Matrix): the kernel
/// :return (Matrix): the resulting output matrix
pub fn cross_correlation(inp_mat: &Matrix, kernel: &Matrix) -> Matrix {
    let mut out_mat = vec![];

    let kernel_size = kernel[0].len();
    
    for row_idx in 0..(inp_mat.len() - kernel_size + 1) {
        let mut intermediate_row = vec![];
        for col_idx in 0..(inp_mat[0].len() - kernel_size + 1) {
            let patch = extract_patch(&inp_mat, row_idx, col_idx, kernel_size);
            let prod = product(&patch, &kernel, kernel_size);
            intermediate_row.push(prod);
        }
        out_mat.push(intermediate_row);
    }
    out_mat
}


/// Add a bias-value to the values of a matrix
/// :mat (Matrix): a matrix (after cross-correlation)
/// :bias (f32): a bias value, often a small number
pub fn add_bias(mat: &Matrix, bias: &f32) -> Matrix {
    let mut upd_mat = vec![];

    for row_idx in 0..mat.len() {
        let mut intermediate_row = vec![];
        for col_idx in 0..mat[0].len() {
            let updated_value = mat[row_idx][col_idx] + bias;
            intermediate_row.push(updated_value);
        }
        upd_mat.push(intermediate_row);
    }
    upd_mat
}


/// Applies ReLU to all values of a matrix
/// :mat (Matrix): a matrix
/// :return (Matrix): matrix with all values at least being 0.0
pub fn relu(mat: &Matrix) -> Matrix {
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


/// Reshape a flat array to a matrix
/// :flat_vec (Vec<f32>): a flat array containing floats
/// :width (usize): the width of the matrix (8 in this case)
/// :return (Matrix): the reshaped matrix
pub fn reshape(flat_vec: &Vec<f32>, width: usize) -> Matrix {
    let mut resized_mat = vec![];
    for chunk in flat_vec.chunks(width) {
        resized_mat.push(chunk.to_vec());
    }
    resized_mat
}


/// Flatten a matrix to a list
/// :mat (Matrix): a matrix
/// :return (Vec<f32>): a flattened list
pub fn flatten(mat: &Matrix) -> Vec<f32> {
    let total_size = mat.len() * mat[0].len();
    let mut flattened_list = Vec::with_capacity(total_size);
    for row in mat {
        flattened_list.extend(row);
    }
    flattened_list
}