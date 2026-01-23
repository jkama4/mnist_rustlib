use itertools::izip;
use rand::Rng;
use std::error::Error;


pub type Matrix = Vec<Vec<f32>>;


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


pub fn dot(vec_1: &Vec<f32>, vec_2: &Vec<f32>) -> f32 {
    let mut res = 0.0;

    for (i, j) in izip!(vec_1, vec_2) {
        res += i * j;
    }
    res
}


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


pub fn create_matrix(n_rows: usize, n_cols: usize) -> Matrix {
    let mut rng = rand::rng();
    let mut mat = vec![];

    for i in 0..n_rows {
        let mut intermediate_row = vec![];
        for j in 0..n_cols {
            intermediate_row.push(
                rng.random_range(-0.25..0.25)
            );
        }
        mat.push(intermediate_row);
    }
    mat
}


pub fn create_kernel(kernel_size: usize) -> Matrix {
    create_matrix(kernel_size, kernel_size)
}


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


pub fn reshape(flat_vec: &Vec<f32>, width: usize) -> Matrix {
    let mut resized_mat = vec![];
    for chunk in flat_vec.chunks(width) {
        resized_mat.push(chunk.to_vec());
    }
    resized_mat
}


pub fn flatten(mat: &Matrix) -> Vec<f32> {
    let total_size = mat.len() * mat[0].len();
    let mut flattened_list = Vec::with_capacity(total_size);
    for row in mat {
        flattened_list.extend(row);
    }
    flattened_list
}


pub fn softmax(vec: &Vec<f32>) -> Vec<f32> {
    let mut res: Vec<f32> = vec![];

    let mut denominator: f32 = 0.0;
    
    for i in vec.iter() {
        denominator += i.exp();
    }

    for j in vec.iter() {
        let val = j.exp() / denominator;
        res.push(val);
    }
    res
}


pub fn extract_prediction(probabilities: &Vec<f32>) -> usize {
    let mut max_proba: &f32 = &probabilities[0];
    let mut pred: usize = 0;
    for (idx, proba) in probabilities.iter().enumerate() {
        if proba > max_proba {
            max_proba = proba;
            pred = idx;
        }
    }
    pred
}