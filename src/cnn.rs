use crate::math;



/// Perform a forward pass
/// :input (Matrix): a matrix (8x8 image, sample from data)
/// :kernel (Matrix): kernel that slides over the image
/// :bias (f32): small value added to the input
/// :return (Matrix): the matrix resulting after the pass
pub fn forward(input: &math::Matrix, kernel: &math::Matrix, bias: &f32) -> math::Matrix {
    let mut output = math::cross_correlation(&input, &kernel);
    output = math::add_bias(&output, &bias);
    output = math::relu(&output);
    output
}



/// Perform max pooling on a sample (8x8 image)
/// :mat (Matrix): a matrix
/// :stride (usize): the step size
/// :pool_size (usize): the size of the patch extracted
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


/// Perform max pooling on a patch
/// :patch (Matrix): a patch of a matrix
/// :return (f32): maximumam value of the patch
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