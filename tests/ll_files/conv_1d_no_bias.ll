module {
  func.func @forward(%arg0: tensor<1x1x6xf32>) -> tensor<1x3x4xf32> {
    %cst = arith.constant dense_resource<torch_tensor_3_1_3_torch.float32> : tensor<3x1x3xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x3x4xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<1x3x4xf32>) -> tensor<1x3x4xf32>
    %2 = linalg.conv_1d_ncw_fcw {dilations = dense<1> : vector<1xi64>, strides = dense<1> : vector<1xi64>} ins(%arg0, %cst : tensor<1x1x6xf32>, tensor<3x1x3xf32>) outs(%1 : tensor<1x3x4xf32>) -> tensor<1x3x4xf32>
    return %2 : tensor<1x3x4xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_3_1_3_torch.float32: "0x040000000AD7233C0AD7A33C8FC2F53C0AD7233DCDCC4C3D8FC2753D295C8F3D0AD7A33DEC51B83D"
    }
  }
#-}

