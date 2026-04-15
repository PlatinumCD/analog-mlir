module {
  func.func @forward(%arg0: tensor<1x1x5x5xf32>) -> tensor<1x1x3x3xf32> {
    %cst = arith.constant dense_resource<torch_tensor_1_1_3_3_torch.float32> : tensor<1x1x3x3xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x1x3x3xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>
    %2 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%arg0, %cst : tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>) outs(%1 : tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>
    return %2 : tensor<1x1x3x3xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_1_1_3_3_torch.float32: "0x040000000AD7A33DEC51B83D9A99193E9A99193FAE47613E0AD7A33DEC51B83D0000003F14AE473F"
    }
  }
#-}

