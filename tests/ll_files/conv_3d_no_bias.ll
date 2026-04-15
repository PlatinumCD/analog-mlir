module {
  func.func @forward(%arg0: tensor<1x1x4x4x4xf32>) -> tensor<1x1x3x3x3xf32> {
    %cst = arith.constant dense_resource<torch_tensor_1_1_2_2_2_torch.float32> : tensor<1x1x2x2x2xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x1x3x3x3xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<1x1x3x3x3xf32>) -> tensor<1x1x3x3x3xf32>
    %2 = linalg.conv_3d_ncdhw_fcdhw {dilations = dense<1> : vector<3xi64>, strides = dense<1> : vector<3xi64>} ins(%arg0, %cst : tensor<1x1x4x4x4xf32>, tensor<1x1x2x2x2xf32>) outs(%1 : tensor<1x1x3x3x3xf32>) -> tensor<1x1x3x3x3xf32>
    return %2 : tensor<1x1x3x3x3xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_1_1_2_2_2_torch.float32: "0x04000000CDCC4C3DCDCCCC3D9A99193ECDCC4C3E0000803E9A99993E3333B33ECDCCCC3E"
    }
  }
#-}

