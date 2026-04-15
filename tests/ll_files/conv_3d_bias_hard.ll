module {
  func.func @forward(%arg0: tensor<1x1x4x4x4xf32>) -> tensor<1x2x3x3x3xf32> {
    %cst = arith.constant dense_resource<torch_tensor_2_torch.float32> : tensor<2xf32>
    %cst_0 = arith.constant dense_resource<torch_tensor_2_1_2_2_2_torch.float32> : tensor<2x1x2x2x2xf32>
    %0 = tensor.empty() : tensor<1x2x3x3x3xf32>
    %broadcasted = linalg.broadcast ins(%cst : tensor<2xf32>) outs(%0 : tensor<1x2x3x3x3xf32>) dimensions = [0, 2, 3, 4] 
    %1 = linalg.conv_3d_ncdhw_fcdhw {dilations = dense<1> : vector<3xi64>, strides = dense<1> : vector<3xi64>} ins(%arg0, %cst_0 : tensor<1x1x4x4x4xf32>, tensor<2x1x2x2x2xf32>) outs(%broadcasted : tensor<1x2x3x3x3xf32>) -> tensor<1x2x3x3x3xf32>
    return %1 : tensor<1x2x3x3x3xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_2_torch.float32: "0x04000000000000000000803F",
      torch_tensor_2_1_2_2_2_torch.float32: "0x040000000AD7233C0AD7A33C8FC2F53C0AD7233DCDCC4C3D8FC2753D295C8F3D0AD7A33DEC51B83DCDCCCC3DAE47E13D8FC2F53DB81E053E295C0F3E9A99193E0AD7233E"
    }
  }
#-}

