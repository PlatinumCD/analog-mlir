module {
  func.func @forward(%arg0: tensor<1x1x5x5xf32>) -> tensor<1x3x3x3xf32> {
    %cst = arith.constant dense_resource<torch_tensor_3_torch.float32> : tensor<3xf32>
    %cst_0 = arith.constant dense_resource<torch_tensor_3_1_3_3_torch.float32> : tensor<3x1x3x3xf32>
    %0 = tensor.empty() : tensor<1x3x3x3xf32>
    %broadcasted = linalg.broadcast ins(%cst : tensor<3xf32>) outs(%0 : tensor<1x3x3x3xf32>) dimensions = [0, 2, 3] 
    %1 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%arg0, %cst_0 : tensor<1x1x5x5xf32>, tensor<3x1x3x3xf32>) outs(%broadcasted : tensor<1x3x3x3xf32>) -> tensor<1x3x3x3xf32>
    return %1 : tensor<1x3x3x3xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_3_torch.float32: "0x04000000000000000000803F00000040",
      torch_tensor_3_1_3_3_torch.float32: "0x040000000AD7233C0AD7A33C8FC2F53C0AD7233DCDCC4C3D8FC2753D295C8F3D0AD7A33DEC51B83DCDCCCC3DAE47E13D8FC2F53DB81E053E295C0F3E9A99193E0AD7233E7B142E3EEC51383E5C8F423ECDCC4C3E3D0A573EAE47613E1F856B3E8FC2753E0000803EB81E853E713D8A3E"
    }
  }
#-}

