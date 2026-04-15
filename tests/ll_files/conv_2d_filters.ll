module {
  func.func @forward(%arg0: tensor<1x1x5x5xf32>) -> tensor<1x6x3x3xf32> {
    %cst = arith.constant dense_resource<torch_tensor_6_torch.float32> : tensor<6xf32>
    %cst_0 = arith.constant dense_resource<torch_tensor_6_1_3_3_torch.float32> : tensor<6x1x3x3xf32>
    %0 = tensor.empty() : tensor<1x6x3x3xf32>
    %broadcasted = linalg.broadcast ins(%cst : tensor<6xf32>) outs(%0 : tensor<1x6x3x3xf32>) dimensions = [0, 2, 3] 
    %1 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%arg0, %cst_0 : tensor<1x1x5x5xf32>, tensor<6x1x3x3xf32>) outs(%broadcasted : tensor<1x6x3x3xf32>) -> tensor<1x6x3x3xf32>
    return %1 : tensor<1x6x3x3xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_6_torch.float32: "0x04000000000000000000000000000000000000000000000000000000",
      torch_tensor_6_1_3_3_torch.float32: "0x040000000AD7A33DEC51B83D9A99193E9A99193FAE47613E0AD7A33DEC51B83D0000003F14AE473F8FC2F53DEC51383E8FC2753E9A99993EEC51B83E3D0AD73E8FC2F53E713D0A3F9A99193FCDCC4C3DCDCCCC3D9A99193ECDCC4C3E0000803E9A99993E3333B33ECDCCCC3E6666E63E6666663FCDCC4C3F3333333F9A99193F0000003FCDCCCC3E9A99993ECDCC4C3ECDCCCC3DAE47E13DAE47613EC3F5A83EAE47E13ECDCC0C3FC3F5283FB81E453FAE47613FA4707D3FB81E053E7B142E3E5C8F423E1F856B3EE17A943E52B89E3EA470BD3E85EBD13EF628DC3E"
    }
  }
#-}

