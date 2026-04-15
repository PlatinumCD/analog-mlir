module {
  func.func @forward(%arg0: tensor<1x4x5x5xf32>) -> tensor<1x6x3x3xf32> {
    %cst = arith.constant dense_resource<torch_tensor_6_torch.float32> : tensor<6xf32>
    %cst_0 = arith.constant dense_resource<torch_tensor_6_2_3_3_torch.float32> : tensor<6x2x3x3xf32>
    %0 = tensor.empty() : tensor<1x6x3x3xf32>
    %broadcasted = linalg.broadcast ins(%cst : tensor<6xf32>) outs(%0 : tensor<1x6x3x3xf32>) dimensions = [0, 2, 3] 
    %expanded = tensor.expand_shape %arg0 [[0], [1, 2], [3], [4]] output_shape [1, 2, 2, 5, 5] : tensor<1x4x5x5xf32> into tensor<1x2x2x5x5xf32>
    %expanded_1 = tensor.expand_shape %cst_0 [[0, 1], [2], [3], [4]] output_shape [2, 3, 2, 3, 3] : tensor<6x2x3x3xf32> into tensor<2x3x2x3x3xf32>
    %expanded_2 = tensor.expand_shape %broadcasted [[0], [1, 2], [3], [4]] output_shape [1, 2, 3, 3, 3] : tensor<1x6x3x3xf32> into tensor<1x2x3x3x3xf32>
    %1 = linalg.conv_2d_ngchw_gfchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%expanded, %expanded_1 : tensor<1x2x2x5x5xf32>, tensor<2x3x2x3x3xf32>) outs(%expanded_2 : tensor<1x2x3x3x3xf32>) -> tensor<1x2x3x3x3xf32>
    %collapsed = tensor.collapse_shape %1 [[0], [1, 2], [3], [4]] : tensor<1x2x3x3x3xf32> into tensor<1x6x3x3xf32>
    return %collapsed : tensor<1x6x3x3xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_6_torch.float32: "0x04000000000000000000803F0000004000004040000080400000A040",
      torch_tensor_6_2_3_3_torch.float32: "0x040000000AD7233C0AD7A33C8FC2F53C0AD7233DCDCC4C3D8FC2753D295C8F3D0AD7A33DEC51B83DCDCCCC3DAE47E13D8FC2F53DB81E053E295C0F3E9A99193E0AD7233E7B142E3EEC51383E5C8F423ECDCC4C3E3D0A573EAE47613E1F856B3E8FC2753E0000803EB81E853E713D8A3E295C8F3EE17A943E9A99993E52B89E3E0AD7A33EC3F5A83E7B14AE3E3333B33EEC51B83EA470BD3E5C8FC23E14AEC73ECDCCCC3E85EBD13E3D0AD73EF628DC3EAE47E13E6666E63E1F85EB3ED7A3F03E8FC2F53E48E1FA3E0000003F5C8F023FB81E053F14AE073F713D0A3FCDCC0C3F295C0F3F85EB113FE17A143F3D0A173F9A99193FF6281C3F52B81E3FAE47213F0AD7233F6666263FC3F5283F1F852B3F7B142E3FD7A3303F3333333F8FC2353FEC51383F48E13A3FA4703D3F0000403F5C8F423FB81E453F14AE473F713D4A3FCDCC4C3F295C4F3F85EB513FE17A543F3D0A573F9A99593FF6285C3F52B85E3FAE47613F0AD7633F6666663FC3F5683F1F856B3F7B146E3FD7A3703F3333733F8FC2753FEC51783F48E17A3FA4707D3F0000803FAE47813F5C8F823F0AD7833FB81E853F6666863F14AE873FC3F5883F713D8A3F"
    }
  }
#-}

