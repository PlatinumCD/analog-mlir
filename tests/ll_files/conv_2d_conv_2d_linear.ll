#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @forward(%arg0: tensor<1x1x8x8xf32>) -> tensor<1x3xf32> {
    %cst = arith.constant dense_resource<torch_tensor_3_torch.float32> : tensor<3xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant dense_resource<torch_tensor_3_1_3_3_torch.float32> : tensor<3x1x3x3xf32>
    %cst_2 = arith.constant dense_resource<torch_tensor_2_3_3_3_torch.float32> : tensor<2x3x3x3xf32>
    %cst_3 = arith.constant dense_resource<torch_tensor_2_torch.float32> : tensor<2xf32>
    %cst_4 = arith.constant dense_resource<torch_tensor_3_32_torch.float32> : tensor<3x32xf32>
    %cst_5 = arith.constant dense_resource<torch_tensor_3_torch.float32_1> : tensor<3xf32>
    %0 = tensor.empty() : tensor<1x3x6x6xf32>
    %broadcasted = linalg.broadcast ins(%cst : tensor<3xf32>) outs(%0 : tensor<1x3x6x6xf32>) dimensions = [0, 2, 3] 
    %1 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%arg0, %cst_1 : tensor<1x1x8x8xf32>, tensor<3x1x3x3xf32>) outs(%broadcasted : tensor<1x3x6x6xf32>) -> tensor<1x3x6x6xf32>
    %2 = tensor.empty() : tensor<1x2x4x4xf32>
    %broadcasted_6 = linalg.broadcast ins(%cst_3 : tensor<2xf32>) outs(%2 : tensor<1x2x4x4xf32>) dimensions = [0, 2, 3] 
    %3 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%1, %cst_2 : tensor<1x3x6x6xf32>, tensor<2x3x3x3xf32>) outs(%broadcasted_6 : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
    %collapsed = tensor.collapse_shape %3 [[0], [1, 2, 3]] : tensor<1x2x4x4xf32> into tensor<1x32xf32>
    %4 = tensor.empty() : tensor<32x3xf32>
    %transposed = linalg.transpose ins(%cst_4 : tensor<3x32xf32>) outs(%4 : tensor<32x3xf32>) permutation = [1, 0] 
    %5 = tensor.empty() : tensor<1x3xf32>
    %6 = linalg.fill ins(%cst_0 : f32) outs(%5 : tensor<1x3xf32>) -> tensor<1x3xf32>
    %7 = linalg.matmul ins(%collapsed, %transposed : tensor<1x32xf32>, tensor<32x3xf32>) outs(%6 : tensor<1x3xf32>) -> tensor<1x3xf32>
    %8 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%7, %cst_5 : tensor<1x3xf32>, tensor<3xf32>) outs(%5 : tensor<1x3xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %9 = arith.addf %in, %in_7 : f32
      linalg.yield %9 : f32
    } -> tensor<1x3xf32>
    return %8 : tensor<1x3xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_3_torch.float32: "0x04000000000000000000000000000000",
      torch_tensor_3_1_3_3_torch.float32: "0x040000000AD7233C0AD7A33C8FC2F53C0AD7233DCDCC4C3D8FC2753D295C8F3D0AD7A33DEC51B83DCDCCCC3DAE47E13D8FC2F53DB81E053E295C0F3E9A99193E0AD7233E7B142E3EEC51383E5C8F423ECDCC4C3E3D0A573EAE47613E1F856B3E8FC2753E0000803EB81E853E713D8A3E",
      torch_tensor_2_3_3_3_torch.float32: "0x040000000AD7A33B0AD7233C8FC2753C0AD7A33CCDCCCC3C8FC2F53C295C0F3D0AD7233DEC51383DCDCC4C3DAE47613D8FC2753DB81E853D295C8F3D9A99993D0AD7A33D7B14AE3DEC51B83D5C8FC23DCDCCCC3D3D0AD73DAE47E13D1F85EB3D8FC2F53D0000003EB81E053E713D0A3E295C0F3EE17A143E9A99193E52B81E3E0AD7233EC3F5283E7B142E3E3333333EEC51383EA4703D3E5C8F423E14AE473ECDCC4C3E85EB513E3D0A573EF6285C3EAE47613E6666663E1F856B3ED7A3703E8FC2753E48E17A3E0000803E5C8F823EB81E853E14AE873E713D8A3E",
      torch_tensor_2_torch.float32: "0x040000000000000000000000",
      torch_tensor_3_32_torch.float32: "0x040000006F12833A6F12033BA69B443B6F12833B0AD7A33BA69BC43B4260E53B6F12033CBC74133C0AD7233C5839343CA69B443CF4FD543C4260653C8FC2753C6F12833C96438B3CBC74933CE3A59B3C0AD7A33C3108AC3C5839B43C7F6ABC3CA69BC43CCDCCCC3CF4FDD43C1B2FDD3C4260E53C6891ED3C8FC2F53CB6F3FD3C6F12033D022B073D96430B3D295C0F3DBC74133D508D173DE3A51B3D77BE1F3D0AD7233D9EEF273D31082C3DC520303D5839343DEC51383D7F6A3C3D1283403DA69B443D39B4483DCDCC4C3D60E5503DF4FD543D8716593D1B2F5D3DAE47613D4260653DD578693D68916D3DFCA9713D8FC2753D23DB793DB6F37D3D2506813D6F12833DB81E853D022B873D4C37893D96438B3DDF4F8D3D295C8F3D7368913DBC74933D0681953D508D973D9A99993DE3A59B3D2DB29D3D77BE9F3DC1CAA13D0AD7A33D54E3A53D9EEFA73DE7FBA93D3108AC3D7B14AE3DC520B03D0E2DB23D5839B43DA245B63DEC51B83D355EBA3D7F6ABC3DC976BE3D1283C03D5C8FC23DA69BC43D",
      torch_tensor_3_torch.float32_1: "0x04000000000000000000000000000000"
    }
  }
#-}

