#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @forward(%arg0: tensor<1x1x5x5xf32>) -> tensor<1x10xf32> {
    %cst = arith.constant dense_resource<torch_tensor_2_torch.float32> : tensor<2xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 4.000000e+00 : f32
    %cst_2 = arith.constant dense_resource<torch_tensor_2_1_2_2_torch.float32> : tensor<2x1x2x2xf32>
    %cst_3 = arith.constant dense_resource<torch_tensor_4_2_2_2_torch.float32> : tensor<4x2x2x2xf32>
    %cst_4 = arith.constant dense_resource<torch_tensor_4_torch.float32> : tensor<4xf32>
    %cst_5 = arith.constant dense_resource<torch_tensor_6_4_torch.float32> : tensor<6x4xf32>
    %cst_6 = arith.constant dense_resource<torch_tensor_6_torch.float32> : tensor<6xf32>
    %cst_7 = arith.constant dense_resource<torch_tensor_10_6_torch.float32> : tensor<10x6xf32>
    %cst_8 = arith.constant dense_resource<torch_tensor_10_torch.float32> : tensor<10xf32>
    %0 = tensor.empty() : tensor<1x2x4x4xf32>
    %broadcasted = linalg.broadcast ins(%cst : tensor<2xf32>) outs(%0 : tensor<1x2x4x4xf32>) dimensions = [0, 2, 3] 
    %1 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%arg0, %cst_2 : tensor<1x1x5x5xf32>, tensor<2x1x2x2xf32>) outs(%broadcasted : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1 : tensor<1x2x4x4xf32>) outs(%0 : tensor<1x2x4x4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %22 = math.tanh %in : f32
      linalg.yield %22 : f32
    } -> tensor<1x2x4x4xf32>
    %3 = tensor.empty() : tensor<1x2x2x2xf32>
    %4 = linalg.fill ins(%cst_0 : f32) outs(%3 : tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
    %5 = tensor.empty() : tensor<2x2xf32>
    %6 = linalg.pooling_nchw_sum {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%2, %5 : tensor<1x2x4x4xf32>, tensor<2x2xf32>) outs(%4 : tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%6 : tensor<1x2x2x2xf32>) outs(%3 : tensor<1x2x2x2xf32>) {
    ^bb0(%in: f32, %out: f32):
      %22 = arith.divf %in, %cst_1 : f32
      linalg.yield %22 : f32
    } -> tensor<1x2x2x2xf32>
    %8 = tensor.empty() : tensor<1x4x1x1xf32>
    %broadcasted_9 = linalg.broadcast ins(%cst_4 : tensor<4xf32>) outs(%8 : tensor<1x4x1x1xf32>) dimensions = [0, 2, 3] 
    %9 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%7, %cst_3 : tensor<1x2x2x2xf32>, tensor<4x2x2x2xf32>) outs(%broadcasted_9 : tensor<1x4x1x1xf32>) -> tensor<1x4x1x1xf32>
    %10 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%9 : tensor<1x4x1x1xf32>) outs(%8 : tensor<1x4x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %22 = math.tanh %in : f32
      linalg.yield %22 : f32
    } -> tensor<1x4x1x1xf32>
    %collapsed = tensor.collapse_shape %10 [[0], [1, 2, 3]] : tensor<1x4x1x1xf32> into tensor<1x4xf32>
    %11 = tensor.empty() : tensor<4x6xf32>
    %transposed = linalg.transpose ins(%cst_5 : tensor<6x4xf32>) outs(%11 : tensor<4x6xf32>) permutation = [1, 0] 
    %12 = tensor.empty() : tensor<1x6xf32>
    %13 = linalg.fill ins(%cst_0 : f32) outs(%12 : tensor<1x6xf32>) -> tensor<1x6xf32>
    %14 = linalg.matmul ins(%collapsed, %transposed : tensor<1x4xf32>, tensor<4x6xf32>) outs(%13 : tensor<1x6xf32>) -> tensor<1x6xf32>
    %15 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%14, %cst_6 : tensor<1x6xf32>, tensor<6xf32>) outs(%12 : tensor<1x6xf32>) {
    ^bb0(%in: f32, %in_11: f32, %out: f32):
      %22 = arith.addf %in, %in_11 : f32
      linalg.yield %22 : f32
    } -> tensor<1x6xf32>
    %16 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%15 : tensor<1x6xf32>) outs(%12 : tensor<1x6xf32>) {
    ^bb0(%in: f32, %out: f32):
      %22 = math.tanh %in : f32
      linalg.yield %22 : f32
    } -> tensor<1x6xf32>
    %17 = tensor.empty() : tensor<6x10xf32>
    %transposed_10 = linalg.transpose ins(%cst_7 : tensor<10x6xf32>) outs(%17 : tensor<6x10xf32>) permutation = [1, 0] 
    %18 = tensor.empty() : tensor<1x10xf32>
    %19 = linalg.fill ins(%cst_0 : f32) outs(%18 : tensor<1x10xf32>) -> tensor<1x10xf32>
    %20 = linalg.matmul ins(%16, %transposed_10 : tensor<1x6xf32>, tensor<6x10xf32>) outs(%19 : tensor<1x10xf32>) -> tensor<1x10xf32>
    %21 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%20, %cst_8 : tensor<1x10xf32>, tensor<10xf32>) outs(%18 : tensor<1x10xf32>) {
    ^bb0(%in: f32, %in_11: f32, %out: f32):
      %22 = arith.addf %in, %in_11 : f32
      linalg.yield %22 : f32
    } -> tensor<1x10xf32>
    return %21 : tensor<1x10xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_2_torch.float32: "0x040000000AD7233C0AD7A33C",
      torch_tensor_2_1_2_2_torch.float32: "0x04000000CDCC4C3D8FC2753D295C8F3D0AD7A33DEC51B83DCDCCCC3DAE47E13D8FC2F53D",
      torch_tensor_4_2_2_2_torch.float32: "0x040000000AD7A33CCDCCCC3C8FC2F53C295C0F3D0AD7233DEC51383DCDCC4C3DAE47613D8FC2753DB81E853D295C8F3D9A99993D0AD7A33D7B14AE3DEC51B83D5C8FC23DCDCCCC3D3D0AD73DAE47E13D1F85EB3D8FC2F53D0000003EB81E053E713D0A3E295C0F3EE17A143E9A99193E52B81E3E0AD7233EC3F5283E7B142E3E3333333E",
      torch_tensor_4_torch.float32: "0x040000008FC2F53C0AD7233DCDCC4C3D8FC2753D",
      torch_tensor_6_4_torch.float32: "0x040000000AD7233CA69B443C4260653C6F12833CBC74933C0AD7A33C5839B43CA69BC43CF4FDD43C4260E53C8FC2F53C6F12033D96430B3DBC74133DE3A51B3D0AD7233D31082C3D5839343D7F6A3C3DA69B443DCDCC4C3DF4FD543D1B2F5D3D4260653D",
      torch_tensor_6_torch.float32: "0x040000000AD7233DCDCC4C3D8FC2753D295C8F3D0AD7A33DEC51B83D",
      torch_tensor_10_6_torch.float32: "0x040000000AD7A33BA69BC43B4260E53B6F12033CBC74133C0AD7233C5839343CA69B443CF4FD543C4260653C8FC2753C6F12833C96438B3CBC74933CE3A59B3C0AD7A33C3108AC3C5839B43C7F6ABC3CA69BC43CCDCCCC3CF4FDD43C1B2FDD3C4260E53C6891ED3C8FC2F53CB6F3FD3C6F12033D022B073D96430B3D295C0F3DBC74133D508D173DE3A51B3D77BE1F3D0AD7233D9EEF273D31082C3DC520303D5839343DEC51383D7F6A3C3D1283403DA69B443D39B4483DCDCC4C3D60E5503DF4FD543D8716593D1B2F5D3DAE47613D4260653DD578693D68916D3DFCA9713D8FC2753D23DB793DB6F37D3D2506813D6F12833D",
      torch_tensor_10_torch.float32: "0x040000000AD7A33CCDCCCC3C8FC2F53C295C0F3D0AD7233DEC51383DCDCC4C3DAE47613D8FC2753DB81E853D"
    }
  }
#-}

