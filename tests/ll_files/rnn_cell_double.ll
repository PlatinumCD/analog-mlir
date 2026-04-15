#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @forward(%arg0: tensor<1x4xf32>, %arg1: tensor<1x3xf32>, %arg2: tensor<1x3xf32>) -> tensor<1x3xf32> {
    %cst = arith.constant dense_resource<torch_tensor_3_3_torch.float32> : tensor<3x3xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant dense_resource<torch_tensor_3_torch.float32> : tensor<3xf32>
    %cst_2 = arith.constant dense_resource<torch_tensor_3_4_torch.float32> : tensor<3x4xf32>
    %cst_3 = arith.constant dense_resource<torch_tensor_3_torch.float32_1> : tensor<3xf32>
    %cst_4 = arith.constant dense_resource<torch_tensor_3_3_torch.float32_1> : tensor<3x3xf32>
    %cst_5 = arith.constant dense_resource<torch_tensor_3_torch.float32_2> : tensor<3xf32>
    %cst_6 = arith.constant dense_resource<torch_tensor_3_3_torch.float32_2> : tensor<3x3xf32>
    %cst_7 = arith.constant dense_resource<torch_tensor_3_torch.float32_3> : tensor<3xf32>
    %0 = tensor.empty() : tensor<3x3xf32>
    %transposed = linalg.transpose ins(%cst : tensor<3x3xf32>) outs(%0 : tensor<3x3xf32>) permutation = [1, 0] 
    %1 = tensor.empty() : tensor<1x3xf32>
    %2 = linalg.fill ins(%cst_0 : f32) outs(%1 : tensor<1x3xf32>) -> tensor<1x3xf32>
    %3 = linalg.matmul ins(%arg1, %transposed : tensor<1x3xf32>, tensor<3x3xf32>) outs(%2 : tensor<1x3xf32>) -> tensor<1x3xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%3, %cst_1 : tensor<1x3xf32>, tensor<3xf32>) outs(%1 : tensor<1x3xf32>) {
    ^bb0(%in: f32, %in_11: f32, %out: f32):
      %16 = arith.addf %in, %in_11 : f32
      linalg.yield %16 : f32
    } -> tensor<1x3xf32>
    %5 = tensor.empty() : tensor<4x3xf32>
    %transposed_8 = linalg.transpose ins(%cst_2 : tensor<3x4xf32>) outs(%5 : tensor<4x3xf32>) permutation = [1, 0] 
    %6 = linalg.matmul ins(%arg0, %transposed_8 : tensor<1x4xf32>, tensor<4x3xf32>) outs(%2 : tensor<1x3xf32>) -> tensor<1x3xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%6, %cst_3 : tensor<1x3xf32>, tensor<3xf32>) outs(%1 : tensor<1x3xf32>) {
    ^bb0(%in: f32, %in_11: f32, %out: f32):
      %16 = arith.addf %in, %in_11 : f32
      linalg.yield %16 : f32
    } -> tensor<1x3xf32>
    %8 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%4, %7 : tensor<1x3xf32>, tensor<1x3xf32>) outs(%1 : tensor<1x3xf32>) {
    ^bb0(%in: f32, %in_11: f32, %out: f32):
      %16 = arith.addf %in, %in_11 : f32
      linalg.yield %16 : f32
    } -> tensor<1x3xf32>
    %9 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%8 : tensor<1x3xf32>) outs(%1 : tensor<1x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %16 = math.tanh %in : f32
      linalg.yield %16 : f32
    } -> tensor<1x3xf32>
    %transposed_9 = linalg.transpose ins(%cst_4 : tensor<3x3xf32>) outs(%0 : tensor<3x3xf32>) permutation = [1, 0] 
    %10 = linalg.matmul ins(%arg2, %transposed_9 : tensor<1x3xf32>, tensor<3x3xf32>) outs(%2 : tensor<1x3xf32>) -> tensor<1x3xf32>
    %11 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%10, %cst_5 : tensor<1x3xf32>, tensor<3xf32>) outs(%1 : tensor<1x3xf32>) {
    ^bb0(%in: f32, %in_11: f32, %out: f32):
      %16 = arith.addf %in, %in_11 : f32
      linalg.yield %16 : f32
    } -> tensor<1x3xf32>
    %transposed_10 = linalg.transpose ins(%cst_6 : tensor<3x3xf32>) outs(%0 : tensor<3x3xf32>) permutation = [1, 0] 
    %12 = linalg.matmul ins(%9, %transposed_10 : tensor<1x3xf32>, tensor<3x3xf32>) outs(%2 : tensor<1x3xf32>) -> tensor<1x3xf32>
    %13 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%12, %cst_7 : tensor<1x3xf32>, tensor<3xf32>) outs(%1 : tensor<1x3xf32>) {
    ^bb0(%in: f32, %in_11: f32, %out: f32):
      %16 = arith.addf %in, %in_11 : f32
      linalg.yield %16 : f32
    } -> tensor<1x3xf32>
    %14 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%11, %13 : tensor<1x3xf32>, tensor<1x3xf32>) outs(%1 : tensor<1x3xf32>) {
    ^bb0(%in: f32, %in_11: f32, %out: f32):
      %16 = arith.addf %in, %in_11 : f32
      linalg.yield %16 : f32
    } -> tensor<1x3xf32>
    %15 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%14 : tensor<1x3xf32>) outs(%1 : tensor<1x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %16 = math.tanh %in : f32
      linalg.yield %16 : f32
    } -> tensor<1x3xf32>
    return %15 : tensor<1x3xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_3_3_torch.float32: "0x040000000AD7A33B0AD7233C8FC2753C0AD7A33CCDCCCC3C8FC2F53C295C0F3D0AD7233DEC51383D",
      torch_tensor_3_torch.float32: "0x040000000AD7233C0AD7A33C8FC2F53C",
      torch_tensor_3_4_torch.float32: "0x040000000AD7233C0AD7A33C8FC2F53C0AD7233DCDCC4C3D8FC2753D295C8F3D0AD7A33DEC51B83DCDCCCC3DAE47E13D8FC2F53D",
      torch_tensor_3_torch.float32_1: "0x040000000AD7A33C0AD7233D8FC2753D",
      torch_tensor_3_3_torch.float32_1: "0x040000006F12833B6F12033CA69B443C6F12833C0AD7A33CA69BC43C4260E53C6F12033DBC74133D",
      torch_tensor_3_torch.float32_2: "0x040000008988083C8988883CCDCCCC3C",
      torch_tensor_3_3_torch.float32_2: "0x040000000E74DA3B0E745A3C0AD7A33C0E74DA3C8988083D0AD7233D8C253F3D0E745A3D8FC2753D",
      torch_tensor_3_torch.float32_3: "0x040000008988883C8988083DCDCC4C3D"
    }
  }
#-}

