#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @forward(%arg0: tensor<1x4xf32>, %arg1: tensor<1x3xf32>) -> tensor<1x3xf32> {
    %cst = arith.constant dense_resource<torch_tensor_3_3_torch.float32> : tensor<3x3xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant dense_resource<torch_tensor_3_4_torch.float32> : tensor<3x4xf32>
    %0 = tensor.empty() : tensor<3x3xf32>
    %transposed = linalg.transpose ins(%cst : tensor<3x3xf32>) outs(%0 : tensor<3x3xf32>) permutation = [1, 0] 
    %1 = tensor.empty() : tensor<1x3xf32>
    %2 = linalg.fill ins(%cst_0 : f32) outs(%1 : tensor<1x3xf32>) -> tensor<1x3xf32>
    %3 = linalg.matmul ins(%arg1, %transposed : tensor<1x3xf32>, tensor<3x3xf32>) outs(%2 : tensor<1x3xf32>) -> tensor<1x3xf32>
    %4 = tensor.empty() : tensor<4x3xf32>
    %transposed_2 = linalg.transpose ins(%cst_1 : tensor<3x4xf32>) outs(%4 : tensor<4x3xf32>) permutation = [1, 0] 
    %5 = linalg.matmul ins(%arg0, %transposed_2 : tensor<1x4xf32>, tensor<4x3xf32>) outs(%2 : tensor<1x3xf32>) -> tensor<1x3xf32>
    %6 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%3, %5 : tensor<1x3xf32>, tensor<1x3xf32>) outs(%1 : tensor<1x3xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %8 = arith.addf %in, %in_3 : f32
      linalg.yield %8 : f32
    } -> tensor<1x3xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%6 : tensor<1x3xf32>) outs(%1 : tensor<1x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %8 = math.tanh %in : f32
      linalg.yield %8 : f32
    } -> tensor<1x3xf32>
    return %7 : tensor<1x3xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_3_3_torch.float32: "0x040000000AD7A33B0AD7233C8FC2753C0AD7A33CCDCCCC3C8FC2F53C295C0F3D0AD7233DEC51383D",
      torch_tensor_3_4_torch.float32: "0x040000000AD7233C0AD7A33C8FC2F53C0AD7233DCDCC4C3D8FC2753D295C8F3D0AD7A33DEC51B83DCDCCCC3DAE47E13D8FC2F53D"
    }
  }
#-}

