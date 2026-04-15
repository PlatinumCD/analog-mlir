#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @forward(%arg0: tensor<1x8xf32>) -> tensor<1x4xf32> {
    %cst = arith.constant dense_resource<torch_tensor_8_torch.float32> : tensor<8xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %cst_2 = arith.constant dense_resource<torch_tensor_8_8_torch.float32> : tensor<8x8xf32>
    %cst_3 = arith.constant dense_resource<torch_tensor_4_8_torch.float32> : tensor<4x8xf32>
    %cst_4 = arith.constant dense_resource<torch_tensor_4_torch.float32> : tensor<4xf32>
    %0 = tensor.empty() : tensor<8x8xf32>
    %transposed = linalg.transpose ins(%cst_2 : tensor<8x8xf32>) outs(%0 : tensor<8x8xf32>) permutation = [1, 0] 
    %1 = tensor.empty() : tensor<1x8xf32>
    %2 = linalg.fill ins(%cst_0 : f32) outs(%1 : tensor<1x8xf32>) -> tensor<1x8xf32>
    %3 = linalg.matmul ins(%arg0, %transposed : tensor<1x8xf32>, tensor<8x8xf32>) outs(%2 : tensor<1x8xf32>) -> tensor<1x8xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%3, %cst : tensor<1x8xf32>, tensor<8xf32>) outs(%1 : tensor<1x8xf32>) {
    ^bb0(%in: f32, %in_6: f32, %out: f32):
      %11 = arith.addf %in, %in_6 : f32
      linalg.yield %11 : f32
    } -> tensor<1x8xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<1x8xf32>) outs(%1 : tensor<1x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %11 = arith.negf %in : f32
      %12 = math.exp %11 : f32
      %13 = arith.addf %12, %cst_1 : f32
      %14 = arith.divf %cst_1, %13 : f32
      linalg.yield %14 : f32
    } -> tensor<1x8xf32>
    %6 = tensor.empty() : tensor<8x4xf32>
    %transposed_5 = linalg.transpose ins(%cst_3 : tensor<4x8xf32>) outs(%6 : tensor<8x4xf32>) permutation = [1, 0] 
    %7 = tensor.empty() : tensor<1x4xf32>
    %8 = linalg.fill ins(%cst_0 : f32) outs(%7 : tensor<1x4xf32>) -> tensor<1x4xf32>
    %9 = linalg.matmul ins(%5, %transposed_5 : tensor<1x8xf32>, tensor<8x4xf32>) outs(%8 : tensor<1x4xf32>) -> tensor<1x4xf32>
    %10 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%9, %cst_4 : tensor<1x4xf32>, tensor<4xf32>) outs(%7 : tensor<1x4xf32>) {
    ^bb0(%in: f32, %in_6: f32, %out: f32):
      %11 = arith.addf %in, %in_6 : f32
      linalg.yield %11 : f32
    } -> tensor<1x4xf32>
    return %10 : tensor<1x4xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_8_torch.float32: "0x040000000000000000000000000000000000000000000000000000000000000000000000",
      torch_tensor_8_8_torch.float32: "0x040000000000803F0000004000004040000080400000A0400000C0400000E0400000004100001041000020410000304100004041000050410000604100007041000080410000884100009041000098410000A0410000A8410000B0410000B8410000C0410000C8410000D0410000D8410000E0410000E8410000F0410000F84100000042000004420000084200000C4200001042000014420000184200001C4200002042000024420000284200002C4200003042000034420000384200003C4200004042000044420000484200004C4200005042000054420000584200005C4200006042000064420000684200006C4200007042000074420000784200007C4200008042",
      torch_tensor_4_8_torch.float32: "0x040000000000803F0000004000004040000080400000A0400000C0400000E0400000004100001041000020410000304100004041000050410000604100007041000080410000884100009041000098410000A0410000A8410000B0410000B8410000C0410000C8410000D0410000D8410000E0410000E8410000F0410000F84100000042",
      torch_tensor_4_torch.float32: "0x0400000000000000000000000000000000000000"
    }
  }
#-}

