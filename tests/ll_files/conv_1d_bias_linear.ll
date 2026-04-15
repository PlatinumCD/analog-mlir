#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @forward(%arg0: tensor<1x1x6xf32>) -> tensor<1x2xf32> {
    %cst = arith.constant dense_resource<torch_tensor_3_torch.float32> : tensor<3xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant dense_resource<torch_tensor_3_1_3_torch.float32> : tensor<3x1x3xf32>
    %cst_2 = arith.constant dense_resource<torch_tensor_2_12_torch.float32> : tensor<2x12xf32>
    %cst_3 = arith.constant dense_resource<torch_tensor_2_torch.float32> : tensor<2xf32>
    %0 = tensor.empty() : tensor<1x3x4xf32>
    %broadcasted = linalg.broadcast ins(%cst : tensor<3xf32>) outs(%0 : tensor<1x3x4xf32>) dimensions = [0, 2] 
    %1 = linalg.conv_1d_ncw_fcw {dilations = dense<1> : vector<1xi64>, strides = dense<1> : vector<1xi64>} ins(%arg0, %cst_1 : tensor<1x1x6xf32>, tensor<3x1x3xf32>) outs(%broadcasted : tensor<1x3x4xf32>) -> tensor<1x3x4xf32>
    %collapsed = tensor.collapse_shape %1 [[0], [1, 2]] : tensor<1x3x4xf32> into tensor<1x12xf32>
    %2 = tensor.empty() : tensor<12x2xf32>
    %transposed = linalg.transpose ins(%cst_2 : tensor<2x12xf32>) outs(%2 : tensor<12x2xf32>) permutation = [1, 0] 
    %3 = tensor.empty() : tensor<1x2xf32>
    %4 = linalg.fill ins(%cst_0 : f32) outs(%3 : tensor<1x2xf32>) -> tensor<1x2xf32>
    %5 = linalg.matmul ins(%collapsed, %transposed : tensor<1x12xf32>, tensor<12x2xf32>) outs(%4 : tensor<1x2xf32>) -> tensor<1x2xf32>
    %6 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%5, %cst_3 : tensor<1x2xf32>, tensor<2xf32>) outs(%3 : tensor<1x2xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %7 = arith.addf %in, %in_4 : f32
      linalg.yield %7 : f32
    } -> tensor<1x2xf32>
    return %6 : tensor<1x2xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_3_torch.float32: "0x04000000000000000000000000000000",
      torch_tensor_3_1_3_torch.float32: "0x040000000AD7233C0AD7A33C8FC2F53C0AD7233DCDCC4C3D8FC2753D295C8F3D0AD7A33DEC51B83D",
      torch_tensor_2_12_torch.float32: "0x040000006F12833A6F12033BA69B443B6F12833B0AD7A33BA69BC43B4260E53B6F12033CBC74133C0AD7233C5839343CA69B443CF4FD543C4260653C8FC2753C6F12833C96438B3CBC74933CE3A59B3C0AD7A33C3108AC3C5839B43C7F6ABC3CA69BC43C",
      torch_tensor_2_torch.float32: "0x040000000000000000000000"
    }
  }
#-}

