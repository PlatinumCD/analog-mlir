#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d1, d2)>
module {
  func.func @forward(%arg0: tensor<1x1x4xf32>, %arg1: tensor<3x1x3xf32>) -> (tensor<1x1x3xf32>, tensor<3x1x3xf32>) {
    %cst = arith.constant dense_resource<torch_tensor_3_4_torch.float32> : tensor<3x4xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant dense_resource<torch_tensor_3_torch.float32> : tensor<3xf32>
    %cst_2 = arith.constant dense_resource<torch_tensor_3_3_torch.float32> : tensor<3x3xf32>
    %cst_3 = arith.constant dense_resource<torch_tensor_3_torch.float32_1> : tensor<3xf32>
    %cst_4 = arith.constant dense_resource<torch_tensor_3_3_torch.float32_1> : tensor<3x3xf32>
    %cst_5 = arith.constant dense_resource<torch_tensor_3_torch.float32_2> : tensor<3xf32>
    %cst_6 = arith.constant dense_resource<torch_tensor_3_3_torch.float32_2> : tensor<3x3xf32>
    %cst_7 = arith.constant dense_resource<torch_tensor_3_torch.float32_3> : tensor<3xf32>
    %cst_8 = arith.constant dense_resource<torch_tensor_3_3_torch.float32_3> : tensor<3x3xf32>
    %cst_9 = arith.constant dense_resource<torch_tensor_3_torch.float32_4> : tensor<3xf32>
    %cst_10 = arith.constant dense_resource<torch_tensor_3_3_torch.float32_4> : tensor<3x3xf32>
    %cst_11 = arith.constant dense_resource<torch_tensor_3_torch.float32_5> : tensor<3xf32>
    %extracted_slice = tensor.extract_slice %arg1[0, 0, 0] [1, 1, 3] [1, 1, 1] : tensor<3x1x3xf32> to tensor<1x1x3xf32>
    %extracted_slice_12 = tensor.extract_slice %arg1[1, 0, 0] [1, 1, 3] [1, 1, 1] : tensor<3x1x3xf32> to tensor<1x1x3xf32>
    %extracted_slice_13 = tensor.extract_slice %arg1[2, 0, 0] [1, 1, 3] [1, 1, 1] : tensor<3x1x3xf32> to tensor<1x1x3xf32>
    %0 = tensor.empty() : tensor<1x1x4xf32>
    %transposed = linalg.transpose ins(%arg0 : tensor<1x1x4xf32>) outs(%0 : tensor<1x1x4xf32>) permutation = [1, 0, 2] 
    %collapsed = tensor.collapse_shape %transposed [[0], [1, 2]] : tensor<1x1x4xf32> into tensor<1x4xf32>
    %1 = tensor.empty() : tensor<4x3xf32>
    %transposed_14 = linalg.transpose ins(%cst : tensor<3x4xf32>) outs(%1 : tensor<4x3xf32>) permutation = [1, 0] 
    %2 = tensor.empty() : tensor<1x3xf32>
    %3 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<1x3xf32>) -> tensor<1x3xf32>
    %4 = linalg.matmul ins(%collapsed, %transposed_14 : tensor<1x4xf32>, tensor<4x3xf32>) outs(%3 : tensor<1x3xf32>) -> tensor<1x3xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%4, %cst_1 : tensor<1x3xf32>, tensor<3xf32>) outs(%2 : tensor<1x3xf32>) {
    ^bb0(%in: f32, %in_32: f32, %out: f32):
      %24 = arith.addf %in, %in_32 : f32
      linalg.yield %24 : f32
    } -> tensor<1x3xf32>
    %collapsed_15 = tensor.collapse_shape %extracted_slice [[0], [1, 2]] : tensor<1x1x3xf32> into tensor<1x3xf32>
    %6 = tensor.empty() : tensor<3x3xf32>
    %transposed_16 = linalg.transpose ins(%cst_2 : tensor<3x3xf32>) outs(%6 : tensor<3x3xf32>) permutation = [1, 0] 
    %7 = linalg.matmul ins(%collapsed_15, %transposed_16 : tensor<1x3xf32>, tensor<3x3xf32>) outs(%3 : tensor<1x3xf32>) -> tensor<1x3xf32>
    %8 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%7, %cst_3 : tensor<1x3xf32>, tensor<3xf32>) outs(%2 : tensor<1x3xf32>) {
    ^bb0(%in: f32, %in_32: f32, %out: f32):
      %24 = arith.addf %in, %in_32 : f32
      linalg.yield %24 : f32
    } -> tensor<1x3xf32>
    %expanded = tensor.expand_shape %8 [[0], [1, 2]] output_shape [1, 1, 3] : tensor<1x3xf32> into tensor<1x1x3xf32>
    %9 = tensor.empty() : tensor<1x1x3xf32>
    %10 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded, %5 : tensor<1x1x3xf32>, tensor<1x3xf32>) outs(%9 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %in_32: f32, %out: f32):
      %24 = arith.addf %in, %in_32 : f32
      linalg.yield %24 : f32
    } -> tensor<1x1x3xf32>
    %11 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%10 : tensor<1x1x3xf32>) outs(%9 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %24 = math.tanh %in : f32
      linalg.yield %24 : f32
    } -> tensor<1x1x3xf32>
    %collapsed_17 = tensor.collapse_shape %11 [[0, 1], [2]] : tensor<1x1x3xf32> into tensor<1x3xf32>
    %collapsed_18 = tensor.collapse_shape %11 [[0], [1, 2]] : tensor<1x1x3xf32> into tensor<1x3xf32>
    %transposed_19 = linalg.transpose ins(%cst_4 : tensor<3x3xf32>) outs(%6 : tensor<3x3xf32>) permutation = [1, 0] 
    %12 = linalg.matmul ins(%collapsed_18, %transposed_19 : tensor<1x3xf32>, tensor<3x3xf32>) outs(%3 : tensor<1x3xf32>) -> tensor<1x3xf32>
    %13 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%12, %cst_5 : tensor<1x3xf32>, tensor<3xf32>) outs(%2 : tensor<1x3xf32>) {
    ^bb0(%in: f32, %in_32: f32, %out: f32):
      %24 = arith.addf %in, %in_32 : f32
      linalg.yield %24 : f32
    } -> tensor<1x3xf32>
    %collapsed_20 = tensor.collapse_shape %extracted_slice_12 [[0], [1, 2]] : tensor<1x1x3xf32> into tensor<1x3xf32>
    %transposed_21 = linalg.transpose ins(%cst_6 : tensor<3x3xf32>) outs(%6 : tensor<3x3xf32>) permutation = [1, 0] 
    %14 = linalg.matmul ins(%collapsed_20, %transposed_21 : tensor<1x3xf32>, tensor<3x3xf32>) outs(%3 : tensor<1x3xf32>) -> tensor<1x3xf32>
    %15 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%14, %cst_7 : tensor<1x3xf32>, tensor<3xf32>) outs(%2 : tensor<1x3xf32>) {
    ^bb0(%in: f32, %in_32: f32, %out: f32):
      %24 = arith.addf %in, %in_32 : f32
      linalg.yield %24 : f32
    } -> tensor<1x3xf32>
    %expanded_22 = tensor.expand_shape %15 [[0], [1, 2]] output_shape [1, 1, 3] : tensor<1x3xf32> into tensor<1x1x3xf32>
    %16 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_22, %13 : tensor<1x1x3xf32>, tensor<1x3xf32>) outs(%9 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %in_32: f32, %out: f32):
      %24 = arith.addf %in, %in_32 : f32
      linalg.yield %24 : f32
    } -> tensor<1x1x3xf32>
    %17 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%16 : tensor<1x1x3xf32>) outs(%9 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %24 = math.tanh %in : f32
      linalg.yield %24 : f32
    } -> tensor<1x1x3xf32>
    %collapsed_23 = tensor.collapse_shape %17 [[0, 1], [2]] : tensor<1x1x3xf32> into tensor<1x3xf32>
    %collapsed_24 = tensor.collapse_shape %17 [[0], [1, 2]] : tensor<1x1x3xf32> into tensor<1x3xf32>
    %transposed_25 = linalg.transpose ins(%cst_8 : tensor<3x3xf32>) outs(%6 : tensor<3x3xf32>) permutation = [1, 0] 
    %18 = linalg.matmul ins(%collapsed_24, %transposed_25 : tensor<1x3xf32>, tensor<3x3xf32>) outs(%3 : tensor<1x3xf32>) -> tensor<1x3xf32>
    %19 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%18, %cst_9 : tensor<1x3xf32>, tensor<3xf32>) outs(%2 : tensor<1x3xf32>) {
    ^bb0(%in: f32, %in_32: f32, %out: f32):
      %24 = arith.addf %in, %in_32 : f32
      linalg.yield %24 : f32
    } -> tensor<1x3xf32>
    %collapsed_26 = tensor.collapse_shape %extracted_slice_13 [[0], [1, 2]] : tensor<1x1x3xf32> into tensor<1x3xf32>
    %transposed_27 = linalg.transpose ins(%cst_10 : tensor<3x3xf32>) outs(%6 : tensor<3x3xf32>) permutation = [1, 0] 
    %20 = linalg.matmul ins(%collapsed_26, %transposed_27 : tensor<1x3xf32>, tensor<3x3xf32>) outs(%3 : tensor<1x3xf32>) -> tensor<1x3xf32>
    %21 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%20, %cst_11 : tensor<1x3xf32>, tensor<3xf32>) outs(%2 : tensor<1x3xf32>) {
    ^bb0(%in: f32, %in_32: f32, %out: f32):
      %24 = arith.addf %in, %in_32 : f32
      linalg.yield %24 : f32
    } -> tensor<1x3xf32>
    %expanded_28 = tensor.expand_shape %21 [[0], [1, 2]] output_shape [1, 1, 3] : tensor<1x3xf32> into tensor<1x1x3xf32>
    %22 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_28, %19 : tensor<1x1x3xf32>, tensor<1x3xf32>) outs(%9 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %in_32: f32, %out: f32):
      %24 = arith.addf %in, %in_32 : f32
      linalg.yield %24 : f32
    } -> tensor<1x1x3xf32>
    %23 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%22 : tensor<1x1x3xf32>) outs(%9 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %24 = math.tanh %in : f32
      linalg.yield %24 : f32
    } -> tensor<1x1x3xf32>
    %collapsed_29 = tensor.collapse_shape %23 [[0, 1], [2]] : tensor<1x1x3xf32> into tensor<1x3xf32>
    %transposed_30 = linalg.transpose ins(%23 : tensor<1x1x3xf32>) outs(%9 : tensor<1x1x3xf32>) permutation = [1, 0, 2] 
    %concat = tensor.concat dim(0) %collapsed_17, %collapsed_23, %collapsed_29 : (tensor<1x3xf32>, tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<3x3xf32>
    %expanded_31 = tensor.expand_shape %concat [[0], [1, 2]] output_shape [3, 1, 3] : tensor<3x3xf32> into tensor<3x1x3xf32>
    return %transposed_30, %expanded_31 : tensor<1x1x3xf32>, tensor<3x1x3xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_3_4_torch.float32: "0x040000000AD7233C0AD7A33C8FC2F53C0AD7233DCDCC4C3D8FC2753D295C8F3D0AD7A33DEC51B83DCDCCCC3DAE47E13D8FC2F53D",
      torch_tensor_3_torch.float32: "0x040000000AD7A33C0AD7233D8FC2753D",
      torch_tensor_3_3_torch.float32: "0x040000000AD7A33B0AD7233C8FC2753C0AD7A33CCDCCCC3C8FC2F53C295C0F3D0AD7233DEC51383D",
      torch_tensor_3_torch.float32_1: "0x040000000AD7233C0AD7A33C8FC2F53C",
      torch_tensor_3_3_torch.float32_1: "0x0400000009F2143C09F2943C0E6BDF3C09F2143D8C2E3A3D0E6B5F3DC853823D09F2943D4A90A73D",
      torch_tensor_3_torch.float32_2: "0x040000008988883C8988083DCDCC4C3D",
      torch_tensor_3_3_torch.float32_2: "0x04000000C1099C3BC1091C3CA10E6A3CC1099C3C310CC33CA10EEA3C8988083DC1091C3DF98A2F3D",
      torch_tensor_3_torch.float32_3: "0x040000008988083C8988883CCDCCCC3C",
      torch_tensor_3_3_torch.float32_3: "0x040000008988083C8988883CCDCCCC3C8988083DABAA2A3DCDCC4C3DEFEE6E3D8988883D9A99993D",
      torch_tensor_3_torch.float32_4: "0x04000000A10E6A3CA10EEA3CF98A2F3D",
      torch_tensor_3_3_torch.float32_4: "0x0400000009F2943B09F2143C0E6B5F3C09F2943C8C2EBA3C0E6BDF3CC853023D09F2143D4A90273D",
      torch_tensor_3_torch.float32_5: "0x04000000A10EEA3BA10E6A3CF98AAF3C"
    }
  }
#-}

