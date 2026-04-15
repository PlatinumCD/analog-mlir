#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0, d1) -> (d0, 0)>
#map4 = affine_map<(d0, d1) -> (0, d1)>
module {
  func.func @forward(%arg0: tensor<1x4xf32>, %arg1: tensor<1x3xf32>, %arg2: tensor<1x3xf32>) -> tensor<1x3xf32> {
    %cst = arith.constant dense_resource<torch_tensor_12_3_torch.float32> : tensor<12x3xf32>
    %c0_i64 = arith.constant 0 : i64
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %cst_2 = arith.constant dense_resource<torch_tensor_12_torch.float32> : tensor<12xf32>
    %cst_3 = arith.constant dense_resource<torch_tensor_12_4_torch.float32> : tensor<12x4xf32>
    %cst_4 = arith.constant dense_resource<torch_tensor_12_torch.float32_1> : tensor<12xf32>
    %c12_i64 = arith.constant 12 : i64
    %c3_i64 = arith.constant 3 : i64
    %c6_i64 = arith.constant 6 : i64
    %c9_i64 = arith.constant 9 : i64
    %0 = tensor.empty() : tensor<3x12xf32>
    %transposed = linalg.transpose ins(%cst : tensor<12x3xf32>) outs(%0 : tensor<3x12xf32>) permutation = [1, 0] 
    %1 = tensor.empty() : tensor<1x12xf32>
    %2 = linalg.fill ins(%cst_0 : f32) outs(%1 : tensor<1x12xf32>) -> tensor<1x12xf32>
    %3 = linalg.matmul ins(%arg1, %transposed : tensor<1x3xf32>, tensor<3x12xf32>) outs(%2 : tensor<1x12xf32>) -> tensor<1x12xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%3, %cst_2 : tensor<1x12xf32>, tensor<12xf32>) outs(%1 : tensor<1x12xf32>) {
    ^bb0(%in: f32, %in_12: f32, %out: f32):
      %35 = arith.addf %in, %in_12 : f32
      linalg.yield %35 : f32
    } -> tensor<1x12xf32>
    %5 = tensor.empty() : tensor<4x12xf32>
    %transposed_5 = linalg.transpose ins(%cst_3 : tensor<12x4xf32>) outs(%5 : tensor<4x12xf32>) permutation = [1, 0] 
    %6 = linalg.matmul ins(%arg0, %transposed_5 : tensor<1x4xf32>, tensor<4x12xf32>) outs(%2 : tensor<1x12xf32>) -> tensor<1x12xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%6, %cst_4 : tensor<1x12xf32>, tensor<12xf32>) outs(%1 : tensor<1x12xf32>) {
    ^bb0(%in: f32, %in_12: f32, %out: f32):
      %35 = arith.addf %in, %in_12 : f32
      linalg.yield %35 : f32
    } -> tensor<1x12xf32>
    %8 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%4, %7 : tensor<1x12xf32>, tensor<1x12xf32>) outs(%1 : tensor<1x12xf32>) {
    ^bb0(%in: f32, %in_12: f32, %out: f32):
      %35 = arith.addf %in, %in_12 : f32
      linalg.yield %35 : f32
    } -> tensor<1x12xf32>
    %collapsed = tensor.collapse_shape %8 [[0, 1]] : tensor<1x12xf32> into tensor<12xf32>
    %9 = tensor.empty() : tensor<1xi64>
    %10 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel"]} outs(%9 : tensor<1xi64>) {
    ^bb0(%out: i64):
      linalg.yield %c0_i64 : i64
    } -> tensor<1xi64>
    %expanded = tensor.expand_shape %10 [[0, 1]] output_shape [1, 1] : tensor<1xi64> into tensor<1x1xi64>
    %11 = tensor.empty() : tensor<1x1xi64>
    %12 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded : tensor<1x1xi64>) outs(%11 : tensor<1x1xi64>) {
    ^bb0(%in: i64, %out: i64):
      %35 = arith.muli %in, %c12_i64 : i64
      linalg.yield %35 : i64
    } -> tensor<1x1xi64>
    %13 = tensor.empty() : tensor<3xi64>
    %14 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel"]} outs(%13 : tensor<3xi64>) {
    ^bb0(%out: i64):
      %35 = linalg.index 0 : index
      %36 = arith.index_cast %35 : index to i64
      linalg.yield %36 : i64
    } -> tensor<3xi64>
    %expanded_6 = tensor.expand_shape %14 [[0, 1]] output_shape [1, 3] : tensor<3xi64> into tensor<1x3xi64>
    %15 = tensor.empty() : tensor<1x3xi64>
    %16 = linalg.generic {indexing_maps = [#map3, #map4, #map], iterator_types = ["parallel", "parallel"]} ins(%12, %expanded_6 : tensor<1x1xi64>, tensor<1x3xi64>) outs(%15 : tensor<1x3xi64>) {
    ^bb0(%in: i64, %in_12: i64, %out: i64):
      %35 = arith.addi %in, %in_12 : i64
      linalg.yield %35 : i64
    } -> tensor<1x3xi64>
    %collapsed_7 = tensor.collapse_shape %16 [[0, 1]] : tensor<1x3xi64> into tensor<3xi64>
    %17 = tensor.empty() : tensor<3xf32>
    %18 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%collapsed_7 : tensor<3xi64>) outs(%17 : tensor<3xf32>) {
    ^bb0(%in: i64, %out: f32):
      %35 = arith.cmpi slt, %in, %c0_i64 : i64
      %36 = arith.addi %in, %c12_i64 : i64
      %37 = arith.select %35, %36, %in : i64
      %38 = arith.index_cast %37 : i64 to index
      %extracted = tensor.extract %collapsed[%38] : tensor<12xf32>
      linalg.yield %extracted : f32
    } -> tensor<3xf32>
    %expanded_8 = tensor.expand_shape %18 [[0, 1]] output_shape [1, 3] : tensor<3xf32> into tensor<1x3xf32>
    %19 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%collapsed_7 : tensor<3xi64>) outs(%13 : tensor<3xi64>) {
    ^bb0(%in: i64, %out: i64):
      %35 = arith.addi %in, %c3_i64 : i64
      linalg.yield %35 : i64
    } -> tensor<3xi64>
    %20 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%19 : tensor<3xi64>) outs(%17 : tensor<3xf32>) {
    ^bb0(%in: i64, %out: f32):
      %35 = arith.cmpi slt, %in, %c0_i64 : i64
      %36 = arith.addi %in, %c12_i64 : i64
      %37 = arith.select %35, %36, %in : i64
      %38 = arith.index_cast %37 : i64 to index
      %extracted = tensor.extract %collapsed[%38] : tensor<12xf32>
      linalg.yield %extracted : f32
    } -> tensor<3xf32>
    %expanded_9 = tensor.expand_shape %20 [[0, 1]] output_shape [1, 3] : tensor<3xf32> into tensor<1x3xf32>
    %21 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%collapsed_7 : tensor<3xi64>) outs(%13 : tensor<3xi64>) {
    ^bb0(%in: i64, %out: i64):
      %35 = arith.addi %in, %c6_i64 : i64
      linalg.yield %35 : i64
    } -> tensor<3xi64>
    %22 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%21 : tensor<3xi64>) outs(%17 : tensor<3xf32>) {
    ^bb0(%in: i64, %out: f32):
      %35 = arith.cmpi slt, %in, %c0_i64 : i64
      %36 = arith.addi %in, %c12_i64 : i64
      %37 = arith.select %35, %36, %in : i64
      %38 = arith.index_cast %37 : i64 to index
      %extracted = tensor.extract %collapsed[%38] : tensor<12xf32>
      linalg.yield %extracted : f32
    } -> tensor<3xf32>
    %expanded_10 = tensor.expand_shape %22 [[0, 1]] output_shape [1, 3] : tensor<3xf32> into tensor<1x3xf32>
    %23 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%collapsed_7 : tensor<3xi64>) outs(%13 : tensor<3xi64>) {
    ^bb0(%in: i64, %out: i64):
      %35 = arith.addi %in, %c9_i64 : i64
      linalg.yield %35 : i64
    } -> tensor<3xi64>
    %24 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%23 : tensor<3xi64>) outs(%17 : tensor<3xf32>) {
    ^bb0(%in: i64, %out: f32):
      %35 = arith.cmpi slt, %in, %c0_i64 : i64
      %36 = arith.addi %in, %c12_i64 : i64
      %37 = arith.select %35, %36, %in : i64
      %38 = arith.index_cast %37 : i64 to index
      %extracted = tensor.extract %collapsed[%38] : tensor<12xf32>
      linalg.yield %extracted : f32
    } -> tensor<3xf32>
    %expanded_11 = tensor.expand_shape %24 [[0, 1]] output_shape [1, 3] : tensor<3xf32> into tensor<1x3xf32>
    %25 = tensor.empty() : tensor<1x3xf32>
    %26 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded_8 : tensor<1x3xf32>) outs(%25 : tensor<1x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.negf %in : f32
      %36 = math.exp %35 : f32
      %37 = arith.addf %36, %cst_1 : f32
      %38 = arith.divf %cst_1, %37 : f32
      linalg.yield %38 : f32
    } -> tensor<1x3xf32>
    %27 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded_9 : tensor<1x3xf32>) outs(%25 : tensor<1x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.negf %in : f32
      %36 = math.exp %35 : f32
      %37 = arith.addf %36, %cst_1 : f32
      %38 = arith.divf %cst_1, %37 : f32
      linalg.yield %38 : f32
    } -> tensor<1x3xf32>
    %28 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded_10 : tensor<1x3xf32>) outs(%25 : tensor<1x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = math.tanh %in : f32
      linalg.yield %35 : f32
    } -> tensor<1x3xf32>
    %29 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded_11 : tensor<1x3xf32>) outs(%25 : tensor<1x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.negf %in : f32
      %36 = math.exp %35 : f32
      %37 = arith.addf %36, %cst_1 : f32
      %38 = arith.divf %cst_1, %37 : f32
      linalg.yield %38 : f32
    } -> tensor<1x3xf32>
    %30 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%27, %arg2 : tensor<1x3xf32>, tensor<1x3xf32>) outs(%25 : tensor<1x3xf32>) {
    ^bb0(%in: f32, %in_12: f32, %out: f32):
      %35 = arith.mulf %in, %in_12 : f32
      linalg.yield %35 : f32
    } -> tensor<1x3xf32>
    %31 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%26, %28 : tensor<1x3xf32>, tensor<1x3xf32>) outs(%25 : tensor<1x3xf32>) {
    ^bb0(%in: f32, %in_12: f32, %out: f32):
      %35 = arith.mulf %in, %in_12 : f32
      linalg.yield %35 : f32
    } -> tensor<1x3xf32>
    %32 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%30, %31 : tensor<1x3xf32>, tensor<1x3xf32>) outs(%25 : tensor<1x3xf32>) {
    ^bb0(%in: f32, %in_12: f32, %out: f32):
      %35 = arith.addf %in, %in_12 : f32
      linalg.yield %35 : f32
    } -> tensor<1x3xf32>
    %33 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%32 : tensor<1x3xf32>) outs(%25 : tensor<1x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = math.tanh %in : f32
      linalg.yield %35 : f32
    } -> tensor<1x3xf32>
    %34 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%29, %33 : tensor<1x3xf32>, tensor<1x3xf32>) outs(%25 : tensor<1x3xf32>) {
    ^bb0(%in: f32, %in_12: f32, %out: f32):
      %35 = arith.mulf %in, %in_12 : f32
      linalg.yield %35 : f32
    } -> tensor<1x3xf32>
    return %34 : tensor<1x3xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_12_3_torch.float32: "0x040000000AD7A33B0AD7233C8FC2753C0AD7A33CCDCCCC3C8FC2F53C295C0F3D0AD7233DEC51383DCDCC4C3DAE47613D8FC2753DB81E853D295C8F3D9A99993D0AD7A33D7B14AE3DEC51B83D5C8FC23DCDCCCC3D3D0AD73DAE47E13D1F85EB3D8FC2F53D0000003EB81E053E713D0A3E295C0F3EE17A143E9A99193E52B81E3E0AD7233EC3F5283E7B142E3E3333333EEC51383E",
      torch_tensor_12_torch.float32: "0x040000000AD7233C0AD7A33C8FC2F53C0AD7233DCDCC4C3D8FC2753D295C8F3D0AD7A33DEC51B83DCDCCCC3DAE47E13D8FC2F53D",
      torch_tensor_12_4_torch.float32: "0x040000000AD7233C0AD7A33C8FC2F53C0AD7233DCDCC4C3D8FC2753D295C8F3D0AD7A33DEC51B83DCDCCCC3DAE47E13D8FC2F53DB81E053E295C0F3E9A99193E0AD7233E7B142E3EEC51383E5C8F423ECDCC4C3E3D0A573EAE47613E1F856B3E8FC2753E0000803EB81E853E713D8A3E295C8F3EE17A943E9A99993E52B89E3E0AD7A33EC3F5A83E7B14AE3E3333B33EEC51B83EA470BD3E5C8FC23E14AEC73ECDCCCC3E85EBD13E3D0AD73EF628DC3EAE47E13E6666E63E1F85EB3ED7A3F03E8FC2F53E",
      torch_tensor_12_torch.float32_1: "0x040000000AD7A33C0AD7233D8FC2753D0AD7A33DCDCCCC3D8FC2F53D295C0F3E0AD7233EEC51383ECDCC4C3EAE47613E8FC2753E"
    }
  }
#-}

