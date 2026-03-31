module {
  func.func @forward(%arg0: tensor<1x8xf32>) -> tensor<1x6xf32> {
    %cst = arith.constant dense_resource<torch_tensor_6_8_torch.float32> : tensor<6x8xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<8x6xf32>
    %transposed = linalg.transpose ins(%cst : tensor<6x8xf32>) outs(%0 : tensor<8x6xf32>) permutation = [1, 0]
    %1 = tensor.empty() : tensor<1x6xf32>
    %2 = linalg.fill ins(%cst_0 : f32) outs(%1 : tensor<1x6xf32>) -> tensor<1x6xf32>
    %3 = linalg.matmul ins(%arg0, %transposed : tensor<1x8xf32>, tensor<8x6xf32>) outs(%2 : tensor<1x6xf32>) -> tensor<1x6xf32>
    return %3 : tensor<1x6xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_6_8_torch.float32: "0x040000000000803F0000004000004040000080400000A0400000C0400000E0400000004100001041000020410000304100004041000050410000604100007041000080410000884100009041000098410000A0410000A8410000B0410000B8410000C0410000C8410000D0410000D8410000E0410000E8410000F0410000F84100000042000004420000084200000C4200001042000014420000184200001C4200002042000024420000284200002C4200003042000034420000384200003C4200004042"
    }
  }
#-}
