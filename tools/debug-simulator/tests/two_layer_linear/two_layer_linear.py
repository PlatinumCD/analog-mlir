import torch
from torch_mlir import fx


class TwoLayerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(8, 8, bias=True)
        self.linear2 = torch.nn.Linear(8, 4, bias=True)

        with torch.no_grad():
            weights1 = torch.arange(
                1,
                8 * 8 + 1,
                dtype=torch.float32,
            ).reshape(8, 8)
            weights2 = torch.arange(
                1,
                4 * 8 + 1,
                dtype=torch.float32,
            ).reshape(4, 8)
            self.linear1.weight.copy_(weights1)
            self.linear1.bias.zero_()
            self.linear2.weight.copy_(weights2)
            self.linear2.bias.zero_()

    def forward(self, x):
        x = self.linear1(x)
        x = torch.sigmoid(x)
        x = self.linear2(x)
        return x


model = TwoLayerModel()
model.eval()

x = torch.arange(
    1,
    8 + 1,
    dtype=torch.float32,
).reshape(1, 8)

mlir_module = fx.export_and_import(
    model,
    x,
    output_type="torch",
    func_name="forward",
)

print(mlir_module)
