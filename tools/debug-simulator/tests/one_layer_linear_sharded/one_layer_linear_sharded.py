import torch
from torch_mlir import fx


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 6, bias=True)

        with torch.no_grad():
            weights = torch.arange(
                1,
                6 * 8 + 1,
                dtype=torch.float32,
            ).reshape(6, 8)
            self.linear.weight.copy_(weights)
            self.linear.bias.zero_()

    def forward(self, x):
        return self.linear(x)


model = SimpleModel()
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
