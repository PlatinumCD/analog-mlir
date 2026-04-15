import argparse

import torch
from torch_mlir import fx


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 6, 3, groups=2, bias=True)

        with torch.no_grad():
            conv1_weights = torch.arange(
                1,
                6 * 2 * 3 * 3 + 1,
                dtype=torch.float32,
            ).reshape(6, 2, 3, 3) / 100.0
            self.conv1.weight.copy_(conv1_weights)
            self.conv1.bias.zero_()

    def forward(self, x):
        return self.conv1(x)


model = SimpleModel()
model.eval()

x = torch.arange(
    1,
    4 * 5 * 5 + 1,
    dtype=torch.float32,
).reshape(1, 4, 5, 5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("run", "mlir"), default="run")
    args = parser.parse_args()

    if args.mode == "mlir":
        mlir_module = fx.export_and_import(
            model,
            x,
            output_type="torch",
            func_name="forward",
        )
        print(mlir_module)
        return

    with torch.no_grad():
        print(model(x))


if __name__ == "__main__":
    main()
