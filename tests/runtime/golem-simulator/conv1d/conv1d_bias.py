import argparse

import torch
from torch_mlir import fx


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(1, 3, 3, bias=True)

        with torch.no_grad():
            conv1_weights = torch.arange(
                1,
                3 * 1 * 3 + 1,
                dtype=torch.float32,
            ).reshape(3, 1, 3) / 100.0
            self.conv1.weight.copy_(conv1_weights)
            self.conv1.bias.zero_()

    def forward(self, x):
        x = self.conv1(x)
        return x


model = SimpleModel()
model.eval()

x = torch.arange(
    1,
    6 + 1,
    dtype=torch.float32,
).reshape(1, 1, 6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("run", "mlir"),
        default="run",
    )
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
