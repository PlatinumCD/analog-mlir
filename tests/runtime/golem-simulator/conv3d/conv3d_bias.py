import argparse

import torch
from torch_mlir import fx


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(1, 1, 2, bias=True)

        with torch.no_grad():
            weights = torch.tensor(
                [[[[[0.05, 0.10],
                    [0.15, 0.20]],
                   [[0.25, 0.30],
                    [0.35, 0.40]]]]],
                dtype=torch.float32,
            )
            self.conv1.weight.copy_(weights)
            self.conv1.bias.zero_()

    def forward(self, x):
        return self.conv1(x)


model = SimpleModel()
model.eval()

x = torch.arange(
    1,
    4 * 4 * 4 + 1,
    dtype=torch.float32,
).reshape(1, 1, 4, 4, 4)


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
