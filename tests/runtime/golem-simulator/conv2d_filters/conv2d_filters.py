import argparse

import torch
from torch_mlir import fx


class MultiFilterModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 3, bias=True)

        with torch.no_grad():
            weights = torch.arange(
                1,
                6 * 1 * 3 * 3 + 1,
                dtype=torch.float32,
            ).reshape(6, 1, 3, 3) / 100.0
            self.conv1.weight.copy_(weights)
            self.conv1.bias.zero_()

    def forward(self, x):
        return self.conv1(x)


model = MultiFilterModel()
model.eval()

x = torch.tensor(
    [[[[1.0, 2.0, 3.0, 4.0, 5.0],
       [6.0, 7.0, 8.0, 9.0, 10.0],
       [11.0, 12.0, 13.0, 14.0, 15.0],
       [16.0, 17.0, 18.0, 19.0, 20.0],
       [21.0, 22.0, 23.0, 24.0, 25.0]]]],
    dtype=torch.float32,
)


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
