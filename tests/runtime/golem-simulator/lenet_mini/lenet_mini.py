import argparse

import torch
from torch_mlir import fx


class MiniLeNet5(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 2, kernel_size=2, stride=1, padding=0, bias=True)
        self.tanh1 = torch.nn.Tanh()
        self.pool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = torch.nn.Conv2d(2, 4, kernel_size=2, stride=1, padding=0, bias=True)
        self.tanh2 = torch.nn.Tanh()

        self.fc1 = torch.nn.Linear(4, 6, bias=True)
        self.tanh3 = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(6, 10, bias=True)

        with torch.no_grad():
            self.conv1.weight.copy_(torch.tensor(
                [
                    [[[0.05, 0.06], [0.07, 0.08]]],
                    [[[0.09, 0.10], [0.11, 0.12]]],
                ],
                dtype=torch.float32,
            ))
            self.conv1.bias.copy_(torch.tensor([0.01, 0.02], dtype=torch.float32))

            self.conv2.weight.copy_(torch.tensor(
                [
                    [
                        [[0.020, 0.025], [0.030, 0.035]],
                        [[0.040, 0.045], [0.050, 0.055]],
                    ],
                    [
                        [[0.060, 0.065], [0.070, 0.075]],
                        [[0.080, 0.085], [0.090, 0.095]],
                    ],
                    [
                        [[0.100, 0.105], [0.110, 0.115]],
                        [[0.120, 0.125], [0.130, 0.135]],
                    ],
                    [
                        [[0.140, 0.145], [0.150, 0.155]],
                        [[0.160, 0.165], [0.170, 0.175]],
                    ],
                ],
                dtype=torch.float32,
            ))
            self.conv2.bias.copy_(torch.tensor([0.03, 0.04, 0.05, 0.06], dtype=torch.float32))

            self.fc1.weight.copy_(torch.tensor(
                [
                    [0.010, 0.012, 0.014, 0.016],
                    [0.018, 0.020, 0.022, 0.024],
                    [0.026, 0.028, 0.030, 0.032],
                    [0.034, 0.036, 0.038, 0.040],
                    [0.042, 0.044, 0.046, 0.048],
                    [0.050, 0.052, 0.054, 0.056],
                ],
                dtype=torch.float32,
            ))
            self.fc1.bias.copy_(torch.tensor([0.04, 0.05, 0.06, 0.07, 0.08, 0.09], dtype=torch.float32))

            self.fc2.weight.copy_(torch.tensor(
                [
                    [0.005, 0.006, 0.007, 0.008, 0.009, 0.010],
                    [0.011, 0.012, 0.013, 0.014, 0.015, 0.016],
                    [0.017, 0.018, 0.019, 0.020, 0.021, 0.022],
                    [0.023, 0.024, 0.025, 0.026, 0.027, 0.028],
                    [0.029, 0.030, 0.031, 0.032, 0.033, 0.034],
                    [0.035, 0.036, 0.037, 0.038, 0.039, 0.040],
                    [0.041, 0.042, 0.043, 0.044, 0.045, 0.046],
                    [0.047, 0.048, 0.049, 0.050, 0.051, 0.052],
                    [0.053, 0.054, 0.055, 0.056, 0.057, 0.058],
                    [0.059, 0.060, 0.061, 0.062, 0.063, 0.064],
                ],
                dtype=torch.float32,
            ))
            self.fc2.bias.copy_(torch.tensor(
                [0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050, 0.055, 0.060, 0.065],
                dtype=torch.float32,
            ))

    def forward(self, x):
        x = self.conv1(x)
        x = self.tanh1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.tanh2(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.tanh3(x)
        x = self.fc2(x)

        return x


model = MiniLeNet5()
model.eval()

x = torch.arange(1, 1 * 1 * 5 * 5 + 1, dtype=torch.float32).reshape(1, 1, 5, 5)


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
