import argparse

import torch
import torch.export
from torch_mlir import fx


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn0 = torch.nn.RNNCell(4, 3, bias=True)
        self.rnn1 = torch.nn.RNNCell(3, 3, bias=True)

        with torch.no_grad():
            weight_ih0 = torch.arange(
                1,
                3 * 4 + 1,
                dtype=torch.float32,
            ).reshape(3, 4) / 100.0
            weight_hh0 = torch.arange(
                1,
                3 * 3 + 1,
                dtype=torch.float32,
            ).reshape(3, 3) / 200.0
            bias_ih0 = torch.arange(1, 3 + 1, dtype=torch.float32) / 50.0
            bias_hh0 = torch.arange(1, 3 + 1, dtype=torch.float32) / 100.0

            weight_ih1 = torch.arange(
                1,
                3 * 3 + 1,
                dtype=torch.float32,
            ).reshape(3, 3) / 150.0
            weight_hh1 = torch.arange(
                1,
                3 * 3 + 1,
                dtype=torch.float32,
            ).reshape(3, 3) / 250.0
            bias_ih1 = torch.arange(1, 3 + 1, dtype=torch.float32) / 60.0
            bias_hh1 = torch.arange(1, 3 + 1, dtype=torch.float32) / 120.0

            self.rnn0.weight_ih.copy_(weight_ih0)
            self.rnn0.weight_hh.copy_(weight_hh0)
            self.rnn0.bias_ih.copy_(bias_ih0)
            self.rnn0.bias_hh.copy_(bias_hh0)

            self.rnn1.weight_ih.copy_(weight_ih1)
            self.rnn1.weight_hh.copy_(weight_hh1)
            self.rnn1.bias_ih.copy_(bias_ih1)
            self.rnn1.bias_hh.copy_(bias_hh1)

    def forward(self, x, hx0, hx1):
        h0 = self.rnn0(x, hx0)
        h1 = self.rnn1(h0, hx1)
        return h1


model = SimpleModel()
model.eval()

x = torch.arange(1, 4 + 1, dtype=torch.float32).reshape(1, 4)
hx0 = torch.arange(1, 3 + 1, dtype=torch.float32).reshape(1, 3) / 10.0
hx1 = torch.arange(4, 6 + 1, dtype=torch.float32).reshape(1, 3) / 10.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("run", "mlir"), default="run")
    args = parser.parse_args()

    if args.mode == "mlir":
        exported_program = torch.export.export(model, (x, hx0, hx1))
        exported_program = exported_program.run_decompositions()
        mlir_module = fx.export_and_import(
            exported_program,
            output_type="torch",
            func_name="forward",
        )
        print(mlir_module)
        return

    with torch.no_grad():
        print(model(x, hx0, hx1))


if __name__ == "__main__":
    main()
