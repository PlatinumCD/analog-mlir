import argparse

import torch
import torch.export
from torch_mlir import fx


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.RNNCell(4, 3, bias=True)

        with torch.no_grad():
            weight_ih = torch.arange(
                1,
                3 * 4 + 1,
                dtype=torch.float32,
            ).reshape(3, 4) / 100.0
            weight_hh = torch.arange(
                1,
                3 * 3 + 1,
                dtype=torch.float32,
            ).reshape(3, 3) / 200.0
            bias_ih = torch.arange(1, 3 + 1, dtype=torch.float32) / 50.0
            bias_hh = torch.arange(1, 3 + 1, dtype=torch.float32) / 100.0

            self.rnn.weight_ih.copy_(weight_ih)
            self.rnn.weight_hh.copy_(weight_hh)
            self.rnn.bias_ih.copy_(bias_ih)
            self.rnn.bias_hh.copy_(bias_hh)

    def forward(self, x, hx):
        return self.rnn(x, hx)


model = SimpleModel()
model.eval()

x = torch.arange(1, 4 + 1, dtype=torch.float32).reshape(1, 4)
hx = torch.arange(1, 3 + 1, dtype=torch.float32).reshape(1, 3) / 10.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("run", "mlir"), default="run")
    args = parser.parse_args()

    if args.mode == "mlir":
        exported_program = torch.export.export(model, (x, hx))
        exported_program = exported_program.run_decompositions()
        mlir_module = fx.export_and_import(
            exported_program,
            output_type="torch",
            func_name="forward",
        )
        print(mlir_module)
        return

    with torch.no_grad():
        print(model(x, hx))


if __name__ == "__main__":
    main()
