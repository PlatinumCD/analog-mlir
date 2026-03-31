import numpy as np


class SimulatedCore:
    def __init__(self, num_arrays, array_rows, array_cols, core_index=0):
        self.num_arrays = num_arrays
        self.array_rows = array_rows
        self.array_cols = array_cols
        self.core_index = core_index
        self.arrays = [
            np.zeros((array_rows, array_cols), dtype=np.float32)
            for _ in range(num_arrays)
        ]
        self.input_buffers = [
            np.zeros(array_rows, dtype=np.float32) for _ in range(num_arrays)
        ]
        self.output_buffers = [
            np.zeros(array_cols, dtype=np.float32) for _ in range(num_arrays)
        ]

    def _check_array_index(self, array_index):
        if not 0 <= array_index < self.num_arrays:
            raise IndexError(
                f"array_index {array_index} out of bounds for {self.num_arrays} arrays"
            )

    def set_array(self, data, array_index):
        self._check_array_index(array_index)
        self.arrays[array_index] = np.asarray(data, dtype=np.float32)
        print(
            f"[simulated core {self.core_index}] set matrix {array_index}:"
            f"\n{self.arrays[array_index]}"
        )

    def load_input(self, data, array_index):
        self._check_array_index(array_index)
        self.input_buffers[array_index] = np.asarray(data, dtype=np.float32)
        print(
            f"[simulated core {self.core_index}] loaded input {array_index}:"
            f"\n{self.input_buffers[array_index]}"
        )

    def compute(self, array_index):
        self._check_array_index(array_index)
        self.output_buffers[array_index] = np.asarray(
            self.arrays[array_index] @ self.input_buffers[array_index],
            dtype=np.float32,
        )
        print(
            f"[simulated core {self.core_index}] computed output {array_index}:"
            f"\n{self.output_buffers[array_index]}"
        )

    def store_output(self, array_index):
        self._check_array_index(array_index)
        return self.output_buffers[array_index]
