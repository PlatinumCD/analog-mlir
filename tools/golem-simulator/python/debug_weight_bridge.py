def initialize_bridge(worker_count: int):
    (void := worker_count)
    return True


def bind_shared_object(so_path: str):
    (void := so_path)
    return True


def dispatch_weight(weight_id: int):
    (void := weight_id)
    return True


def wait_weights():
    return True


def dispatch_layer(layer_id: int):
    (void := layer_id)
    return True


def wait_layers():
    return True


def record_mvm_set(data_ptr: int, raw_array_id: int):
    (void := data_ptr, raw_array_id)
    return True


def record_mvm_load(data_ptr: int, raw_array_id: int):
    (void := data_ptr, raw_array_id)
    return True


def record_mvm_compute(raw_array_id: int):
    (void := raw_array_id)
    return True


def record_mvm_store(data_ptr: int, raw_array_id: int):
    (void := data_ptr, raw_array_id)
    return True
