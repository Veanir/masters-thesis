"""Atomic weights-only recovery with exact bindings, RNG and log-prefix checks."""

import hashlib
import json
import os
import random

import numpy as np

from scripts.common.paths import script_help

script_help(__doc__, __name__)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, model, optimizer, step, binding, schedule_sha256, log_path):
    import torch

    state = np.random.get_state()
    payload = {
        "delta": {
            n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad
        },
        "optimizer": optimizer.state_dict(),
        "step": step,
        "binding": binding,
        "schedule_sha256": schedule_sha256,
        "log_sha256": digest(log_path),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "numpy_rng": [state[0], state[1].tolist(), state[2], state[3], state[4]],
        "python_rng": random.getstate(),
    }
    temporary = path.with_suffix(".pending")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def restore(path, model, optimizer, binding, schedule_sha256, log_path):
    import torch

    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        saved = torch.load(path, map_location="cpu", weights_only=True)
    assert saved["binding"] == binding and saved["schedule_sha256"] == schedule_sha256, (
        "Recovery binding mismatch"
    )
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert set(saved["delta"]) == set(trainable)
    for name, parameter in trainable.items():
        value = saved["delta"][name]
        assert value.shape == parameter.shape and torch.isfinite(value).all()
        parameter.data.copy_(value)
    optimizer.load_state_dict(saved["optimizer"])
    lines = log_path.read_bytes().splitlines(keepends=True)
    step = saved["step"]
    assert 0 <= step <= len(lines)
    prefix = b"".join(lines[:step])
    assert hashlib.sha256(prefix).hexdigest() == saved["log_sha256"]
    assert [json.loads(line)["step"] for line in lines[:step]] == list(range(1, step + 1))
    if len(lines) > step:
        # Keep abandoned work as evidence; canonical log contains each update once.
        tail = log_path.with_name(
            "abandoned-after-step-" + str(step) + "-" + digest(log_path)[:12] + ".jsonl"
        )
        assert not tail.exists()
        tail.write_bytes(b"".join(lines[step:]))
        log_path.write_bytes(prefix)
    torch.set_rng_state(saved["torch_rng"])
    torch.cuda.set_rng_state_all(saved["cuda_rng"])
    n = saved["numpy_rng"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    random.setstate(saved["python_rng"])
    return step
