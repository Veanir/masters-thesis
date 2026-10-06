"""Run the unchanged frozen trainer with a separately bound numerical amendment."""

import hashlib
import json
import os
import runpy
import sys
import time
from pathlib import Path

from scripts.common.paths import script_help

script_help(__doc__, __name__)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    assert sys.argv[1] == "--amendment-control"
    control = Path(sys.argv[2])
    launch = json.loads((control / "launch.json").read_text())
    assert sha(__file__) == launch["training_wrapper_sha256"]
    trainer = Path(launch["trainer"])
    assert sha(trainer) == launch["trainer_sha256"]
    arguments = sys.argv[3:]
    target = Path(arguments[arguments.index("--output") + 1])
    arm = arguments[arguments.index("--arm") + 1]
    assert arm in ["BASE", "CLASSIC", "PHOTO"] and target.parent == Path(
        launch["result_dir_absolute"]
    )
    os.environ.update(CUBLAS_WORKSPACE_CONFIG=":4096:8")
    sys.path[:0] = [
        str(trainer.parent),
        str(Path(arguments[arguments.index("--work") + 1]) / "vendor/rayst3r"),
    ]
    import engine
    import torch

    import scripts.train.confidence_exponential as amendment

    assert sha(amendment.__file__) == launch["amendment_sha256"]
    amendment.install()
    step = 0
    if "--resume" in arguments:
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            ck = torch.load(target / "recovery.pt", map_location="cpu", weights_only=True)
        step = ck["step"]
        del ck
    start = step
    original_eval = engine.eval_model

    def evaluate(*args, **kwargs):
        nonlocal step
        step += 1
        amendment.CONTEXT.update(step=step, arm=arm, seed=launch["seed"])
        return original_eval(*args, **kwargs)

    engine.eval_model = evaluate
    receipt = {
        "status": "running",
        "seed": launch["seed"],
        "arm": arm,
        "start_checkpoint_step": start,
        "launch_sha256": sha(control / "launch.json"),
        "amendment_sha256": sha(amendment.__file__),
        "training_wrapper_sha256": sha(__file__),
        "frozen_trainer_sha256": sha(trainer),
        "started_unix": time.time(),
    }
    sys.argv = [str(trainer), *arguments]
    try:
        runpy.run_path(str(trainer), run_name="__main__")
        receipt["status"] = "completed"
    finally:
        receipt.update(observed_step=step, events=amendment.EVENTS, finished_unix=time.time())
        if target.exists():
            path = target / (
                "numerical-amendment-" + str(start) + "-" + str(time.time_ns()) + ".json"
            )
            path.write_text(json.dumps(receipt, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
