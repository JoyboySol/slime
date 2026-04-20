import os
import sys
from pathlib import Path

import ray

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


if __name__ == "__main__":
    for key in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]:
        os.environ.pop(key, None)
    os.environ["NO_PROXY"] = ",".join(filter(None, [os.environ.get("NO_PROXY"), "127.0.0.1", "localhost", os.environ.get("MASTER_ADDR", "")]))
    num_gpus = int(os.environ.get("RAY_NUM_GPUS", os.environ.get("NUM_GPUS", "4")))
    print(f"[driver] ray.init(local) begin num_gpus={num_gpus} dashboard_disabled=1", flush=True)
    ray.init(
        num_gpus=num_gpus,
        include_dashboard=False,
        log_to_driver=False,
    )
    print("[driver] ray.init(local) done", flush=True)

    print("[driver] importing parse_args/train", flush=True)
    from slime.utils.arguments import parse_args
    from train import train
    print("[driver] imports done", flush=True)

    print("[driver] parse_args begin", flush=True)
    args = parse_args()
    print("[driver] parse_args done", flush=True)
    print("[driver] train(args) begin", flush=True)
    train(args)
    print("[driver] train(args) done", flush=True)
