"""预建 11 个 holdout repo 镜像（评测池全部 profile），2 路并行。"""
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, "/data/wangshenghua/wsh/teacher_data")
import envs_v2 as V2

REPOS = [
    "aio-libs__async-timeout.d0baa9f1",
    "alanjds__drf-nested-routers.6144169d",
    "borntyping__python-colorlog.dfa10f59",
    "buriy__python-readability.40256f40",
    "gruns__icecream.f76fef56",
    "kennethreitz__records.5941ab27",
    "mewwts__addict.75284f95",
    "rustedpy__result.0b855e1e",
    "termcolor__termcolor.3a42086f",
]

log = Path("/data/wangshenghua/wsh/teacher_data/outputs/eval/base_v1/logs/build_holdout_envs.log")
lock = threading.Lock()


def build_all(tag: str, items: list[str]) -> None:
    for p in items:
        info = V2.build_profile_image(p)
        with lock:
            with log.open("a") as f:
                f.write(f"[{tag}] {json.dumps(info, ensure_ascii=False)}\n")
                f.flush()


if __name__ == "__main__":
    a, b = REPOS[:5], REPOS[5:]
    t1 = threading.Thread(target=build_all, args=("A", a))
    t2 = threading.Thread(target=build_all, args=("B", b))
    t1.start(); t2.start(); t1.join(); t2.join()
    with log.open("a") as f:
        f.write("[done] all builds attempted\n")
