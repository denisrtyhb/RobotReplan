import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import shutil

from tqdm import tqdm

from utils.config import ADAPTIVE_BASE_K, ADAPTIVE_DIR, ADAPTIVE_MODES, RESULTS, SWEEP
from utils.gates import HorizonGate
from utils.rollout import Session, run_episode


def main():
    free_gb = shutil.disk_usage(RESULTS if RESULTS.exists() else RESULTS.parent).free / 1024**3
    if free_gb < 8:
        raise SystemExit(f"stopping: {free_gb:.1f} GB free on the results disk")
    session = Session()
    ADAPTIVE_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(spec, init, mode) for spec in SWEEP for init in spec["inits"] for mode in ADAPTIVE_MODES]
    env = None
    open_key = None
    task_id = language = None
    for spec, init, mode in tqdm(jobs, desc="adaptive"):
        stem = ADAPTIVE_DIR / f"{spec['block']}_{spec['name']}_init{init:02d}_k{ADAPTIVE_BASE_K:02d}_{mode}.json"
        if stem.exists():
            continue
        key = (spec["suite"], spec["match"])
        if key != open_key:
            if env is not None:
                env.close()
            task_id, language = session.resolve(spec["suite"], spec["match"])
            env = session.open_env(spec["suite"], task_id)
            open_key = key
        record = run_episode(
            session, env, spec, task_id, language, init, ADAPTIVE_BASE_K,
            seed=init, out_dir=ADAPTIVE_DIR, gate=HorizonGate(mode), tag=mode,
        )
        print(f"{spec['name']} init={init} {mode} success={record['success']} calls={record['n_calls']}", flush=True)
    if env is not None:
        env.close()


if __name__ == "__main__":
    main()
