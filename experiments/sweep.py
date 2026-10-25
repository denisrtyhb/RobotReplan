import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import shutil

from tqdm import tqdm

from utils.config import EPISODES, HORIZONS, RESULTS, SWEEP
from utils.metrics import write_summary
from utils.rollout import Session, run_episode


def main():
    free_gb = shutil.disk_usage(RESULTS if RESULTS.exists() else RESULTS.parent).free / 1024**3
    if free_gb < 8:
        raise SystemExit(f"stopping: {free_gb:.1f} GB free on the results disk")
    session = Session()
    EPISODES.mkdir(parents=True, exist_ok=True)
    jobs = [(spec, init, k) for spec in SWEEP for init in spec["inits"] for k in HORIZONS]
    env = None
    open_key = None
    for spec, init, k in tqdm(jobs, desc="sweep"):
        stem = EPISODES / f"{spec['block']}_{spec['name']}_init{init:02d}_k{k:02d}.json"
        if stem.exists():
            continue
        key = (spec["suite"], spec["match"])
        if key != open_key:
            if env is not None:
                env.close()
            task_id, language = session.resolve(spec["suite"], spec["match"])
            env = session.open_env(spec["suite"], task_id)
            open_key = key
        record = run_episode(session, env, spec, task_id, language, init, k, seed=init, out_dir=EPISODES)
        print(
            f"{spec['block']} {spec['name']} init={init} k={k} "
            f"success={record['success']} calls={record['n_calls']} policy_s={record['policy_s']:.1f}",
            flush=True,
        )
    if env is not None:
        env.close()
    summary = write_summary(EPISODES, RESULTS / "summary.json")
    print(f"wrote {len(summary)} summary rows")


if __name__ == "__main__":
    main()
