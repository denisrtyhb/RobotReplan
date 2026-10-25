import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm

from utils.config import DEMO_HORIZONS, DEMOS, EPISODES
from utils.rollout import Session, run_episode


def main():
    session = Session()
    jobs = [(spec, init, k) for spec in DEMOS for init in (0,) for k in DEMO_HORIZONS]
    for spec, init, k in tqdm(jobs, desc="demos"):
        task_id, language = session.resolve(spec["suite"], spec["match"])
        env = session.open_env(spec["suite"], task_id)
        record = run_episode(
            session, env, spec, task_id, language, init, k, seed=init, out_dir=EPISODES, save_video=True
        )
        env.close()
        print(f"{spec['name']} k={k} success={record['success']} calls={record['n_calls']}", flush=True)


if __name__ == "__main__":
    main()
