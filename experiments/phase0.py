import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from utils.config import PHASE0, SMOKE
from utils.rollout import Session, run_episode


def main():
    session = Session()
    task_id, language = session.resolve(SMOKE["suite"], SMOKE["match"])
    env = session.open_env(SMOKE["suite"], task_id)
    short = run_episode(session, env, SMOKE, task_id, language, 0, 1, 0, PHASE0, max_steps=50)
    long = run_episode(session, env, SMOKE, task_id, language, 0, 50, 0, PHASE0, max_steps=50)
    env.close()
    print("bodies", short["body_names"])
    print("target", short["target_names"])
    if short["n_calls"] != 50 or long["n_calls"] != 1:
        raise SystemExit(f"call counts {short['n_calls']} and {long['n_calls']}")
    if not np.allclose(short["first_action"], long["first_action"], atol=1e-4):
        raise SystemExit(f"first actions differ\n{short['first_action']}\n{long['first_action']}")
    print("phase0 ok", language)


if __name__ == "__main__":
    main()
