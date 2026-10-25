import json
import random
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
from tqdm import trange

from utils.config import CHECKPOINT, CONTROL_HZ, DEVICE, IMAGE_SIZE, RESULTS
from utils.metrics import episode_metrics

OBJECT_TOKENS = (
    "bowl",
    "soup",
    "sauce",
    "milk",
    "plate",
    "stove",
    "cabinet",
    "moka",
    "ramekin",
    "basket",
    "pudding",
    "butter",
    "juice",
    "ketchup",
    "cheese",
    "dressing",
    "drawer",
)


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _sim(vec):
    return vec.envs[0]._env.env.sim


def _body_map(sim):
    names = {}
    for index in range(sim.model.nbody):
        name = sim.model.body(index).name
        if any(token in name for token in OBJECT_TOKENS):
            names[name] = np.array(sim.data.body(index).xpos, dtype=np.float64)
    return names


def _hit(bodies, token):
    hits = [name for name in bodies if token in name]
    mains = [name for name in hits if name.endswith("_main")]
    chosen = hits if len(hits) == 1 else mains
    if len(chosen) != 1:
        raise ValueError(f"token {token!r} matched {hits} among {sorted(bodies)}")
    return chosen[0]


def _one(bodies, token):
    return bodies[_hit(bodies, token)]


def select_target_names(bodies, rule, tokens):
    if rule == "none":
        return []
    if rule == "token":
        return [_hit(bodies, token) for token in tokens]
    bowls = {name: pos for name, pos in bodies.items() if "bowl" in name}
    if not bowls:
        raise ValueError(f"no bowl in {sorted(bodies)}")
    if rule == "bowl_center":
        return [min(bowls, key=lambda name: float(np.linalg.norm(bowls[name][:2])))]
    if rule == "bowl_between":
        mid = (_one(bodies, "plate") + _one(bodies, "ramekin")) / 2.0
        return [min(bowls, key=lambda name: float(np.linalg.norm(bowls[name][:2] - mid[:2])))]
    if rule == "bowl_stove":
        stove = _one(bodies, "stove_1_main")
        return [min(bowls, key=lambda name: float(np.linalg.norm(bowls[name][:2] - stove[:2])))]
    raise ValueError(f"unknown target rule {rule}")


def _positions(sim, names):
    rows = []
    for name in names:
        index = sim.model.body(name).id
        rows.append(np.array(sim.data.body(index).xpos, dtype=np.float64))
    return np.stack(rows, axis=0)


class _CallMeter:
    def __init__(self, policy):
        self.n = 0
        self.seconds = 0.0
        self._original = policy._get_action_chunk
        self._policy = policy

    def install(self):
        meter = self

        def wrapped(*args, **kwargs):
            start = time.perf_counter()
            out = meter._original(*args, **kwargs)
            meter.seconds += time.perf_counter() - start
            meter.n += 1
            return out

        self._policy._get_action_chunk = wrapped

    def remove(self):
        self._policy._get_action_chunk = self._original


_FONT = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 13)


def _frame(image, text, replan):
    view = np.ascontiguousarray(image[::-1, ::-1])
    canvas = Image.fromarray(view)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, canvas.width, 20], fill=(0, 0, 0))
    draw.text((4, 3), text, font=_FONT, fill=(255, 220, 0))
    if replan:
        draw.rectangle([0, 20, canvas.width, 26], fill=(255, 32, 32))
    return np.asarray(canvas)


class Session:
    def __init__(self):
        import lerobot.policies.smolvla.configuration_smolvla  # noqa: F401  registers the config
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs import make_env, make_env_pre_post_processors
        from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
        from lerobot.policies import make_policy, make_pre_post_processors

        self._make_env = make_env
        self._make_env_processors = make_env_pre_post_processors
        self._env_config_cls = LiberoEnvConfig

        probe = LiberoEnvConfig(
            task="libero_spatial",
            task_ids=[0],
            observation_height=IMAGE_SIZE,
            observation_width=IMAGE_SIZE,
            fps=CONTROL_HZ,
        )
        policy_cfg = PreTrainedConfig.from_pretrained(CHECKPOINT)
        policy_cfg.pretrained_path = CHECKPOINT
        policy_cfg.device = DEVICE
        self.policy = make_policy(cfg=policy_cfg, env_cfg=probe)
        self.policy.eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=CHECKPOINT,
            preprocessor_overrides={"device_processor": {"device": DEVICE}},
        )
        self.env_preprocessor, self.env_postprocessor = self._make_env_processors(probe, policy_cfg)
        self.revision = str(getattr(self.policy.config, "revision", CHECKPOINT))
        RESULTS.mkdir(parents=True, exist_ok=True)

    def resolve(self, suite_name, match):
        from libero.libero import benchmark

        suite = benchmark.get_benchmark_dict()[suite_name]()
        hits = []
        for task_id in range(len(suite.tasks)):
            language = suite.get_task(task_id).language
            if match in language:
                hits.append((task_id, language))
        if len(hits) != 1:
            raise ValueError(f"{suite_name} match {match!r} found {hits}")
        return hits[0]

    def open_env(self, suite_name, task_id):
        cfg = self._env_config_cls(
            task=suite_name,
            task_ids=[task_id],
            observation_height=IMAGE_SIZE,
            observation_width=IMAGE_SIZE,
            fps=CONTROL_HZ,
        )
        created = self._make_env(cfg, n_envs=1, use_async_envs=False)
        return created[suite_name][task_id]


def run_episode(session, vec, spec, task_id, language, init_index, k, seed, out_dir, max_steps=None, save_video=False, gate=None, tag=None):
    from lerobot.envs.utils import NEW_ROLLOUT_OPTION, preprocess_observation
    from lerobot.utils.constants import ACTION

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{spec.get('block', 'demo')}_{spec['name']}_init{init_index:02d}_k{k:02d}"
    if tag:
        stem = f"{stem}_{tag}"
    _set_seed(seed)
    session.policy.config.n_action_steps = int(k)
    session.policy.reset()
    session.preprocessor.reset()
    session.postprocessor.reset()
    vec.envs[0].init_state_id = int(init_index)

    meter = _CallMeter(session.policy)
    meter.install()
    started = time.perf_counter()
    observation, info = vec.reset(seed=[seed], options={NEW_ROLLOUT_OPTION: True})
    bodies = _body_map(_sim(vec))
    target_names = select_target_names(bodies, spec["rule"], spec["tokens"])
    limit = vec.envs[0]._max_episode_steps if max_steps is None else max_steps
    writer = None
    if save_video:
        video_dir = out_dir.parent / "videos" if out_dir.name == "episodes" else out_dir
        video_dir.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(video_dir / f"{stem}.mp4", fps=CONTROL_HZ, codec="libx264", pixelformat="yuv420p")

    actions, eefs, targets, replans = [], [], [], []
    success = False
    first_action = None
    for step in trange(limit, desc=f"{spec['name']} k={k} init={init_index}", leave=False):
        queue = session.policy._queues[ACTION]
        queued = len(queue) == 0
        if queued and gate is not None:
            gate.reset_chunk()
        batch = preprocess_observation(observation)
        batch["task"] = list(vec.call("task_description"))
        batch = session.env_preprocessor(batch)
        batch = session.preprocessor(batch)
        with torch.inference_mode():
            action = session.policy.select_action(batch)
        action = session.postprocessor(action)
        action = session.env_postprocessor({ACTION: action})[ACTION]
        action_np = action.detach().to("cpu").numpy()
        if first_action is None:
            first_action = action_np[0].copy()
        observation, reward, terminated, truncated, info = vec.step(action_np)
        sim = _sim(vec)
        eef = np.asarray(observation["robot_state"]["eef"]["pos"][0], dtype=np.float64)
        if gate is not None:
            fingers = np.asarray(observation["robot_state"]["gripper"]["qpos"][0], dtype=np.float64)
            gate.after_step(session.policy._queues[ACTION], action_np[0], fingers)
        actions.append(action_np[0].copy())
        eefs.append(eef)
        if target_names:
            targets.append(_positions(sim, target_names))
        replans.append(queued)
        if "is_success" in info and bool(np.asarray(info["is_success"]).any()):
            success = True
        if writer is not None:
            label = f"K={k}   step {step}" + (f"   {tag}" if tag else "") + ("   new plan" if queued else "")
            writer.append_data(_frame(observation["pixels"]["image"][0], label, queued))
        if bool(np.asarray(terminated).any()) or bool(np.asarray(truncated).any()):
            break
    if writer is not None:
        writer.close()
    meter.remove()

    action_a = np.stack(actions)
    eef_a = np.stack(eefs)
    target_a = np.stack(targets) if targets else None
    counts = episode_metrics(action_a, eef_a, target_a, replans)
    record = {
        "block": spec.get("block", "demo"),
        "suite": spec["suite"],
        "name": spec["name"],
        "task_id": task_id,
        "language": language,
        "init_index": init_index,
        "k": int(k),
        "gate": None if gate is None else gate.mode,
        "seed": int(seed),
        "checkpoint": CHECKPOINT,
        "revision": session.revision,
        "success": success,
        "steps": int(len(actions)),
        "n_calls": meter.n,
        "policy_s": meter.seconds,
        "wall_s": time.perf_counter() - started,
        "target_names": target_names,
        "body_names": sorted(bodies),
        **counts,
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(record, indent=2))
    np.savez(
        out_dir / f"{stem}.npz",
        action=action_a,
        eef=eef_a,
        target=target_a if target_a is not None else np.zeros((0, 0, 3)),
        replan=np.asarray(replans, dtype=bool),
    )
    record["first_action"] = first_action
    return record
