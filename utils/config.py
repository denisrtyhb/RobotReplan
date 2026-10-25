from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EPISODES = RESULTS / "episodes"
VIDEOS = RESULTS / "videos"
PHASE0 = RESULTS / "phase0"

CHECKPOINT = "HuggingFaceVLA/smolvla_libero"
DEVICE = "cuda"
IMAGE_SIZE = 256
CONTROL_HZ = 20
GRASP_RADIUS_M = 0.05
LIFT_M = 0.04
CORRECTION_DEG = 45.0

HORIZONS = (1, 10, 25, 50)
DEMO_HORIZONS = (1, 10, 50)

# Adaptive gates run on a fixed base queue, then shorten it.
ADAPTIVE_BASE_K = 25
ADAPTIVE_MODES = ("gripper", "fingers", "both")
ADAPTIVE_DIR = RESULTS / "adaptive"
CLOSED_KEEP = 10
CHECK_EVERY = 5
# Panda fingers: 0 is shut, ±0.04 m is open. Sum of absolute qpos below this is an empty close.
EMPTY_FINGER_SUM = 0.01

# `match` is a unique substring of the LIBERO language string.
# `rule` selects which simulator body is the grasp target.
SMOKE = {
    "suite": "libero_spatial",
    "match": "from table center",
    "name": "table_center",
    "rule": "bowl_center",
    "tokens": (),
}

DEMOS = (
    {
        "suite": "libero_spatial",
        "match": "between the plate and the ramekin",
        "name": "bowl_between",
        "rule": "bowl_between",
        "tokens": (),
    },
    {
        "suite": "libero_spatial",
        "match": "from table center",
        "name": "table_center",
        "rule": "bowl_center",
        "tokens": (),
    },
    {
        "suite": "libero_object",
        "match": "alphabet soup",
        "name": "alphabet_soup",
        "rule": "token",
        "tokens": ("alphabet_soup",),
    },
    {
        "suite": "libero_10",
        "match": "alphabet soup and the tomato sauce",
        "name": "soup_and_sauce",
        "rule": "token",
        "tokens": ("alphabet_soup", "tomato_sauce"),
    },
)

SWEEP = (
    {**DEMOS[0], "block": "E1", "inits": tuple(range(10))},
    {**DEMOS[1], "block": "E1", "inits": tuple(range(10))},
    {
        "suite": "libero_spatial",
        "match": "on the stove",
        "name": "bowl_stove",
        "rule": "bowl_stove",
        "tokens": (),
        "block": "E1",
        "inits": tuple(range(10)),
    },
    {**DEMOS[2], "block": "E2", "inits": tuple(range(10))},
    {
        "suite": "libero_object",
        "match": "tomato sauce",
        "name": "tomato_sauce",
        "rule": "token",
        "tokens": ("tomato_sauce",),
        "block": "E2",
        "inits": tuple(range(10)),
    },
    {
        "suite": "libero_object",
        "match": "milk",
        "name": "milk",
        "rule": "token",
        "tokens": ("milk",),
        "block": "E2",
        "inits": tuple(range(10)),
    },
    {**DEMOS[3], "block": "E3", "inits": tuple(range(5))},
    {
        "suite": "libero_10",
        "match": "turn on the stove and put the moka pot",
        "name": "stove_moka",
        "rule": "token",
        "tokens": ("moka",),
        "block": "E3",
        "inits": tuple(range(5)),
    },
    {
        "suite": "libero_goal",
        "match": "open the middle drawer",
        "name": "open_drawer",
        "rule": "none",
        "tokens": (),
        "block": "E4",
        "inits": tuple(range(10)),
    },
)
