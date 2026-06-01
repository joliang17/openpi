import collections
import sys
import os
import argparse
import json
import logging
import math
import pathlib
import shutil
import tempfile
import traceback

import websockets.exceptions

import cv2
import imageio
import numpy as np
import torch
import tqdm

from pathlib import Path

_LIBERO_ROOT = Path("/fs/nexus-scratch/yliang17/Research/VLA/LIBERO")
_LIBERO_PRO_REPO = Path("/fs/nexus-scratch/yliang17/Research/VLA/LIBERO-PRO")
_LIBERO_PRO_DATASET = Path("/fs/nexus-projects/wilddiffusion/vla/libero_pro")

for p in (str(_LIBERO_ROOT), str(_LIBERO_PRO_REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ["PYTHONPATH"] = os.pathsep.join(
    [str(_LIBERO_ROOT), str(_LIBERO_PRO_REPO), os.environ.get("PYTHONPATH", "")]
)

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

BAD_ACTION_SINGLE = np.array([0.078563, 0.0360005, 0.065438, -0.00272493, -0.00059135, -0.01451939, -0.0001995])

PERTURB_SUFFIX = {
    "object": "_object",
    "position": "_swap",
}

PERTURBED_SUITES = {"libero_spatial", "libero_goal", "libero_object", "libero_10"}

MAX_STEPS_MAP = {
    "libero_spatial": 270,
    "libero_object": 300,
    "libero_goal": 320,
    "libero_10": 700,
    "libero_90": 400,
}

# Mirrors embed_sigma in tokenizer.py
SIGMA_INV = {0.0: "pick", 1.0: "place", 2.0: "open", 3.0: "close", 4.0: "turn"}


def _skill_overlay_label(result: dict) -> str | None:
    skill_name = result.get("skill_name")
    if skill_name is None:
        return None

    parts = [f"skill={skill_name}"]
    if "skill_prob" in result:
        parts[0] += f" p={float(result['skill_prob']):.3f}"
    if "skill_action_gate_prob" in result:
        parts.append(f"action_gate={float(result['skill_action_gate_prob']):.3f}")
    if "skill_effect_gate_prob" in result:
        parts.append(f"effect_gate={float(result['skill_effect_gate_prob']):.3f}")
    return "\n".join(parts)


def _draw_overlay_label(image: np.ndarray, label: str | None) -> np.ndarray:
    if not label:
        return image

    frame = image.copy()
    lines = str(label).splitlines()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.42
    thickness = 1
    pad = 4
    line_gap = 3
    sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    width = max(w for w, _ in sizes) + 2 * pad
    line_height = max(h for _, h in sizes)
    box_height = len(lines) * line_height + (len(lines) - 1) * line_gap + 2 * pad
    cv2.rectangle(frame, (0, 0), (width, box_height), (0, 0, 0), -1)
    y = pad + line_height
    for line in lines:
        cv2.putText(frame, line, (pad, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += line_height + line_gap
    return frame


# ---------------------------------------------------------------------------
# Skill / expert extraction
# ---------------------------------------------------------------------------

def _skill_and_expert(result: dict) -> tuple[str, str]:
    """Return (vlm_skill_name, expert_label) from inference result.

    VLM skill = skill keyword embedded into atomic_token by the tokenizer.
    Expert used = same index (router is identity-initialized from sigma_emb).
    """
    tok = result.get("atomic_token")
    if tok is not None:
        idx = int(round(float(tok)))
        name = SIGMA_INV.get(float(idx), f"unknown({tok})")
        return name, f"{name}(expert{idx})"
    return "unknown", "unknown"


def _infer_with_reconnect(client, element: dict, max_retries: int = 3) -> dict:
    """Call client.infer(), reconnecting on ConnectionClosedError."""
    for attempt in range(max_retries):
        try:
            return client.infer(element)
        except websockets.exceptions.ConnectionClosedError:
            if attempt == max_retries - 1:
                raise
            logging.warning(f"WebSocket connection lost, reconnecting (attempt {attempt + 1}/{max_retries})...")
            client._ws, client._server_metadata = client._wait_for_server()


# ---------------------------------------------------------------------------
# LIBERO-PRO helpers (ported from GR00T libero_pro_eval.py)
# ---------------------------------------------------------------------------

def bddl_stem_to_language(stem: str) -> str:
    return stem.replace("_", " ")


def get_perturbed_task_list(suite_name: str, perturb_type: str):
    suffix = PERTURB_SUFFIX[perturb_type]
    folder_name = f"{suite_name}{suffix}"
    bddl_dir = _LIBERO_PRO_DATASET / "bddl_files" / folder_name
    init_dir = _LIBERO_PRO_DATASET / "init_files" / folder_name

    if not bddl_dir.exists():
        raise FileNotFoundError(
            f"Perturbed BDDL dir not found: {bddl_dir}\n"
            f"Available: {[d.name for d in (_LIBERO_PRO_DATASET / 'bddl_files').iterdir()]}"
        )

    tasks = []
    for bddl_file in sorted(bddl_dir.glob("*.bddl")):
        stem = bddl_file.stem
        init_path = init_dir / f"{stem}.pruned_init"
        tasks.append({
            "bddl_path": str(bddl_file),
            "init_states_path": str(init_path) if init_path.exists() else None,
            "language": bddl_stem_to_language(stem),
        })
    return tasks


def make_env_from_bddl(bddl_path: str, resolution: int = LIBERO_ENV_RESOLUTION) -> OffScreenRenderEnv:
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(0)
    return env


def load_init_states(init_states_path: str):
    return torch.load(init_states_path)


def get_environment_perturbed_bddl(bddl_path: str, suite_name: str, task_name: str, seed: int = 42) -> str:
    from perturbation import BDDLCombinedPerturbator, PerturbFlags

    ood_configs = {
        "environment": str(_LIBERO_PRO_REPO / "libero_ood" / "ood_environment.yaml"),
    }
    perturbator = BDDLCombinedPerturbator(configs=ood_configs)
    flags = PerturbFlags(use_environment=True)

    with open(bddl_path, "r") as f:
        content = f.read()

    return perturbator.perturb_content(
        content=content,
        task_suite_name=suite_name,
        task_name=task_name,
        flags=flags,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# LIBERO env helper (from main.py)
# ---------------------------------------------------------------------------

def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def eval_libero_pro(args) -> None:
    print(f"Task suite:        {args.task_suite_name}")
    print(f"Perturbation type: {args.perturbation_type}")
    print(f"Replan steps:      {args.replan_steps}")

    log_dir = pathlib.Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_suffix = f"{args.task_suite_name}_{args.perturbation_type}_seed{args.seed}"
    log_path = log_dir / f"atomicvla_pro_{log_suffix}.log"
    log_file = open(log_path, "w")
    log_file.write(f"Task suite: {args.task_suite_name}\n")
    log_file.write(f"Perturbation type: {args.perturbation_type}\n")
    log_file.write(f"Replan steps: {args.replan_steps}\n")

    np.random.seed(args.seed)
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    max_steps = MAX_STEPS_MAP.get(args.task_suite_name, 600)

    # ---- build task list ----
    perturb_type = args.perturbation_type
    temp_bddl_dir = None

    if perturb_type == "none":
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[args.task_suite_name]()
        num_tasks = task_suite.n_tasks
        use_benchmark_api = True
    elif perturb_type in ("object", "position"):
        if args.task_suite_name not in PERTURBED_SUITES:
            raise ValueError(
                f"Perturbed assets for '{args.task_suite_name}' not available. "
                f"Supported suites: {PERTURBED_SUITES}"
            )
        task_list = get_perturbed_task_list(args.task_suite_name, perturb_type)
        num_tasks = len(task_list)
        use_benchmark_api = False
    elif perturb_type == "environment":
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[args.task_suite_name]()
        num_tasks = task_suite.n_tasks
        use_benchmark_api = False
    else:
        raise ValueError(f"Unknown perturbation type: {perturb_type}")

    print(f"Number of tasks: {num_tasks}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    total_episodes, total_successes = 0, 0
    task_results = []

    for task_id in tqdm.tqdm(range(num_tasks)):

        # -- set up env and task description --
        if perturb_type == "none":
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
            has_fixed_init = True
        elif perturb_type in ("object", "position"):
            t_info = task_list[task_id]
            task_description = t_info["language"]
            env = make_env_from_bddl(t_info["bddl_path"])
            initial_states = (
                load_init_states(t_info["init_states_path"])
                if t_info["init_states_path"] else None
            )
            has_fixed_init = initial_states is not None
        else:  # environment perturbation
            task = task_suite.get_task(task_id)
            task_name = task.name
            bddl_path = os.path.join(
                get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
            )
            perturbed_content = get_environment_perturbed_bddl(
                bddl_path, args.task_suite_name, task_name, seed=args.seed
            )
            if temp_bddl_dir is None:
                temp_bddl_dir = tempfile.mkdtemp(prefix="libero_pro_env_")
            temp_bddl_path = os.path.join(temp_bddl_dir, task.bddl_file)
            with open(temp_bddl_path, "w") as f:
                f.write(perturbed_content)
            task_description = task.language
            env = make_env_from_bddl(temp_bddl_path)
            has_fixed_init = False

        task_episodes, task_successes = 0, 0

        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            env.reset()
            if has_fixed_init:
                obs = env.set_init_state(initial_states[episode_idx])
            else:
                env.seed(episode_idx)
                obs = env.reset()

            action_plan = collections.deque()
            t = 0
            replay_images = []
            current_skill_label = None
            done = False

            logging.info(f"Starting episode {task_episodes + 1}...")
            log_file.write(f"Starting episode {task_episodes + 1}...\n")

            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    replay_img = image_tools.convert_to_uint8(img)

                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    if not action_plan:
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        # result = _infer_with_reconnect(client, element)
                        # while np.allclose(result["actions"], BAD_ACTION_SINGLE[None], atol=1e-6):
                        #     result = _infer_with_reconnect(client, element)

                        # action_chunk = result["actions"]
                        # assert len(action_chunk) >= args.replan_steps, (
                        #     f"replan_steps={args.replan_steps} but policy only returned {len(action_chunk)} actions"
                        # )
                        # action_plan.extend(action_chunk[: args.replan_steps])

                        # Query model to get action
                        result = client.infer(element)
                        action_chunk = result["actions"]
                        current_skill_label = _skill_overlay_label(result)
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    replay_images.append(_draw_overlay_label(replay_img, current_skill_label))

                    action = action_plan.popleft()
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    traceback.print_exc()
                    logging.error(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break

            task_episodes += 1
            total_episodes += 1

            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}_{task_episodes}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
                codec="libx264",
            )

            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        env.close()

        task_success_rate = float(task_successes) / float(task_episodes)
        total_success_rate = float(total_successes) / float(total_episodes)
        print(f"Current task success rate: {task_success_rate:.3f}")
        print(f"Current total success rate: {total_success_rate:.3f}")
        log_file.write(f"Current task success rate: {task_success_rate:.3f}\n")
        log_file.write(f"Current total success rate: {total_success_rate:.3f}\n")
        log_file.flush()
        task_results.append(
            {
                "task_id": task_id,
                "task_description": task_description,
                "episodes": task_episodes,
                "successes": task_successes,
                "success_rate": task_success_rate,
            }
        )

    if temp_bddl_dir and os.path.isdir(temp_bddl_dir):
        shutil.rmtree(temp_bddl_dir, ignore_errors=True)

    log_file.close()
    success_rate = total_successes / total_episodes
    print(f"\nFinal: {total_successes}/{total_episodes} = {success_rate * 100:.1f}%")

    results_dir = pathlib.Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    model_name = args.model_name
    log_suffix = f"model{model_name}_task{args.task_suite_name}_seed{args.seed}_h{args.action_horizon}"
    result_path = f"{results_dir}/libero_eval_{log_suffix}.json"
    result = {
        "model_name": model_name,
        "task_suite_name": args.task_suite_name,
        "perturbation_type": args.perturbation_type,
        "random_seed": args.seed,
        "action_horizon": args.action_horizon,
        "num_trials_per_task": args.num_trials_per_task,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "success_rate": success_rate,
        "success_percent": success_rate * 100.0,
        "task_results": task_results,
    }
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {result_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()

    # Model server
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8005)
    parser.add_argument("--resize_size", type=int, default=224)
    parser.add_argument("--replan_steps", type=int, default=5)

    # Task / environment
    parser.add_argument(
        "--task_suite_name", type=str, default="libero_10",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
    )
    parser.add_argument(
        "--perturbation_type", type=str, default="none",
        choices=["none", "object", "position", "environment"],
        help=(
            "'none': standard tasks; "
            "'object'/'position': pre-computed LIBERO-PRO perturbations; "
            "'environment': runtime BDDL perturbation."
        ),
    )
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--num_trials_per_task", type=int, default=10)

    # Output
    parser.add_argument("--video_out_path", type=str, default="data/libero_pro/videos")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--model_name", type=str, default="openpi")
    parser.add_argument("--action_horizon", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)

    args = parser.parse_args()
    eval_libero_pro(args)
