import collections
import sys
import os
import dataclasses
import json
import logging
import math
import pathlib

import imageio
from pathlib import Path

_LIBERO_ROOT = Path("/fs/nexus-scratch/yliang17/Research/VLA/LIBERO")

for p in (str(_LIBERO_ROOT), ):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ["PYTHONPATH"] = os.pathsep.join(
    [str(_LIBERO_ROOT), os.environ.get("PYTHONPATH", "")]
)
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import cv2
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
import time

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8005
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_10"#,"libero_object","libero_goal","libero_10"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 10  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    results_dir: str = "results"
    model_name: str = "openpi"
    action_horizon: int = 10

    seed: int = 7  # Random Seed (for reproducibility)


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


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 270  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 300  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 320  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 700  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    # Start evaluation
    total_episodes, total_successes = 0, 0
    task_results = []
    # summery = []
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            current_skill_label = None
            done = False

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
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

                        # Query model to get action
                        result = client.infer(element)
                        action_chunk = result["actions"]
                        current_skill_label = _skill_overlay_label(result)
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    # Save preprocessed image for replay video with the current routed skill metadata.
                    replay_images.append(_draw_overlay_label(img, current_skill_label))

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}_{task_episodes}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
                codec="libx264",
            )
            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        task_success_rate = float(task_successes) / float(task_episodes)
        total_success_rate = float(total_successes) / float(total_episodes)
        logging.info(f"Current task success rate: {task_success_rate}")
        logging.info(f"Current total success rate: {total_success_rate}")
        task_results.append(
            {
                "task_id": task_id,
                "task_description": task_description,
                "episodes": task_episodes,
                "successes": task_successes,
                "success_rate": task_success_rate,
            }
        )
        # summery.append(f"{total_successes / total_episodes * 100:.1f}")

    success_rate = float(total_successes) / float(total_episodes)
    logging.info(f"Total success rate: {success_rate}")
    logging.info(f"Total episodes: {total_episodes}")
    message = f"Current total success rate: {success_rate:.4f}\n"  # 可选：格式化小数位

    with open("success_rate1.txt", "a") as f:
        f.write(message)
    results_dir = pathlib.Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    model_name = args.model_name
    log_suffix = f"model{model_name}_task{args.task_suite_name}_seed{args.seed}_h{args.action_horizon}"
    result_path = f"{results_dir}/libero_eval_{log_suffix}.json"
    result = {
        "model_name": model_name,
        "task_suite_name": args.task_suite_name,
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
    # print(summery)


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    # import ipdb;ipdb.set_trace()
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
