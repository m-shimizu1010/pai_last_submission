import argparse
import os
import pickle
from importlib import metadata
import torch

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from jump_env import JumpEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    parser.add_argument("--ckpt", type=int, default=100)
    args = parser.parse_args()

    gs.init()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}

    env = JumpEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    # record_video = True             

    # if record_video:
    #     env.cam.start_recording()  

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=gs.device)


    # import signal, sys

    # def clean_exit(*_):
    #     env.cam.stop_recording("jump_demo.mp4", fps=int(1/env.dt))
    #     # gs.shutdown()
    #     sys.exit(0)

    # signal.signal(signal.SIGINT, clean_exit)   # Ctrl‑C
    # signal.signal(signal.SIGTERM, clean_exit)  # kill など

    obs, _ = env.reset()
    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, rews, dones, infos = env.step(actions)

    # with torch.no_grad():
    #     while True:
    #         actions = policy(obs.to(gs.device))
    #         obs, _, dones, _ = env.step(actions)
    #         env.cam.render()
    #         if dones.any():
    #             obs, _ = env.reset()

    # with torch.no_grad():
    #     while env.scene.viewer.is_running():   # ← 視聴ウィンドウが開いている間だけ
    #         actions = policy(obs.to(gs.device))
    #         obs, _, dones, _ = env.step(actions)
    #         env.cam.render()
    #         if dones.any():
    #             obs, _ = env.reset()

    #     env.cam.stop_recording("jump_demo.mp4", fps=int(1/env.dt))



if __name__ == "__main__":
    main()

"""
# evaluation
python examples/locomotion/go2_eval.py -e go2-walking -v --ckpt 100
"""
