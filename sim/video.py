"""Record a driven episode to an MP4 file.

Works on every Python version (uses imageio + its bundled ffmpeg). Gymnasium's own
RecordVideo wrapper also works with RacingEnv(render_mode="rgb_array"), but it needs
moviepy, which currently can't be installed on Python 3.15.

    from sim import RacingEnv
    from sim.video import record_episode

    env = RacingEnv(render_mode="rgb_array", max_laps=1)
    record_episode(env, lambda obs, info: my_controller(obs, info), "lap.mp4")
"""
from __future__ import annotations


def save_video(frames, path, fps=20):
    """Write a list of RGB frames (height x width x 3, uint8) to an MP4 file."""
    try:
        import imageio.v2 as imageio
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("saving videos needs `pip install imageio[ffmpeg]`") from exc
    imageio.mimwrite(path, frames, fps=fps, codec="libx264", macro_block_size=16)


def record_episode(env, act, path, seed=None, options=None, max_steps=None):
    """Run one episode with act(obs, info) -> action and save it as an MP4 (one frame per
    50 ms step, so it plays in real time). env must use render_mode="rgb_array".
    Returns the last info dict."""
    if getattr(env, "render_mode", None) != "rgb_array":
        raise ValueError('record_episode needs an env made with render_mode="rgb_array"')
    obs, info = env.reset(seed=seed, options=options)
    frames = [env.render()]
    steps, done = 0, False
    while not done and (max_steps is None or steps < max_steps):
        obs, reward, terminated, truncated, info = env.step(act(obs, info))
        frames.append(env.render())
        steps += 1
        done = terminated or truncated
    save_video(frames, path, fps=env.metadata.get("render_fps", 20))
    return info
