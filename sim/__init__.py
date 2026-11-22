"""AI Racing Project - racing environment."""
from .car_numpy import CarParams
from .env import OBS_NAMES, RacingEnv
from .track import Track, load_track, wrap_angle

__all__ = ["RacingEnv", "Track", "load_track", "CarParams", "OBS_NAMES", "wrap_angle"]

# Register with Gymnasium so `gym.make("Racing-v0", **kwargs)` works (kwargs go to RacingEnv).
# Creating the env through gym.make also lets Gymnasium's check_env test every render mode.
try:
    import gymnasium as _gym

    if "Racing-v0" not in _gym.registry:
        _gym.register(id="Racing-v0", entry_point="sim.env:RacingEnv")
except ImportError:  # Gymnasium is optional
    pass
