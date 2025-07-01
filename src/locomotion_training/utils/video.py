"""Video rendering utilities for locomotion evaluation."""

import functools
from pathlib import Path
from typing import List, Optional, Callable, Any

import mujoco
import numpy as np
import jax.numpy as jp
import mediapy as media
from mujoco_playground._src.gait import draw_joystick_command


def setup_scene_options(show_contacts: bool = True,
                        show_perturbations: bool = True,
                        show_transparent: bool = False) -> mujoco.MjvOption:
    """Setup MuJoCo scene rendering options."""
    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[2] = True
    scene_option.geomgroup[3] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = show_contacts
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = show_transparent
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = show_perturbations
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
    
    return scene_option


def create_joystick_command_overlay(state: Any, 
                                   env: Any,
                                   command: jp.ndarray,
                                   scale_factor: float = 1.0) -> Callable:
    """Create joystick command overlay function for rendering."""
    xyz = np.array(state.data.xpos[env._torso_body_id])
    xyz += np.array([0, 0, 0.2])  # Offset above robot
    
    x_axis = state.data.xmat[env._torso_body_id, 0]
    yaw = -np.arctan2(x_axis[1], x_axis[0])
    
    return functools.partial(
        draw_joystick_command,
        cmd=command,
        xyz=xyz,
        theta=yaw,
        scl=scale_factor,
    )


def render_trajectory(env: Any,
                     trajectory: List[Any],
                     camera: str = "track",
                     width: int = 640,
                     height: int = 480,
                     render_every: int = 2,
                     modify_scene_fns: Optional[List[Callable]] = None,
                     scene_option: Optional[mujoco.MjvOption] = None) -> np.ndarray:
    """Render trajectory to video frames."""
    if scene_option is None:
        scene_option = setup_scene_options()
    
    # Subsample trajectory
    traj = trajectory[::render_every]
    mod_fns = modify_scene_fns[::render_every] if modify_scene_fns else None
    
    frames = env.render(
        traj,
        camera=camera,
        scene_option=scene_option,
        width=width,
        height=height,
        modify_scene_fns=mod_fns,
    )
    
    # Ensure frames is a numpy array
    if isinstance(frames, list):
        frames = np.array(frames)
    
    return frames


def save_video(frames,
              filepath: str,
              fps: float = 30.0,
              loop: bool = False):
    """Save video frames to file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to numpy array if it's a list
    if isinstance(frames, list):
        frames = np.array(frames)
    
    # Convert to proper format if needed
    if frames.dtype != np.uint8:
        frames = (frames * 255).astype(np.uint8)
    
    media.write_video(filepath, frames, fps=fps)
    print(f"Video saved to {filepath}")


def show_video(frames: np.ndarray,
              fps: float = 30.0,
              loop: bool = False):
    """Display video in notebook/interactive environment."""
    media.show_video(frames, fps=fps, loop=loop)


class VideoRenderer:
    """Helper class for consistent video rendering across evaluations."""
    
    def __init__(self,
                 camera: str = "track",
                 width: int = 640,
                 height: int = 480,
                 render_every: int = 2,
                 show_contacts: bool = True,
                 show_perturbations: bool = True):
        self.camera = camera
        self.width = width
        self.height = height
        self.render_every = render_every
        self.scene_option = setup_scene_options(show_contacts, show_perturbations)
        self.fps = None  # Will be set based on environment dt
    
    def render(self,
               env: Any,
               trajectory: List[Any],
               modify_scene_fns: Optional[List[Callable]] = None,
               save_path: Optional[str] = None) -> np.ndarray:
        """Render trajectory and optionally save to file."""
        # Calculate FPS based on environment
        if self.fps is None:
            self.fps = 1.0 / env.dt / self.render_every
        
        frames = render_trajectory(
            env=env,
            trajectory=trajectory,
            camera=self.camera,
            width=self.width,
            height=self.height,
            render_every=self.render_every,
            modify_scene_fns=modify_scene_fns,
            scene_option=self.scene_option
        )
        
        if save_path:
            save_video(frames, save_path, fps=self.fps)
        
        return frames
    
    def show(self, frames: np.ndarray, loop: bool = False):
        """Display rendered frames."""
        show_video(frames, fps=self.fps, loop=loop)


def create_command_sequence_video(env: Any,
                                 make_inference_fn: Callable,
                                 params: Any,
                                 command_sequence: List[jp.ndarray],
                                 steps_per_command: int = 200,
                                 save_path: Optional[str] = None) -> np.ndarray:
    """Create video showing robot following a sequence of commands."""
    import jax
    
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))
    
    rng = jax.random.PRNGKey(0)
    rollout = []
    modify_scene_fns = []
    
    state = jit_reset(rng)
    
    for i, command in enumerate(command_sequence):
        print(f"Command {i+1}/{len(command_sequence)}: {command}")
        
        for step in range(steps_per_command):
            state.info["command"] = command
            act_rng, rng = jax.random.split(rng)
            ctrl, _ = jit_inference_fn(state.obs, act_rng)
            state = jit_step(state, ctrl)
            rollout.append(state)
            
            # Create command overlay
            overlay_fn = create_joystick_command_overlay(
                state, env, command, 
                scale_factor=abs(command[0]) / 2.0 if len(command) > 0 else 1.0
            )
            modify_scene_fns.append(overlay_fn)
    
    # Render video
    renderer = VideoRenderer(camera="track", width=640, height=480)
    frames = renderer.render(env, rollout, modify_scene_fns, save_path)
    
    return frames