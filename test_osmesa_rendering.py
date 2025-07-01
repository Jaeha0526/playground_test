#!/usr/bin/env python3
"""Test software rendering with OSMesa for headless video generation."""

import os

# Try OSMesa (software rendering) which doesn't need GPU
print("Testing OSMesa (software rendering)...")
os.environ['MUJOCO_GL'] = 'osmesa'

try:
    import mujoco
    print("✅ MuJoCo imported successfully")
    
    # Create a simple model
    model = mujoco.MjModel.from_xml_string('<mujoco/>')
    data = mujoco.MjData(model)
    print("✅ Basic model created successfully")
    
    # Try rendering
    renderer = mujoco.Renderer(model, height=240, width=320)
    renderer.update_scene(data)
    pixels = renderer.render()
    print(f"✅ Rendering successful - shape: {pixels.shape}")
    
    # Test video generation
    print("\nTesting video generation with OSMesa...")
    
    import jax
    import jax.numpy as jp
    import mediapy as media
    from mujoco_playground import registry
    
    # Load environment
    env_name = "Go1JoystickFlatTerrain"
    env = registry.load(env_name)
    print(f"✅ Environment loaded: {env_name}")
    
    # Create trajectory
    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)
    trajectory = [state]
    
    for i in range(20):  # Short trajectory
        rng, act_rng = jax.random.split(rng)
        action = jax.random.normal(act_rng, (12,)) * 0.1
        state = env.step(state, action)
        trajectory.append(state)
    
    print(f"✅ Created trajectory with {len(trajectory)} states")
    
    # Render video
    frames = env.render(
        trajectory,
        camera="track",
        width=320,
        height=240,
    )
    
    print(f"✅ Rendered {len(frames)} frames")
    
    # Save video
    video_path = "test_osmesa_video.mp4"
    media.write_video(video_path, frames, fps=30)
    
    if os.path.exists(video_path):
        file_size = os.path.getsize(video_path)
        print(f"✅ Video saved successfully: {video_path} ({file_size} bytes)")
        
        # Clean up
        os.remove(video_path)
        print("🧹 Test video cleaned up")
        
        print("\n🎉 OSMESA VIDEO GENERATION WORKING!")
        print("To use OSMesa rendering in your scripts:")
        print("export MUJOCO_GL=osmesa")
        
    else:
        print("❌ Video file was not created")

except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()