#!/usr/bin/env python3
"""Test script to understand headless video generation setup requirements.

This script replicates the exact environment setup from the Colab notebook
to understand what's needed for video generation in headless SSH environments.
"""

import os
import subprocess
import mujoco
import numpy as np
from pathlib import Path

def setup_headless_video_environment():
    """Setup environment for headless video generation following Colab notebook pattern."""
    
    print("🔧 Setting up headless video environment...")
    
    # 1. Check GPU availability (following notebook pattern)
    print("1. Checking GPU availability...")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode != 0:
            print("   ❌ No GPU detected or nvidia-smi failed")
            print(f"   Error: {result.stderr}")
            gpu_available = False
        else:
            print("   ✅ GPU detected")
            print(f"   {result.stdout.split('Driver Version:')[1].split('|')[0].strip()}")
            gpu_available = True
    except FileNotFoundError:
        print("   ❌ nvidia-smi not found")
        gpu_available = False
    
    # Test different MuJoCo GL backends
    print("2. Testing MuJoCo GL backends...")
    
    backends_to_test = ['egl', 'osmesa', 'glfw'] if gpu_available else ['osmesa', 'glfw']
    
    for backend in backends_to_test:
        print(f"   Testing MUJOCO_GL={backend}...")
        os.environ['MUJOCO_GL'] = backend
        
        try:
            # Basic MuJoCo test
            model = mujoco.MjModel.from_xml_string('<mujoco/>')
            print(f"      ✅ Basic MuJoCo model creation successful with {backend}")
            
            # Test with actual rendering
            data = mujoco.MjData(model)
            renderer = mujoco.Renderer(model, height=240, width=320)
            renderer.update_scene(data)
            pixels = renderer.render()
            print(f"      ✅ MuJoCo rendering successful with {backend} - got {pixels.shape} pixels")
            
            return True, backend
            
        except Exception as e:
            print(f"      ❌ MuJoCo rendering test failed with {backend}: {e}")
            print(f"      Error type: {type(e).__name__}")
            continue
    
    print("   ❌ All MuJoCo GL backends failed")
    return False, None

def setup_egl_config_if_needed(backend):
    """Setup EGL configuration if using EGL backend."""
    if backend != 'egl':
        return True
        
    print("3. Setting up Nvidia EGL ICD configuration...")
    
    nvidia_icd_config_path = '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
    
    if not os.path.exists(nvidia_icd_config_path):
        print(f"   Creating {nvidia_icd_config_path}...")
        try:
            # Create directory if it doesn't exist
            Path(nvidia_icd_config_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Write the ICD configuration
            with open(nvidia_icd_config_path, 'w') as f:
                f.write("""{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
""")
            print("   ✅ Nvidia EGL ICD configuration created")
            return True
        except PermissionError:
            print("   ❌ Permission denied - need sudo access")
            print("   Try running: sudo mkdir -p /usr/share/glvnd/egl_vendor.d")
            print("   Then: sudo tee /usr/share/glvnd/egl_vendor.d/10_nvidia.json > /dev/null <<'EOF'")
            print("   {")
            print('       "file_format_version" : "1.0.0",')
            print('       "ICD" : {')
            print('           "library_path" : "libEGL_nvidia.so.0"')
            print('       }')
            print("   }")
            print("   EOF")
            return False
    else:
        print("   ✅ Nvidia EGL ICD configuration already exists")
        return True

def test_video_generation():
    """Test video generation using MuJoCo Playground pattern."""
    
    print("\n🎬 Testing video generation...")
    
    try:
        # Import required modules
        from mujoco_playground import registry
        import mediapy as media
        
        # Load a simple environment
        env_name = "Go1JoystickFlatTerrain"
        env = registry.load(env_name)
        
        print(f"   Environment loaded: {env_name}")
        
        # Simple rollout
        import jax
        rng = jax.random.PRNGKey(0)
        state = env.reset(rng)
        
        trajectory = [state]
        for i in range(10):  # Short trajectory
            rng, act_rng = jax.random.split(rng)
            action = jax.random.normal(act_rng, (12,)) * 0.1  # Small random actions
            state = env.step(state, action)
            trajectory.append(state)
        
        print(f"   Created trajectory with {len(trajectory)} states")
        
        # Test rendering
        scene_option = mujoco.MjvOption()
        scene_option.geomgroup[2] = True
        scene_option.geomgroup[3] = False
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
        
        frames = env.render(
            trajectory,
            camera="track",
            scene_option=scene_option,
            width=320,
            height=240,
        )
        
        print(f"   ✅ Rendered {len(frames)} frames, shape: {frames.shape}")
        
        # Test video saving
        test_video_path = "test_video.mp4"
        media.write_video(test_video_path, frames, fps=30)
        
        if os.path.exists(test_video_path):
            file_size = os.path.getsize(test_video_path)
            print(f"   ✅ Video saved successfully: {test_video_path} ({file_size} bytes)")
            
            # Clean up
            os.remove(test_video_path)
            print("   🧹 Test video cleaned up")
            
            return True
        else:
            print("   ❌ Video file was not created")
            return False
            
    except Exception as e:
        print(f"   ❌ Video generation test failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

def diagnose_environment():
    """Diagnose current environment setup."""
    
    print("\n🔍 Environment Diagnosis:")
    print("=" * 50)
    
    # Check environment variables
    print("Environment variables:")
    for var in ['MUJOCO_GL', 'DISPLAY', 'XAUTHORITY', 'XLA_FLAGS']:
        value = os.environ.get(var, 'NOT SET')
        print(f"   {var}: {value}")
    
    # Check EGL libraries
    print("\nEGL Libraries:")
    egl_libs = [
        '/usr/lib/x86_64-linux-gnu/libEGL.so',
        '/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0',
        '/usr/lib/x86_64-linux-gnu/libGL.so',
    ]
    
    for lib in egl_libs:
        exists = "✅" if os.path.exists(lib) else "❌"
        print(f"   {exists} {lib}")
    
    # Check ICD configuration
    print("\nICD Configuration:")
    icd_path = '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
    if os.path.exists(icd_path):
        print(f"   ✅ {icd_path}")
        try:
            with open(icd_path, 'r') as f:
                content = f.read()
            print(f"   Content: {content[:100]}...")
        except Exception as e:
            print(f"   ❌ Could not read: {e}")
    else:
        print(f"   ❌ {icd_path} not found")
    
    # Check Python packages
    print("\nPython packages:")
    required_packages = ['mujoco', 'mediapy', 'jax', 'brax', 'mujoco_playground']
    for pkg in required_packages:
        try:
            __import__(pkg)
            print(f"   ✅ {pkg}")
        except ImportError:
            print(f"   ❌ {pkg}")

if __name__ == "__main__":
    print("🚀 MuJoCo Headless Video Generation Setup Test")
    print("=" * 60)
    
    # Diagnose current state
    diagnose_environment()
    
    # Setup environment
    setup_result = setup_headless_video_environment()
    
    if isinstance(setup_result, tuple):
        setup_success, working_backend = setup_result
    else:
        setup_success = setup_result
        working_backend = None
    
    if setup_success:
        print(f"\n✅ Environment setup completed successfully with {working_backend} backend!")
        
        # Setup EGL config if needed
        if working_backend == 'egl':
            egl_setup = setup_egl_config_if_needed(working_backend)
            if not egl_setup:
                print("⚠️  EGL config setup failed, but continuing with working backend")
        
        # Test video generation
        video_success = test_video_generation()
        
        if video_success:
            print(f"\n🎉 HEADLESS VIDEO GENERATION WORKING with {working_backend} backend!")
            print("Your environment is properly configured for video generation.")
            print(f"\nTo use this in your scripts, set: export MUJOCO_GL={working_backend}")
        else:
            print("\n❌ Video generation test failed")
            print("Check the error messages above for troubleshooting.")
    else:
        print("\n❌ Environment setup failed")
        print("Please address the issues above before proceeding.")