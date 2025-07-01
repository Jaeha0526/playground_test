#!/usr/bin/env python3
"""Simple test of MuJoCo backends for headless video generation."""

import os
import subprocess
import sys

def test_backend(backend):
    """Test a specific MuJoCo backend."""
    print(f"Testing MUJOCO_GL={backend}...")
    
    # Set environment variables
    os.environ['MUJOCO_GL'] = backend
    
    # Set PYOPENGL_PLATFORM for EGL
    if backend == 'egl':
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        print(f"  Set PYOPENGL_PLATFORM=egl for {backend}")
    elif backend == 'osmesa':
        os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
        print(f"  Set PYOPENGL_PLATFORM=osmesa for {backend}")
    elif 'PYOPENGL_PLATFORM' in os.environ:
        del os.environ['PYOPENGL_PLATFORM']
    
    try:
        import mujoco
        
        # Basic model test
        model = mujoco.MjModel.from_xml_string('<mujoco/>')
        print(f"  ✅ Model creation successful with {backend}")
        
        # Rendering test
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=240, width=320)
        renderer.update_scene(data)
        pixels = renderer.render()
        print(f"  ✅ Rendering successful with {backend} - shape: {pixels.shape}")
        
        # Clean up
        renderer.close()
        
        return True
        
    except Exception as e:
        print(f"  ❌ Failed with {backend}: {e}")
        return False

def main():
    print("🔧 Testing MuJoCo backends individually...")
    
    # Check if GPU is available
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        gpu_available = result.returncode == 0
        if gpu_available:
            print("✅ GPU detected")
        else:
            print("❌ No GPU detected")
    except FileNotFoundError:
        print("❌ nvidia-smi not found")
        gpu_available = False
    
    # Test backends
    backends = ['osmesa']  # Start with most likely to work
    if gpu_available:
        backends.append('egl')
    
    working_backends = []
    
    for backend in backends:
        print(f"\n--- Testing {backend} ---")
        if test_backend(backend):
            working_backends.append(backend)
            print(f"✅ {backend} works!")
        else:
            print(f"❌ {backend} failed")
    
    print(f"\n📊 Results:")
    print(f"Working backends: {working_backends}")
    
    if working_backends:
        recommended = working_backends[0]
        print(f"🎯 Recommended backend: {recommended}")
        print(f"   To use: export MUJOCO_GL={recommended}")
        return recommended
    else:
        print("❌ No working backends found")
        return None

if __name__ == "__main__":
    main()