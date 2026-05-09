"""
Installation verification and environment check script.
Run this to verify all dependencies are correctly installed.
"""

import sys
import importlib


def check_module(name, package=None, min_version=None):
    """Check if a module is importable and optionally verify version."""
    try:
        mod = importlib.import_module(package or name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"  [OK] {name} (version: {version})")
        return True
    except ImportError as e:
        print(f"  [MISSING] {name}: {e}")
        return False


def check_cuda():
    """Check CUDA/GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            print(f"  [OK] CUDA available: {device_name} (CUDA {cuda_version})")
            print(f"       GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
            return True
        else:
            print(f"  [WARNING] PyTorch installed but CUDA not available")
            return False
    except ImportError:
        print(f"  [MISSING] PyTorch not installed")
        return False


def main():
    print("=" * 60)
    print("  Gait Analysis System - Environment Check")
    print("=" * 60)
    print(f"\nPython: {sys.version}")
    print(f"Platform: {sys.platform}")
    print()

    all_ok = True

    # Core dependencies
    print("Core Dependencies:")
    all_ok &= check_module("numpy")
    all_ok &= check_module("scipy")
    all_ok &= check_module("cv2", "cv2")
    all_ok &= check_module("yaml", "yaml")
    print()

    # GPU
    print("GPU Support:")
    cuda_ok = check_cuda()
    print()

    # Pose Estimation
    print("Pose Estimation:")
    mp_ok = check_module("mediapipe")
    
    mmpose_ok = False
    try:
        import mmpose
        print(f"  [OK] mmpose (version: {mmpose.__version__})")
        mmpose_ok = True
    except ImportError:
        print(f"  [INFO] mmpose not installed (optional, needed for MMPose backend)")
    
    try:
        import mmdet
        print(f"  [OK] mmdet (version: {mmdet.__version__})")
    except ImportError:
        if mmpose_ok:
            print(f"  [WARNING] mmdet not installed (required for MMPose backend)")
    print()

    # Video Readers
    print("Video Readers:")
    try:
        import pyorbbecsdk
        print(f"  [OK] pyorbbecsdk (Orbbec SDK)")
    except ImportError:
        print(f"  [INFO] pyorbbecsdk not installed (optional)")

    check_module("open3d")
    print()

    # Summary
    print("-" * 60)
    if not mp_ok and not mmpose_ok:
        print("\n[ERROR] No pose estimation backend available!")
        print("  Install at least one of:")
        print("    pip install mediapipe")
        print("    OR follow MMPose installation guide in README")
        all_ok = False
    
    if not cuda_ok:
        print("\n[WARNING] GPU acceleration not available.")
        print("  Install PyTorch with CUDA for faster processing:")
        print("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")

    if all_ok:
        print("\n[SUCCESS] All core dependencies are installed!")
        print("  You can run: python main.py --help")
    else:
        print("\n[ACTION NEEDED] Some dependencies are missing. See above.")

    print("=" * 60)


if __name__ == "__main__":
    main()
