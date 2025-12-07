#!/usr/bin/env python3
"""
MetaboAI - Installation Verification Script
Checks if all required packages and dependencies are properly installed.
"""

import sys
import subprocess
from importlib import import_module


def check_python_version():
    """Check if Python version meets requirements."""
    print("Checking Python version...")
    version = sys.version_info
    
    if version.major == 3 and version.minor >= 9:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} (OK)")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (Need 3.9+)")
        return False


def check_package(package_name, import_name=None, min_version=None):
    """Check if a package is installed and optionally verify version."""
    if import_name is None:
        import_name = package_name
    
    try:
        module = import_module(import_name)
        
        # Get version if available
        version = None
        for attr in ['__version__', 'VERSION', 'version']:
            if hasattr(module, attr):
                version = getattr(module, attr)
                if isinstance(version, tuple):
                    version = '.'.join(map(str, version))
                break
        
        if version:
            status = f"✓ {package_name} ({version})"
        else:
            status = f"✓ {package_name} (installed)"
        
        print(status)
        return True
        
    except ImportError:
        print(f"✗ {package_name} (NOT INSTALLED)")
        return False


def check_gpu_support():
    """Check if GPU is available for deep learning."""
    print("\nChecking GPU support...")
    
    # Check TensorFlow GPU
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ TensorFlow GPU: {len(gpus)} GPU(s) available")
            for i, gpu in enumerate(gpus):
                print(f"  - GPU {i}: {gpu.name}")
        else:
            print("○ TensorFlow: No GPU detected (CPU only)")
    except ImportError:
        print("○ TensorFlow not installed")
    except Exception as e:
        print(f"○ TensorFlow GPU check failed: {e}")
    
    # Check PyTorch GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ PyTorch CUDA: {torch.cuda.device_count()} GPU(s) available")
            for i in range(torch.cuda.device_count()):
                print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("○ PyTorch: No CUDA GPU detected (CPU only)")
    except ImportError:
        print("○ PyTorch not installed")
    except Exception as e:
        print(f"○ PyTorch GPU check failed: {e}")


def check_system_resources():
    """Check system resources (RAM, disk space)."""
    print("\nChecking system resources...")
    
    try:
        import psutil
        
        # Check RAM
        ram = psutil.virtual_memory()
        ram_gb = ram.total / (1024**3)
        print(f"✓ RAM: {ram_gb:.1f} GB total, {ram.percent}% used")
        
        if ram_gb < 8:
            print("  ⚠ Warning: Less than 8GB RAM. May struggle with large datasets.")
        
        # Check disk space
        disk = psutil.disk_usage('/')
        disk_gb = disk.free / (1024**3)
        print(f"✓ Disk Space: {disk_gb:.1f} GB free")
        
        if disk_gb < 10:
            print("  ⚠ Warning: Less than 10GB free space. May need more for data.")
        
        # Check CPU cores
        cpu_count = psutil.cpu_count()
        print(f"✓ CPU Cores: {cpu_count}")
        
    except ImportError:
        print("○ psutil not installed (optional, skipping resource check)")


def main():
    """Main verification function."""
    
    print("=" * 70)
    print("MetaboAI - Installation Verification")
    print("=" * 70)
    print()
    
    all_good = True
    
    # Check Python version
    all_good = check_python_version() and all_good
    print()
    
    # Core packages
    print("Checking core packages...")
    core_packages = [
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('scipy', 'scipy'),
    ]
    
    for pkg_name, import_name in core_packages:
        all_good = check_package(pkg_name, import_name) and all_good
    
    print()
    
    # Mass spectrometry packages
    print("Checking mass spectrometry packages...")
    ms_packages = [
        ('pyopenms', 'pyopenms'),
        ('pymzml', 'pymzml'),
        ('matchms', 'matchms'),
    ]
    
    for pkg_name, import_name in ms_packages:
        result = check_package(pkg_name, import_name)
        if pkg_name == 'pyopenms' and not result:
            print("  ⚠ PyOpenMS is CRITICAL! Install with: conda install -c bioconda pyopenms")
        all_good = result and all_good
    
    print()
    
    # Statistical packages
    print("Checking statistical packages...")
    stats_packages = [
        ('scikit-learn', 'sklearn'),
        ('statsmodels', 'statsmodels'),
    ]
    
    for pkg_name, import_name in stats_packages:
        all_good = check_package(pkg_name, import_name) and all_good
    
    print()
    
    # Machine learning packages
    print("Checking machine learning packages...")
    ml_packages = [
        ('xgboost', 'xgboost'),
        ('lightgbm', 'lightgbm'),
    ]
    
    for pkg_name, import_name in ml_packages:
        check_package(pkg_name, import_name)  # Optional
    
    print()
    
    # Deep learning packages
    print("Checking deep learning packages...")
    dl_packages = [
        ('tensorflow', 'tensorflow'),
        ('torch', 'torch'),
    ]
    
    has_dl = False
    for pkg_name, import_name in dl_packages:
        if check_package(pkg_name, import_name):
            has_dl = True
    
    if not has_dl:
        print("  ⚠ No deep learning framework found. Install TensorFlow or PyTorch.")
    
    print()
    
    # Explainability packages
    print("Checking explainability packages...")
    explainability_packages = [
        ('shap', 'shap'),
        ('lime', 'lime'),
    ]
    
    for pkg_name, import_name in explainability_packages:
        check_package(pkg_name, import_name)  # Optional
    
    print()
    
    # Visualization packages
    print("Checking visualization packages...")
    viz_packages = [
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('plotly', 'plotly'),
    ]
    
    for pkg_name, import_name in viz_packages:
        check_package(pkg_name, import_name)
    
    print()
    
    # Web interface
    print("Checking web interface...")
    check_package('streamlit', 'streamlit')
    
    print()
    
    # Utilities
    print("Checking utility packages...")
    util_packages = [
        ('tqdm', 'tqdm'),
        ('yaml', 'yaml'),
        ('joblib', 'joblib'),
    ]
    
    for pkg_name, import_name in util_packages:
        check_package(pkg_name, import_name)
    
    # Check GPU support
    check_gpu_support()
    
    # Check system resources
    check_system_resources()
    
    print()
    print("=" * 70)
    
    if all_good:
        print("✓ All critical packages are installed!")
        print("You're ready to start using MetaboAI!")
    else:
        print("✗ Some critical packages are missing.")
        print("Please install missing packages:")
        print("  pip install -r requirements.txt")
        print("Or for PyOpenMS:")
        print("  conda install -c bioconda pyopenms")
    
    print("=" * 70)
    print()
    
    return all_good


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
