#!/usr/bin/env python3
"""
Setup script for StyleAI - Hairstyle Recommendation System
This script helps set up the development environment
"""

import os
import subprocess
import sys
import platform

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required. Current version:", sys.version)
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def check_node_installed():
    """Check if Node.js is installed"""
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Node.js version: {result.stdout.strip()}")
            return True
        else:
            print("❌ Node.js is not installed or not accessible")
            return False
    except FileNotFoundError:
        print("❌ Node.js is not installed. Please install Node.js 16+ from https://nodejs.org/")
        return False

def check_npm_installed():
    """Check if npm is installed"""
    try:
        result = subprocess.run(['npm', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ npm version: {result.stdout.strip()}")
            return True
        else:
            print("❌ npm is not accessible")
            return False
    except FileNotFoundError:
        print("❌ npm is not installed")
        return False

def create_virtual_environment():
    """Create Python virtual environment"""
    if not os.path.exists('backend/venv'):
        print("📦 Creating Python virtual environment...")
        try:
            subprocess.run([sys.executable, '-m', 'venv', 'backend/venv'], check=True)
            print("✅ Virtual environment created successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to create virtual environment")
            return False
    else:
        print("✅ Virtual environment already exists")
    return True

def install_backend_dependencies():
    """Install Python backend dependencies"""
    print("📦 Installing backend dependencies...")
    
    # Determine activation script based on OS
    if platform.system() == "Windows":
        activate_script = "backend\\venv\\Scripts\\activate.bat"
        pip_path = "backend\\venv\\Scripts\\pip"
    else:
        activate_script = "backend/venv/bin/activate"
        pip_path = "backend/venv/bin/pip"
    
    try:
        # Install requirements
        subprocess.run([pip_path, 'install', '-r', 'requirements.txt'], check=True)
        print("✅ Backend dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install backend dependencies")
        return False

def install_frontend_dependencies():
    """Install Node.js frontend dependencies"""
    print("📦 Installing frontend dependencies...")
    try:
        subprocess.run(['npm', 'install'], cwd='frontend', check=True)
        print("✅ Frontend dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install frontend dependencies")
        return False

def main():
    """Main setup function"""
    print("🚀 StyleAI - Development Environment Setup")
    print("=" * 50)
    
    # Check prerequisites
    print("\n🔍 Checking prerequisites...")
    if not check_python_version():
        return False
    
    if not check_node_installed():
        return False
    
    if not check_npm_installed():
        return False
    
    # Create virtual environment
    print("\n🐍 Setting up Python environment...")
    if not create_virtual_environment():
        return False
    
    # Install dependencies
    print("\n📚 Installing dependencies...")
    if not install_backend_dependencies():
        return False
    
    if not install_frontend_dependencies():
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Start the backend: cd backend && python app.py")
    print("2. Start the frontend: cd frontend && npm start")
    print("3. Or use the provided startup scripts:")
    print("   - Windows: run start.bat")
    print("   - PowerShell: run start.ps1")
    print("\n🌐 The application will be available at:")
    print("   - Backend: http://localhost:5000")
    print("   - Frontend: http://localhost:3000")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)
