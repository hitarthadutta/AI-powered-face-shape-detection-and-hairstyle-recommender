#!/usr/bin/env python3
"""
Setup script to train the ML model and start the application
"""

import os
import sys
import subprocess

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing ML dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def train_model():
    """Train the ML model"""
    print("🎯 Training ML model...")
    try:
        # Change to backend directory
        os.chdir("backend")
        
        # Run training script
        result = subprocess.run([sys.executable, "train_model.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Model training completed successfully!")
            print(result.stdout)
            return True
        else:
            print("❌ Model training failed:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return False
    finally:
        # Return to root directory
        os.chdir("..")

def start_ml_app():
    """Start the ML-powered application"""
    print("🚀 Starting ML-powered application...")
    try:
        # Change to backend directory
        os.chdir("backend")
        
        # Start the ML app
        subprocess.run([sys.executable, "app_ml.py"])
        
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
    finally:
        # Return to root directory
        os.chdir("..")

def main():
    print("🎯 StyleAI ML Model Setup")
    print("=" * 40)
    
    # Check if dataset exists
    dataset_path = "faceshape-master/published_dataset"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        print("Please ensure the faceshape-master folder is in the current directory")
        return
    
    # Install dependencies
    if not install_dependencies():
        return
    
    # Train model
    if not train_model():
        print("⚠️ Model training failed, but you can still use the simple version")
        print("Run: python backend/app_simple.py")
        return
    
    # Ask user if they want to start the app
    response = input("\n🎉 Setup complete! Start the ML-powered app? (y/n): ").lower()
    if response in ['y', 'yes']:
        start_ml_app()
    else:
        print("👋 Setup complete! You can start the app later with:")
        print("   python backend/app_ml.py")

if __name__ == "__main__":
    main()
