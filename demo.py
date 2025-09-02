#!/usr/bin/env python3
"""
Demo script for StyleAI - Test face detection with dataset images
"""

import os
import base64
import requests
import time
from PIL import Image
import io

def encode_image_to_base64(image_path):
    """Convert image to base64 string"""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:image/jpeg;base64,{encoded_string}"
    except Exception as e:
        print(f"❌ Error encoding image {image_path}: {e}")
        return None

def test_face_detection(image_path):
    """Test face detection with a specific image"""
    print(f"🔍 Testing with image: {os.path.basename(image_path)}")
    
    # Encode image
    image_data = encode_image_to_base64(image_path)
    if not image_data:
        return False
    
    # Send request to backend
    try:
        response = requests.post(
            'http://localhost:5000/api/detect-face-shape',
            json={'image': image_data},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Face detected successfully!")
            print(f"   Face shape: {result['face_shape']}")
            print(f"   Confidence: {result['confidence']:.2f}")
            print(f"   Description: {result['recommendations']['description']}")
            print(f"   Hairstyles recommended: {len(result['recommendations']['styles'])}")
            return True
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to backend. Make sure it's running on http://localhost:5000")
        return False
    except Exception as e:
        print(f"❌ Error during request: {e}")
        return False

def find_test_images():
    """Find test images from the dataset"""
    dataset_path = "faceshape-master/published_dataset"
    test_images = []
    
    if not os.path.exists(dataset_path):
        print("❌ Dataset directory not found")
        return test_images
    
    # Look for images in each face shape folder
    face_shapes = ['oval', 'round', 'square', 'heart', 'oblong']
    
    for shape in face_shapes:
        shape_path = os.path.join(dataset_path, shape)
        if os.path.exists(shape_path):
            # Get first few images from each category
            images = [f for f in os.listdir(shape_path) if f.endswith('.jpg')][:2]
            for img in images:
                test_images.append((os.path.join(shape_path, img), shape))
    
    return test_images

def main():
    """Main demo function"""
    print("🎭 StyleAI - Face Detection Demo")
    print("=" * 40)
    
    # Check if backend is running
    print("🔍 Checking if backend is running...")
    try:
        response = requests.get('http://localhost:5000/api/health', timeout=5)
        if response.status_code == 200:
            print("✅ Backend is running!")
        else:
            print("❌ Backend responded with error")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Backend is not running. Please start it first:")
        print("   cd backend && python app.py")
        return
    
    # Find test images
    print("\n🔍 Finding test images from dataset...")
    test_images = find_test_images()
    
    if not test_images:
        print("❌ No test images found")
        return
    
    print(f"✅ Found {len(test_images)} test images")
    
    # Test each image
    print("\n🧪 Testing face detection...")
    successful_tests = 0
    
    for image_path, expected_shape in test_images:
        print(f"\n--- Testing {expected_shape} face ---")
        if test_face_detection(image_path):
            successful_tests += 1
        time.sleep(1)  # Small delay between requests
    
    # Summary
    print(f"\n📊 Demo Results:")
    print(f"   Total tests: {len(test_images)}")
    print(f"   Successful: {successful_tests}")
    print(f"   Success rate: {(successful_tests/len(test_images)*100):.1f}%")
    
    if successful_tests > 0:
        print("\n🎉 Demo completed successfully!")
        print("🌐 Open http://localhost:3000 in your browser to try the web interface")
    else:
        print("\n❌ Demo failed. Check the backend logs for more details.")

if __name__ == "__main__":
    main()
