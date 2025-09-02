#!/usr/bin/env python3
"""
Simple test script to verify the backend API is working
"""

import requests
import json
import base64
from PIL import Image
import numpy as np

def test_health_endpoint():
    """Test the health check endpoint"""
    try:
        response = requests.get('http://localhost:5000/api/health')
        if response.status_code == 200:
            print("✅ Health check passed!")
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"❌ Health check failed with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to backend. Make sure it's running on http://localhost:5000")
        return False

def test_face_detection_endpoint():
    """Test the face detection endpoint with a sample image"""
    try:
        # Create a simple test image (1x1 pixel)
        test_image = Image.new('RGB', (100, 100), color='red')
        
        # Convert to base64
        import io
        buffer = io.BytesIO()
        test_image.save(buffer, format='JPEG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        # Test the endpoint
        response = requests.post('http://localhost:5000/api/detect-face-shape', 
                               json={'image': f'data:image/jpeg;base64,{img_str}'})
        
        if response.status_code == 400:
            print("✅ Face detection endpoint is working (expected error for invalid image)")
            print(f"Response: {response.json()}")
            return True
        elif response.status_code == 200:
            print("✅ Face detection endpoint is working!")
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"❌ Face detection endpoint failed with status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to backend. Make sure it's running on http://localhost:5000")
        return False
    except Exception as e:
        print(f"❌ Error testing face detection: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing StyleAI Backend API")
    print("=" * 40)
    
    # Test health endpoint
    health_ok = test_health_endpoint()
    print()
    
    # Test face detection endpoint
    detection_ok = test_face_detection_endpoint()
    print()
    
    # Summary
    if health_ok and detection_ok:
        print("🎉 All tests passed! Backend is working correctly.")
    else:
        print("❌ Some tests failed. Check the backend logs for more details.")
    
    print("\nTo test with a real image:")
    print("1. Open http://localhost:3000 in your browser")
    print("2. Upload a photo or use the webcam")
    print("3. Check the results!")

if __name__ == "__main__":
    main()
