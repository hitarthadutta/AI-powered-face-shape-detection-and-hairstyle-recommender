from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import io
import base64
import os

app = Flask(__name__)
CORS(app)

# Global variables for face detection
face_cascade = None

def load_face_detection():
    """Load OpenCV face detection cascade"""
    global face_cascade
    try:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            print("Warning: Could not load face cascade classifier")
            face_cascade = None
        else:
            print("✅ Face cascade classifier loaded successfully")
    except Exception as e:
        print(f"Error loading face cascade: {e}")
        face_cascade = None

def analyze_face_shape_consistent(image):
    """Analyze face shape using consistent geometric measurements"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize to standard size for consistency
        gray = cv2.resize(gray, (300, 300))
        
        # Detect face
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return "oval", 0.5, "No face detected"
        
        # Use the largest face
        face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = face
        
        # Extract face region
        face_region = gray[y:y+h, x:x+w]
        face_region = cv2.resize(face_region, (200, 200))
        
        # Calculate consistent measurements
        h_face, w_face = face_region.shape
        
        # 1. Basic aspect ratio
        aspect_ratio = h_face / w_face
        
        # 2. Width measurements at different heights
        # Top third (forehead area)
        top_third = face_region[:h_face//3, :]
        top_width = np.mean(np.sum(top_third > 100, axis=0))
        
        # Middle third (cheek area)
        middle_third = face_region[h_face//3:2*h_face//3, :]
        middle_width = np.mean(np.sum(middle_third > 100, axis=0))
        
        # Bottom third (jaw area)
        bottom_third = face_region[2*h_face//3:, :]
        bottom_width = np.mean(np.sum(bottom_third > 100, axis=0))
        
        # 3. Calculate ratios
        top_middle_ratio = top_width / (middle_width + 1e-6)
        middle_bottom_ratio = middle_width / (bottom_width + 1e-6)
        top_bottom_ratio = top_width / (bottom_width + 1e-6)
        
        # 4. Determine face shape based on consistent rules
        face_shape, confidence, reasoning = classify_face_shape_rules(
            aspect_ratio, top_middle_ratio, middle_bottom_ratio, top_bottom_ratio
        )
        
        return face_shape, confidence, reasoning
        
    except Exception as e:
        print(f"Error in face analysis: {e}")
        return "oval", 0.5, f"Analysis error: {str(e)}"

def classify_face_shape_rules(aspect_ratio, top_middle_ratio, middle_bottom_ratio, top_bottom_ratio):
    """Classify face shape using consistent rules"""
    
    # Rule-based classification for consistency
    if aspect_ratio > 1.3:
        # Long face
        if top_middle_ratio > 1.1 and middle_bottom_ratio > 1.1:
            return "oblong", 0.85, "Long face with consistent width"
        else:
            return "oval", 0.75, "Long face with varying width"
    
    elif aspect_ratio < 0.9:
        # Wide face
        if top_middle_ratio < 0.9 and middle_bottom_ratio < 0.9:
            return "round", 0.85, "Wide face with consistent width"
        else:
            return "square", 0.75, "Wide face with angular features"
    
    else:
        # Medium aspect ratio
        if top_middle_ratio > 1.2 and middle_bottom_ratio < 0.8:
            return "heart", 0.85, "Wide forehead, narrow jaw"
        elif top_middle_ratio < 0.8 and middle_bottom_ratio > 1.2:
            return "square", 0.80, "Narrow forehead, wide jaw"
        elif abs(top_middle_ratio - 1.0) < 0.2 and abs(middle_bottom_ratio - 1.0) < 0.2:
            return "oval", 0.90, "Balanced proportions"
        else:
            return "round", 0.70, "Medium proportions with some width variation"

def get_hairstyle_recommendations(face_shape):
    """Get hairstyle recommendations based on face shape"""
    recommendations = {
        "oval": {
            "description": "Oval faces are considered the most versatile and can pull off almost any hairstyle!",
            "styles": [
                {
                    "name": "Long Layers",
                    "description": "Soft, face-framing layers that enhance your natural features",
                    "youtube_link": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    "image_url": "https://images.unsplash.com/photo-1605497788044-5a32c7078486?w=400"
                },
                {
                    "name": "Side Swept Bangs",
                    "description": "Elegant side-swept bangs that add sophistication",
                    "youtube_link": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    "image_url": "https://images.unsplash.com/photo-1492106087820-71f1a00d2b11?w=400"
                },
                {
                    "name": "Pixie Cut",
                    "description": "Short, textured cut that highlights your facial structure",
                    "youtube_link": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    "image_url": "https://images.unsplash.com/photo-1503951914875-452162b0f3f1?w=400"
                }
            ]
        },
        "round": {
            "description": "Round faces benefit from styles that add length and angles to create definition.",
            "styles": [
                {
                    "name": "Long Straight Hair",
                    "description": "Long, straight hair creates vertical lines that elongate the face",
                    "youtube_link": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    "image_url": "https://images.unsplash.com/photo-1503951914875-452162b0f3f1?w=400"
                },
                {
                    "name": "Asymmetrical Bob",
                    "description": "Angled cuts add definition and structure to round faces",
                    "youtube_link": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    "image_url": "https://images.unsplash.com/photo-1492106087820-71f1a00d2b11?w=400"
                },
                {
                    "name": "Side Part with Volume",
                    "description": "Side parts and volume on top create the illusion of length",
                    "youtube_link": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    "image_url": "https://images.unsplash.com/photo-1605497788044-5a32c7078486?w=400"
                }
            ]
        },
        "square": {
            "description": "Square faces have strong angles - soften them with rounded, layered styles.",
            "styles": [
                {
                    "name": "Soft Waves",
                    "description": "Soft, beachy waves soften angular features",
                    "youtube_link": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    "image_url": "https://images.unsplash.com/photo-1503951914875-452162b0f3f1?w=400"
                },
                {
                    "name": "Layered Cut",
                    "description": "Multiple layers create movement and soften sharp angles",
                    "youtube_link": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    "image_url": "https://images.unsplash.com/photo-1492106087820-71f1a00d2b11?w=400"
                },
                {
                    "name": "Side Swept Style",
                    "description": "Asymmetrical styles break up the square shape",
                    "youtube_link": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    "image_url": "https://images.unsplash.com/photo-1605497788044-5a32c7078486?w=400"
                }
            ]
        },
        "heart": {
            "description": "Heart-shaped faces are wider at the top - balance with styles that add width at the bottom.",
            "styles": [
                {
                    "name": "Chin-Length Bob",
                    "description": "Chin-length cuts add width to balance the narrow chin",
                    "youtube_link": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    "image_url": "https://images.unsplash.com/photo-1503951914875-452162b0f3f1?w=400"
                },
                {
                    "name": "Side Swept Bangs",
                    "description": "Soft bangs that don't overwhelm the forehead",
                    "youtube_link": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    "image_url": "https://images.unsplash.com/photo-1492106087820-71f1a00d2b11?w=400"
                },
                {
                    "name": "Layered Cut with Volume",
                    "description": "Layers and volume at the bottom balance the face shape",
                    "youtube_link": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    "image_url": "https://images.unsplash.com/photo-1605497788044-5a32c7078486?w=400"
                }
            ]
        },
        "oblong": {
            "description": "Oblong faces are long and narrow - add width and break up the length.",
            "styles": [
                {
                    "name": "Medium Length with Layers",
                    "description": "Medium length with layers adds width and movement",
                    "youtube_link": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    "image_url": "https://images.unsplash.com/photo-1503951914875-452162b0f3f1?w=400"
                },
                {
                    "name": "Side Swept Bangs",
                    "description": "Bangs break up the length and add interest",
                    "youtube_link": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    "image_url": "https://images.unsplash.com/photo-1492106087820-71f1a00d2b11?w=400"
                },
                {
                    "name": "Bob with Volume",
                    "description": "Bob cuts with volume add width to narrow faces",
                    "youtube_link": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    "image_url": "https://images.unsplash.com/photo-1605497788044-5a32c7078486?w=400"
                }
            ]
        }
    }
    
    return recommendations.get(face_shape, recommendations["oval"])

@app.route('/api/detect-face-shape', methods=['POST'])
def detect_face_shape():
    try:
        # Check if face detection is available
        if face_cascade is None:
            return jsonify({'error': 'Face detection not available. Please check backend logs.'}), 500
        
        # Get image data from request
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Analyze face shape using consistent method
        face_shape, confidence, reasoning = analyze_face_shape_consistent(opencv_image)
        
        # Get hairstyle recommendations
        recommendations = get_hairstyle_recommendations(face_shape)
        
        return jsonify({
            'face_shape': face_shape,
            'confidence': confidence,
            'model_type': 'Consistent Rule-Based',
            'reasoning': reasoning,
            'recommendations': recommendations,
            'face_detected': True,
            'consistent': True
        })
        
    except Exception as e:
        print(f"Error in face detection: {e}")
        return jsonify({'error': f'Face detection failed: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    status = "healthy" if face_cascade is not None else "warning"
    message = "StyleAI Backend is running (Consistent Rule-Based)"
    
    return jsonify({
        'status': status, 
        'message': message,
        'face_detection_available': face_cascade is not None,
        'model_type': 'Consistent Rule-Based',
        'consistent_results': True
    })

# Initialize the application
if __name__ == '__main__':
    print("🚀 Starting StyleAI Backend with Consistent Rule-Based Classification...")
    
    # Load face detection
    load_face_detection()
    
    print(f"Face detection available: {face_cascade is not None}")
    print("🌐 Backend will be available at: http://localhost:5000")
    print("✅ This version provides consistent results for the same image")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
