from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import io
import base64
import os
import joblib
import json

app = Flask(__name__)
CORS(app)

# Global variables for model and face detection
face_cascade = None
ml_model = None
model_info = None

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

def load_ml_model():
    """Load the trained ML model"""
    global ml_model, model_info
    try:
        model_path = "face_shape_model.pkl"
        info_path = "model_info.json"
        
        if os.path.exists(model_path) and os.path.exists(info_path):
            ml_model = joblib.load(model_path)
            with open(info_path, 'r') as f:
                model_info = json.load(f)
            print("✅ ML model loaded successfully")
            print(f"📊 Model: {model_info['model_type']}, Accuracy: {model_info['accuracy']:.3f}")
        else:
            print("⚠️ ML model not found. Run train_model.py first to train a model.")
            ml_model = None
            model_info = None
    except Exception as e:
        print(f"Error loading ML model: {e}")
        ml_model = None
        model_info = None

def extract_features_from_image(image):
    """Extract features from face image (same as training)"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize to standard size
        gray = cv2.resize(gray, (100, 100))
        
        # Extract basic features (same as training)
        features = []
        
        # 1. Face aspect ratio (height/width)
        h, w = gray.shape
        aspect_ratio = h / w
        features.append(aspect_ratio)
        
        # 2. Face area ratio (face area / total image area)
        face_area = np.sum(gray > 50)  # Assuming face pixels are brighter
        total_area = h * w
        area_ratio = face_area / total_area
        features.append(area_ratio)
        
        # 3. Histogram features
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
        features.extend(hist.flatten())
        
        # 4. Edge features
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (h * w)
        features.append(edge_density)
        
        # 5. Geometric features
        # Top half vs bottom half brightness
        top_half = gray[:h//2, :]
        bottom_half = gray[h//2:, :]
        top_brightness = np.mean(top_half)
        bottom_brightness = np.mean(bottom_half)
        features.extend([top_brightness, bottom_brightness])
        
        # Left half vs right half brightness
        left_half = gray[:, :w//2]
        right_half = gray[:, w//2:]
        left_brightness = np.mean(left_half)
        right_brightness = np.mean(right_half)
        features.extend([left_brightness, right_brightness])
        
        return np.array(features)
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def classify_face_shape_ml(face_image):
    """Classify face shape using ML model"""
    if ml_model is None:
        return "oval", 0.5  # Fallback
    
    try:
        # Extract features
        features = extract_features_from_image(face_image)
        if features is None:
            return "oval", 0.5
        
        # Reshape for single prediction
        features = features.reshape(1, -1)
        
        # Predict
        prediction = ml_model.predict(features)[0]
        
        # Get confidence (probability)
        if hasattr(ml_model, 'predict_proba'):
            probabilities = ml_model.predict_proba(features)[0]
            confidence = np.max(probabilities)
        else:
            confidence = 0.8  # Default confidence for models without probability
        
        return prediction, confidence
        
    except Exception as e:
        print(f"Error in ML classification: {e}")
        return "oval", 0.5

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
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces using OpenCV
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return jsonify({'error': 'No face detected in the image. Please ensure the face is clearly visible and well-lit.'}), 400
        
        # Use the first detected face
        face_rect = faces[0]
        x, y, w, h = face_rect
        
        # Extract face region
        face_image = opencv_image[y:y+h, x:x+w]
        
        # Classify face shape using ML model
        if ml_model is not None:
            face_shape, confidence = classify_face_shape_ml(face_image)
            model_type = "ML Model"
        else:
            # Fallback to simple classification
            face_ratio = h / w
            if face_ratio > 1.4:
                face_shape = "oblong"
            elif face_ratio > 1.2:
                face_shape = "oval"
            elif face_ratio > 1.0:
                face_shape = "round"
            elif face_ratio > 0.8:
                face_shape = "square"
            else:
                face_shape = "heart"
            confidence = 0.6
            model_type = "Simple Classification"
        
        # Get hairstyle recommendations
        recommendations = get_hairstyle_recommendations(face_shape)
        
        return jsonify({
            'face_shape': face_shape,
            'confidence': confidence,
            'model_type': model_type,
            'recommendations': recommendations,
            'face_detected': True,
            'face_rect': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)}
        })
        
    except Exception as e:
        print(f"Error in face detection: {e}")
        return jsonify({'error': f'Face detection failed: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    status = "healthy" if face_cascade is not None else "warning"
    message = "StyleAI Backend is running"
    
    if ml_model is not None:
        message += f" (ML Model: {model_info['model_type']}, Accuracy: {model_info['accuracy']:.3f})"
    else:
        message += " (Simple Classification Mode)"
    
    return jsonify({
        'status': status, 
        'message': message,
        'face_detection_available': face_cascade is not None,
        'ml_model_available': ml_model is not None,
        'model_info': model_info
    })

# Initialize the application
if __name__ == '__main__':
    print("🚀 Starting StyleAI Backend with ML Model...")
    
    # Load face detection
    load_face_detection()
    
    # Load ML model
    load_ml_model()
    
    print(f"Face detection available: {face_cascade is not None}")
    print(f"ML model available: {ml_model is not None}")
    print("🌐 Backend will be available at: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
