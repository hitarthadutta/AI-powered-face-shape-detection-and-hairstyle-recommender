from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import io
import base64
import os
import json

app = Flask(__name__)
CORS(app)

# Load OpenCV face detection cascade with error handling
try:
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_path):
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

def classify_face_shape(face_rect, image_shape):
    x, y, w, h = face_rect
    face_ratio = h / w
    if face_ratio > 1.4:
        return "oblong"
    elif face_ratio > 1.2:
        return "oval"
    elif face_ratio > 1.0:
        return "round"
    elif face_ratio > 0.8:
        return "square"
    else:
        return "heart"

def get_female_recs():
    return {
        "oval": {
            "description": "Oval faces are considered the most versatile and can pull off almost any hairstyle!",
            "styles": [
                {"name": "Long Layers", "description": "Soft, face-framing layers that enhance your natural features", "youtube_link": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "image_url": "https://images.unsplash.com/photo-1605497788044-5a32c7078486?w=400"},
                {"name": "Side Swept Bangs", "description": "Elegant side-swept bangs that add sophistication", "youtube_link": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "image_url": "https://images.unsplash.com/photo-1492106087820-71f1a00d2b11?w=400"},
                {"name": "Pixie Cut", "description": "Short, textured cut that highlights your facial structure", "youtube_link": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "image_url": "https://images.unsplash.com/photo-1503951914875-452162b0f3f1?w=400"}
            ]
        },
        "round": {
            "description": "Round faces benefit from styles that add length and angles to create definition.",
            "styles": [
                {"name": "Long Straight Hair", "description": "Long, straight hair creates vertical lines that elongate the face", "youtube_link": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "image_url": "https://images.unsplash.com/photo-1503951914875-452162b0f3f1?w=400"},
                {"name": "Asymmetrical Bob", "description": "Angled cuts add definition and structure to round faces", "youtube_link": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "image_url": "https://images.unsplash.com/photo-1492106087820-71f1a00d2b11?w=400"},
                {"name": "Side Part with Volume", "description": "Side parts and volume on top create the illusion of length", "youtube_link": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "image_url": "https://images.unsplash.com/photo-1605497788044-5a32c7078486?w=400"}
            ]
        },
        "square": {
            "description": "Square faces have strong angles - soften them with rounded, layered styles.",
            "styles": [
                {"name": "Soft Waves", "description": "Soft, beachy waves soften angular features", "youtube_link": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "image_url": "https://images.unsplash.com/photo-1503951914875-452162b0f3f1?w=400"},
                {"name": "Layered Cut", "description": "Multiple layers create movement and soften sharp angles", "youtube_link": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "image_url": "https://images.unsplash.com/photo-1492106087820-71f1a00d2b11?w=400"},
                {"name": "Side Swept Style", "description": "Asymmetrical styles break up the square shape", "youtube_link": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "image_url": "https://images.unsplash.com/photo-1605497788044-5a32c7078486?w=400"}
            ]
        },
        "heart": {
            "description": "Heart-shaped faces are wider at the top - balance with styles that add width at the bottom.",
            "styles": [
                {"name": "Chin-Length Bob", "description": "Chin-length cuts add width to balance the narrow chin", "youtube_link": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "image_url": "https://images.unsplash.com/photo-1503951914875-452162b0f3f1?w=400"},
                {"name": "Side Swept Bangs", "description": "Soft bangs that don't overwhelm the forehead", "youtube_link": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "image_url": "https://images.unsplash.com/photo-1492106087820-71f1a00d2b11?w=400"},
                {"name": "Layered Cut with Volume", "description": "Layers and volume at the bottom balance the face shape", "youtube_link": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "image_url": "https://images.unsplash.com/photo-1605497788044-5a32c7078486?w=400"}
            ]
        },
        "oblong": {
            "description": "Oblong faces are long and narrow - add width and break up the length.",
            "styles": [
                {"name": "Medium Length with Layers", "description": "Medium length with layers adds width and movement", "youtube_link": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "image_url": "https://images.unsplash.com/photo-1503951914875-452162b0f3f1?w=400"},
                {"name": "Side Swept Bangs", "description": "Bangs break up the length and add interest", "youtube_link": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "image_url": "https://images.unsplash.com/photo-1492106087820-71f1a00d2b11?w=400"},
                {"name": "Bob with Volume", "description": "Bob cuts with volume add width to narrow faces", "youtube_link": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "image_url": "https://images.unsplash.com/photo-1605497788044-5a32c7078486?w=400"}
            ]
        }
    }

def get_male_recs():
    return {
        "oval": {
            "description": "Oval faces can pull off most men's hairstyles.",
            "styles": [
                {"name": "Textured Quiff", "description": "Classic quiff with texture for volume.", "youtube_link": "https://www.youtube.com/watch?v=smULlWVSj6A", "image_url": "https://images.unsplash.com/photo-1520975922215-230f77a5a89c?w=400"},
                {"name": "Side Part", "description": "Clean side part for a sharp look.", "youtube_link": "https://www.youtube.com/watch?v=GQz9bbOQG30", "image_url": "https://images.unsplash.com/photo-1520975954732-35dd22ab9f6b?w=400"},
                {"name": "Short Pompadour", "description": "Modern pompadour with neat sides.", "youtube_link": "https://www.youtube.com/watch?v=7s5n4v1xS6Q", "image_url": "https://images.unsplash.com/photo-1520975922215-230f77a5a89c?w=400"}
            ]
        },
        "round": {
            "description": "Round faces benefit from height and structure.",
            "styles": [
                {"name": "High Fade + Pompadour", "description": "Add height to elongate the face.", "youtube_link": "https://www.youtube.com/watch?v=3gPG0dG4qTI", "image_url": "https://images.unsplash.com/photo-1516646255117-8c7f98d53d44?w=400"},
                {"name": "Angular Fringe", "description": "Angles add definition to soft features.", "youtube_link": "https://www.youtube.com/watch?v=JBn3iX3g5M8", "image_url": "https://images.unsplash.com/photo-1516646255117-8c7f98d53d44?w=400"},
                {"name": "Side Part + Volume", "description": "Volume on top creates length illusion.", "youtube_link": "https://www.youtube.com/watch?v=GQz9bbOQG30", "image_url": "https://images.unsplash.com/photo-1516646255117-8c7f98d53d44?w=400"}
            ]
        },
        "square": {
            "description": "Square faces suit clean, structured cuts.",
            "styles": [
                {"name": "Crew Cut", "description": "Short and sharp to highlight jawline.", "youtube_link": "https://www.youtube.com/watch?v=7QkqQ0bT0yI", "image_url": "https://images.unsplash.com/photo-1513836279014-a89f7a76ae86?w=400"},
                {"name": "Side Part", "description": "Classic and masculine.", "youtube_link": "https://www.youtube.com/watch?v=GQz9bbOQG30", "image_url": "https://images.unsplash.com/photo-1513836279014-a89f7a76ae86?w=400"},
                {"name": "Textured Crop", "description": "Texture softens strong angles.", "youtube_link": "https://www.youtube.com/watch?v=NeDtnxE7Jpc", "image_url": "https://images.unsplash.com/photo-1513836279014-a89f7a76ae86?w=400"}
            ]
        },
        "heart": {
            "description": "Heart faces need width at the jaw to balance.",
            "styles": [
                {"name": "Medium Length Sweep", "description": "Soft volume, avoid heavy top.", "youtube_link": "https://www.youtube.com/watch?v=bXoF6p-2vYg", "image_url": "https://images.unsplash.com/photo-1520975922215-230f77a5a89c?w=400"},
                {"name": "Side Fringe", "description": "Breaks up forehead width.", "youtube_link": "https://www.youtube.com/watch?v=JBn3iX3g5M8", "image_url": "https://images.unsplash.com/photo-1520975922215-230f77a5a89c?w=400"},
                {"name": "Low Fade + Texture", "description": "Adds width near the jaw.", "youtube_link": "https://www.youtube.com/watch?v=NeDtnxE7Jpc", "image_url": "https://images.unsplash.com/photo-1513836279014-a89f7a76ae86?w=400"}
            ]
        },
        "oblong": {
            "description": "Oblong faces benefit from less height and more width.",
            "styles": [
                {"name": "Classic Crop", "description": "Keep top moderate, add width.", "youtube_link": "https://www.youtube.com/watch?v=NeDtnxE7Jpc", "image_url": "https://images.unsplash.com/photo-1513836279014-a89f7a76ae86?w=400"},
                {"name": "Side Part Medium", "description": "Balance length with side volume.", "youtube_link": "https://www.youtube.com/watch?v=GQz9bbOQG30", "image_url": "https://images.unsplash.com/photo-1513836279014-a89f7a76ae86?w=400"},
                {"name": "Textured Fringe", "description": "Reduce vertical length visually.", "youtube_link": "https://www.youtube.com/watch?v=JBn3iX3g5M8", "image_url": "https://images.unsplash.com/photo-1513836279014-a89f7a76ae86?w=400"}
            ]
        }
    }

def get_hairstyle_recommendations(face_shape, gender="female"):
    female = get_female_recs()
    male = get_male_recs()
    source = female if gender == "female" else male
    return source.get(face_shape, source["oval"])

@app.route('/api/detect-face-shape', methods=['POST'])
def detect_face_shape():
    try:
        if face_cascade is None:
            return jsonify({'error': 'Face detection not available. Please check backend logs.'}), 500
        data = request.get_json()
        image_data = data.get('image')
        gender = (data.get('gender') or 'female').lower()
        if gender not in ['male', 'female']:
            gender = 'female'
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes))
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            return jsonify({'error': 'No face detected in the image. Please ensure the face is clearly visible and well-lit.'}), 400
        face_rect = faces[0]
        face_shape = classify_face_shape(face_rect, opencv_image.shape)
        recommendations = get_hairstyle_recommendations(face_shape, gender)
        return jsonify({
            'face_shape': face_shape,
            'gender': gender,
            'confidence': 0.75,
            'recommendations': recommendations
        })
    except Exception as e:
        print(f"Error in face detection: {e}")
        return jsonify({'error': f'Face detection failed: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    status = "healthy" if face_cascade is not None else "warning"
    message = "Face Shape Detection API is running (Gender-aware)"
    return jsonify({'status': status, 'message': message, 'face_detection_available': face_cascade is not None})

if __name__ == '__main__':
    print("🚀 Starting StyleAI Backend (Gender-aware)...")
    app.run(debug=True, host='0.0.0.0', port=5000)
