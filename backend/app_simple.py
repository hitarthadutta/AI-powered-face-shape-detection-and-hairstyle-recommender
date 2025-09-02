from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
from PIL import Image
import random

app = Flask(__name__)
CORS(app)

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
        # Get image data from request
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # For demo purposes, randomly select a face shape
        # In production, this would use actual face detection
        face_shapes = ['oval', 'round', 'square', 'heart', 'oblong']
        face_shape = random.choice(face_shapes)
        
        # Get hairstyle recommendations
        recommendations = get_hairstyle_recommendations(face_shape)
        
        return jsonify({
            'face_shape': face_shape,
            "confidence": 0.85,
            'recommendations': recommendations,
            'note': 'This is a demo version. Face detection will be implemented with OpenCV/MediaPipe.'
        })
        
    except Exception as e:
        return jsonify({'error': f'Face detection failed: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy', 
        'message': 'StyleAI Backend is running (Demo Mode)',
        'face_detection_available': True,
        'mode': 'demo'
    })

if __name__ == '__main__':
    print("🚀 Starting StyleAI Backend (Demo Mode)...")
    print("✅ This version will work without OpenCV issues")
    print("🌐 Backend will be available at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
