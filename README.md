# StyleAI - AI-Powered Face Shape Detection & Hairstyle Recommendations

A full-stack web application that uses AI to detect a user's face shape via webcam or image upload, and then recommends the best hairstyles with direct YouTube tutorial links suited for their unique facial structure.

## ✨ Features

- **Real-time Webcam Capture**: Take photos directly from your device's camera
- **Image Upload**: Upload existing photos from your device
- **AI Face Shape Detection**: Uses MediaPipe for accurate facial landmark detection
- **Smart Hairstyle Recommendations**: Curated hairstyles mapped to each face shape
- **YouTube Tutorial Links**: Direct links to video tutorials for each recommended style
- **Beautiful UI**: Modern, responsive design with smooth animations
- **Mobile Responsive**: Works perfectly on all device sizes

## 🎯 Face Shapes Supported

- **Oval** 🥚 - Most versatile face shape
- **Round** ⭕ - Benefits from length and angles
- **Square** ⬜ - Strong angles, softened by rounded styles
- **Heart** ❤️ - Wider at top, balanced with bottom width
- **Oblong** 📏 - Long and narrow, needs width and breaks

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn

### Backend Setup

1. **Navigate to backend directory:**
   ```bash
   cd backend
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask server:**
   ```bash
   python app.py
   ```

The backend will start on `http://localhost:5000`

### Frontend Setup

1. **Navigate to frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start the React development server:**
   ```bash
   npm start
   ```

The frontend will start on `http://localhost:3000`

## 🏗️ Project Structure

```
hairstyle-recommendation-system/
├── backend/
│   ├── app.py                 # Flask backend with face detection API
│   └── requirements.txt       # Python dependencies
├── frontend/
│   ├── public/
│   │   └── index.html        # Main HTML file
│   ├── src/
│   │   ├── components/
│   │   │   ├── Header.js     # Application header
│   │   │   ├── FaceDetection.js  # Webcam and upload functionality
│   │   │   └── Results.js    # Results display component
│   │   ├── App.js            # Main application component
│   │   ├── index.js          # React entry point
│   │   └── index.css         # Global styles
│   ├── package.json          # Node.js dependencies
│   └── README.md
├── faceshape-master/         # Dataset directory
└── README.md                 # This file
```

## 🔧 API Endpoints

### POST `/api/detect-face-shape`
Analyzes an uploaded image to detect face shape and provide hairstyle recommendations.

**Request Body:**
```json
{
  "image": "base64_encoded_image_data"
}
```

**Response:**
```json
{
  "face_shape": "oval",
  "confidence": 0.85,
  "recommendations": {
    "description": "Oval faces are considered the most versatile...",
    "styles": [
      {
        "name": "Long Layers",
        "description": "Soft, face-framing layers...",
        "youtube_link": "https://youtube.com/...",
        "image_url": "https://images.unsplash.com/..."
      }
    ]
  }
}
```

### GET `/api/health`
Health check endpoint to verify the API is running.

## 🎨 Customization

### Adding New Hairstyles
Edit the `get_hairstyle_recommendations()` function in `backend/app.py` to add new hairstyles for each face shape.

### Modifying Face Shape Detection
The current implementation uses a simplified ratio-based approach. You can enhance it by:
- Training a CNN model on the provided dataset
- Using more sophisticated facial measurements
- Implementing ensemble methods for better accuracy

### Styling Changes
All styles are in CSS files within the `frontend/src/components/` directory. The app uses a modern design system with CSS variables for easy theming.

## 📱 Usage

1. **Open the application** in your web browser
2. **Choose input method**:
   - Use webcam to take a photo
   - Upload an existing image
3. **Follow instructions** for best results:
   - Ensure face is clearly visible
   - Remove accessories (glasses, hats)
   - Look directly at camera
4. **View results**:
   - Your detected face shape
   - Personalized hairstyle recommendations
   - YouTube tutorial links
   - Pro tips for your face shape

## 🛠️ Technology Stack

### Backend
- **Flask**: Python web framework
- **MediaPipe**: Google's ML solution for face detection
- **OpenCV**: Computer vision library
- **PIL**: Image processing

### Frontend
- **React**: JavaScript UI library
- **React Webcam**: Webcam integration
- **Axios**: HTTP client
- **Lucide React**: Icon library
- **CSS3**: Modern styling with animations

## 🔮 Future Enhancements

- [ ] User accounts and history
- [ ] Feedback system (like/dislike hairstyles)
- [ ] Virtual try-on with AR
- [ ] Integration with salon booking systems
- [ ] More sophisticated ML models
- [ ] Support for different hair textures
- [ ] Seasonal trend recommendations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: [Face Shape Classification Dataset](https://github.com/dsmlr/faceshape)
- **Face Detection**: [MediaPipe](https://mediapipe.dev/)
- **UI Inspiration**: Modern web design principles
- **Hairstyle Knowledge**: Beauty industry expertise

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/yourusername/hairstyle-recommendation-system/issues) page
2. Create a new issue with detailed information
3. Include your operating system and browser information

---

**Happy styling! 💇‍♀️✨**
"# AI-powered-face-shape-detection-and-hairstyle-recommender" 
