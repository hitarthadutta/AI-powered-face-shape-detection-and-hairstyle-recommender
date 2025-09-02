import React, { useState, useRef, useCallback } from 'react';
import Webcam from 'react-webcam';
import { Camera, Upload, RotateCcw } from 'lucide-react';
import axios from 'axios';
import './FaceDetection.css';

const FaceDetection = ({ onResults, onError, onLoading }) => {
  const [capturedImage, setCapturedImage] = useState(null);
  const [isWebcamActive, setIsWebcamActive] = useState(false);
  const [uploadedImage, setUploadedImage] = useState(null);
  const [gender, setGender] = useState('');
  const [genderError, setGenderError] = useState('');
  const webcamRef = useRef(null);
  const fileInputRef = useRef(null);

  const capture = useCallback(() => {
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot();
      setCapturedImage(imageSrc);
      setIsWebcamActive(false);
    }
  }, [webcamRef]);

  const retake = () => {
    setCapturedImage(null);
    setUploadedImage(null);
    setIsWebcamActive(false);
    setGenderError('');
  };

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setUploadedImage(e.target.result);
        setCapturedImage(null);
        setIsWebcamActive(false);
      };
      reader.readAsDataURL(file);
    }
  };

  const analyzeImage = async () => {
    const imageToAnalyze = capturedImage || uploadedImage;
    if (!imageToAnalyze) return;

    if (!gender) {
      setGenderError('Please select your gender.');
      return;
    }
    setGenderError('');

    onLoading(true);
    try {
      const response = await axios.post('/api/detect-face-shape', {
        image: imageToAnalyze,
        gender: gender
      });
      onResults(response.data);
    } catch (error) {
      console.error('Error analyzing image:', error);
      const errorMessage = error.response?.data?.error || 'Failed to analyze image. Please try again.';
      onError(errorMessage);
    } finally {
      onLoading(false);
    }
  };

  const startWebcam = () => {
    setIsWebcamActive(true);
    setCapturedImage(null);
    setUploadedImage(null);
  };

  const triggerFileUpload = () => {
    fileInputRef.current.click();
  };

  return (
    <div className="face-detection">
      <div className="detection-options">
        <div className="option-card">
          <Camera className="option-icon" />
          <h3>Use Webcam</h3>
          <p>Take a photo using your device's camera</p>
          <button 
            className="btn btn-primary" 
            onClick={startWebcam}
            disabled={isWebcamActive}
          >
            Start Camera
          </button>
        </div>

        <div className="option-card">
          <Upload className="option-icon" />
          <h3>Upload Image</h3>
          <p>Upload a photo from your device</p>
          <button 
            className="btn btn-secondary" 
            onClick={triggerFileUpload}
          >
            Choose File
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileUpload}
            style={{ display: 'none' }}
          />
        </div>
      </div>

      {/* Gender Selector */}
      <div className="card" style={{ marginBottom: '24px' }}>
        <h3 style={{ marginBottom: '12px', color: '#333' }}>Select Gender</h3>
        <div style={{ display: 'flex', gap: '16px', alignItems: 'center', flexWrap: 'wrap' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <input
              type="radio"
              name="gender"
              value="female"
              checked={gender === 'female'}
              onChange={() => setGender('female')}
            />
            Female
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <input
              type="radio"
              name="gender"
              value="male"
              checked={gender === 'male'}
              onChange={() => setGender('male')}
            />
            Male
          </label>
          {genderError && (
            <span style={{ color: '#e74c3c', fontWeight: 600 }}>{genderError}</span>
          )}
        </div>
      </div>

      {(isWebcamActive || capturedImage || uploadedImage) && (
        <div className="camera-section">
          {isWebcamActive && (
            <div className="webcam-container">
              <Webcam
                ref={webcamRef}
                screenshotFormat="image/jpeg"
                className="webcam"
              />
              <button className="btn btn-primary capture-btn" onClick={capture}>
                Capture Photo
              </button>
            </div>
          )}

          {(capturedImage || uploadedImage) && (
            <div className="captured-image-container">
              <img 
                src={capturedImage || uploadedImage} 
                alt="Captured" 
                className="captured-image"
              />
              <div className="image-actions">
                <button className="btn btn-secondary" onClick={retake}>
                  <RotateCcw size={16} />
                  Retake
                </button>
                <button className="btn btn-primary" onClick={analyzeImage}>
                  Analyze Face Shape
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      <div className="instructions">
        <h3>How to get the best results:</h3>
        <ul>
          <li>Ensure your face is clearly visible and well-lit</li>
          <li>Remove glasses, hats, or other accessories</li>
          <li>Look directly at the camera with a neutral expression</li>
          <li>Make sure your entire face is in the frame</li>
        </ul>
      </div>
    </div>
  );
};

export default FaceDetection;
