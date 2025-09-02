import React from 'react';
import { RotateCcw, ExternalLink, Heart, Star } from 'lucide-react';
import './Results.css';

const Results = ({ results, onReset }) => {
  const { face_shape, confidence, recommendations } = results;

  const getFaceShapeIcon = (shape) => {
    const icons = {
      oval: '🥚',
      round: '⭕',
      square: '⬜',
      heart: '❤️',
      oblong: '📏'
    };
    return icons[shape] || '👤';
  };

  const getFaceShapeDescription = (shape) => {
    const descriptions = {
      oval: "Oval faces are considered the most versatile and can pull off almost any hairstyle!",
      round: "Round faces benefit from styles that add length and angles to create definition.",
      square: "Square faces have strong angles - soften them with rounded, layered styles.",
      heart: "Heart-shaped faces are wider at the top - balance with styles that add width at the bottom.",
      oblong: "Oblong faces are long and narrow - add width and break up the length."
    };
    return descriptions[shape] || "Your face shape has been detected!";
  };

  const formatConfidence = (conf) => {
    return Math.round(conf * 100);
  };

  return (
    <div className="results">
      <div className="results-header">
        <h2>Your Face Shape Analysis</h2>
        <button className="btn btn-secondary" onClick={onReset}>
          <RotateCcw size={16} />
          Analyze Another Photo
        </button>
      </div>

      <div className="face-shape-result">
        <div className="shape-display">
          <div className="shape-icon">
            {getFaceShapeIcon(face_shape)}
          </div>
          <div className="shape-info">
            <h3 className="shape-name">{face_shape.charAt(0).toUpperCase() + face_shape.slice(1)}</h3>
            <div className="confidence">
              <Star size={16} />
              <span>{formatConfidence(confidence)}% Confidence</span>
            </div>
          </div>
        </div>
        <p className="shape-description">
          {getFaceShapeDescription(face_shape)}
        </p>
      </div>

      <div className="recommendations-section">
        <h3>Recommended Hairstyles for You</h3>
        <p className="recommendations-intro">
          Based on your {face_shape} face shape, here are some hairstyles that will complement your features:
        </p>

        <div className="hairstyles-grid">
          {recommendations.styles.map((style, index) => (
            <div key={index} className="hairstyle-card">
              <div className="hairstyle-image">
                <img src={style.image_url} alt={style.name} />
              </div>
              <div className="hairstyle-content">
                <h4>{style.name}</h4>
                <p>{style.description}</p>
                <a 
                  href={style.youtube_link} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="tutorial-link"
                >
                  <ExternalLink size={16} />
                  Watch Tutorial
                </a>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="tips-section">
        <h3>💡 Pro Tips for {face_shape.charAt(0).toUpperCase() + face_shape.slice(1)} Faces</h3>
        <div className="tips-content">
          {face_shape === 'oval' && (
            <ul>
              <li>You can experiment with any hairstyle length</li>
              <li>Try different partings to change your look</li>
              <li>Consider your hair texture and thickness</li>
            </ul>
          )}
          {face_shape === 'round' && (
            <ul>
              <li>Opt for styles that add height and length</li>
              <li>Side parts work great for creating angles</li>
              <li>Avoid very short, rounded cuts</li>
            </ul>
          )}
          {face_shape === 'square' && (
            <ul>
              <li>Soft, layered cuts will soften your angles</li>
              <li>Consider side-swept bangs</li>
              <li>Medium to long lengths work best</li>
            </ul>
          )}
          {face_shape === 'heart' && (
            <ul>
              <li>Chin-length cuts add width to balance your chin</li>
              <li>Side-swept bangs can work well</li>
              <li>Avoid very short cuts that emphasize the narrow chin</li>
            </ul>
          )}
          {face_shape === 'oblong' && (
            <ul>
              <li>Bangs help break up the length</li>
              <li>Medium lengths with layers add width</li>
              <li>Avoid very long, straight styles</li>
            </ul>
          )}
        </div>
      </div>

      <div className="action-buttons">
        <button className="btn btn-primary" onClick={onReset}>
          Analyze Another Photo
        </button>
      </div>
    </div>
  );
};

export default Results;
