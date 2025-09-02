import React, { useState } from 'react';
import Header from './components/Header';
import FaceDetection from './components/FaceDetection';
import Results from './components/Results';
import './App.css';

function App() {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleResults = (data) => {
    setResults(data);
    setError(null);
  };

  const handleError = (errorMessage) => {
    setError(errorMessage);
    setResults(null);
  };

  const handleLoading = (isLoading) => {
    setLoading(isLoading);
  };

  const resetApp = () => {
    setResults(null);
    setError(null);
    setLoading(false);
  };

  return (
    <div className="App">
      <Header />
      <div className="container">
        {!results && !error && (
          <FaceDetection 
            onResults={handleResults}
            onError={handleError}
            onLoading={handleLoading}
          />
        )}
        
        {loading && (
          <div className="loading">
            <div className="spinner"></div>
            <p style={{ marginLeft: '16px', color: 'white', fontSize: '18px' }}>
              Analyzing your face shape...
            </p>
          </div>
        )}
        
        {error && (
          <div className="error-container">
            <div className="card">
              <h2 style={{ color: '#e74c3c', marginBottom: '16px' }}>Oops! Something went wrong</h2>
              <p style={{ marginBottom: '24px', color: '#666' }}>{error}</p>
              <button className="btn btn-primary" onClick={resetApp}>
                Try Again
              </button>
            </div>
          </div>
        )}
        
        {results && (
          <Results 
            results={results}
            onReset={resetApp}
          />
        )}
      </div>
    </div>
  );
}

export default App;
