import React from 'react';
import { Sparkles } from 'lucide-react';
import './Header.css';

const Header = () => {
  return (
    <header className="header">
      <div className="container">
        <div className="header-content">
          <div className="logo">
            <Sparkles className="logo-icon" />
            <h1>StyleAI</h1>
          </div>
          <p className="tagline">
            Discover Your Perfect Hairstyle with AI-Powered Face Shape Detection
          </p>
        </div>
      </div>
    </header>
  );
};

export default Header;
