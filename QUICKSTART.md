# 🚀 StyleAI Quick Start Guide

Get your AI-powered hairstyle recommendation system running in minutes!

## ⚡ Super Quick Start (Windows)

1. **Double-click `start.bat`** - This will automatically:
   - Install all dependencies
   - Start the backend server
   - Start the frontend application
   - Open both services in separate windows

2. **Wait for both services to start** (about 2-3 minutes)
3. **Open your browser** and go to `http://localhost:3000`

## ⚡ Super Quick Start (PowerShell)

1. **Right-click `start.ps1`** and select "Run with PowerShell"
2. **Wait for both services to start** (about 2-3 minutes)
3. **Open your browser** and go to `http://localhost:3000`

## 🔧 Manual Setup (if automatic doesn't work)

### Prerequisites Check
```bash
python --version  # Should be 3.8+
node --version   # Should be 16+
npm --version    # Should be available
```

### Step 1: Setup Environment
```bash
# Run the setup script
python setup.py
```

### Step 2: Start Backend
```bash
cd backend
python app.py
```
✅ Backend will be running at `http://localhost:5000`

### Step 3: Start Frontend (in new terminal)
```bash
cd frontend
npm start
```
✅ Frontend will be running at `http://localhost:3000`

## 🧪 Testing the System

### Test Backend API
```bash
python test_backend.py
```

### Test with Dataset Images
```bash
python demo.py
```

## 🌐 Using the Application

1. **Open** `http://localhost:3000` in your browser
2. **Choose input method**:
   - 📷 **Webcam**: Click "Start Camera" and take a photo
   - 📁 **Upload**: Click "Choose File" to upload an image
3. **Follow instructions** for best results
4. **View your results**:
   - Detected face shape
   - Personalized hairstyle recommendations
   - YouTube tutorial links
   - Pro tips for your face shape

## 🎯 Supported Face Shapes

- **🥚 Oval** - Most versatile, can pull off any style
- **⭕ Round** - Benefits from length and angles
- **⬜ Square** - Strong angles, softened by rounded styles
- **❤️ Heart** - Wider at top, balanced with bottom width
- **📏 Oblong** - Long and narrow, needs width and breaks

## 🐛 Troubleshooting

### Backend won't start?
- Check if Python 3.8+ is installed
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check if port 5000 is available

### Frontend won't start?
- Check if Node.js 16+ is installed
- Ensure all dependencies are installed: `npm install`
- Check if port 3000 is available

### Face detection not working?
- Ensure good lighting in photos
- Remove glasses, hats, or accessories
- Look directly at camera
- Check backend logs for errors

### Common Issues
- **Port already in use**: Close other applications using ports 3000 or 5000
- **Permission denied**: Run PowerShell as Administrator
- **Dependencies failed**: Try running `python setup.py` again

## 📱 Features

- ✨ **Real-time webcam capture**
- 📁 **Image upload support**
- 🤖 **AI-powered face shape detection**
- 💇‍♀️ **Personalized hairstyle recommendations**
- 🎥 **Direct YouTube tutorial links**
- 📱 **Mobile responsive design**
- 🎨 **Beautiful, modern UI**

## 🔮 What's Next?

After getting the basic system running, you can:

1. **Train better models** using the provided dataset
2. **Add more hairstyles** to the recommendations
3. **Implement user accounts** and history
4. **Add feedback system** (like/dislike styles)
5. **Integrate with salon booking** systems

## 📞 Need Help?

1. Check the main `README.md` for detailed documentation
2. Run the test scripts to diagnose issues
3. Check the console logs for error messages
4. Ensure all prerequisites are met

---

**🎉 You're all set! Enjoy discovering your perfect hairstyle with AI! 💇‍♀️✨**
