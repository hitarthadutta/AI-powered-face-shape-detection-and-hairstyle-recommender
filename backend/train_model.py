import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib
from PIL import Image
import glob

def extract_features(image_path):
    """Extract features from face image"""
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize to standard size
        gray = cv2.resize(gray, (100, 100))
        
        # Extract basic features
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
        print(f"Error processing {image_path}: {e}")
        return None

def load_dataset():
    """Load the face shape dataset"""
    dataset_path = "../faceshape-master/published_dataset"
    face_shapes = ['heart', 'oblong', 'oval', 'round', 'square']
    
    X = []  # Features
    y = []  # Labels
    
    for shape in face_shapes:
        shape_path = os.path.join(dataset_path, shape)
        if not os.path.exists(shape_path):
            print(f"Warning: {shape_path} not found")
            continue
            
        print(f"Loading {shape} images...")
        image_files = glob.glob(os.path.join(shape_path, "*.jpg"))
        
        for image_file in image_files:
            features = extract_features(image_file)
            if features is not None:
                X.append(features)
                y.append(shape)
    
    return np.array(X), np.array(y)

def train_models():
    """Train multiple ML models and select the best one"""
    print("🚀 Loading dataset...")
    X, y = load_dataset()
    
    if len(X) == 0:
        print("❌ No data loaded. Check dataset path.")
        return None
    
    print(f"✅ Loaded {len(X)} samples with {X.shape[1]} features each")
    face_shapes = ['heart', 'oblong', 'oval', 'round', 'square']
    print(f"📊 Class distribution: {np.bincount([face_shapes.index(label) for label in y])}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train multiple models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42, probability=True)
    }
    
    best_model = None
    best_score = 0
    best_name = ""
    
    for name, model in models.items():
        print(f"\n🔧 Training {name}...")
        model.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"📈 {name} Accuracy: {accuracy:.3f}")
        print(f"📋 Classification Report:")
        print(classification_report(y_test, y_pred))
        
        if accuracy > best_score:
            best_score = accuracy
            best_model = model
            best_name = name
    
    print(f"\n🏆 Best Model: {best_name} with accuracy: {best_score:.3f}")
    
    # Save the best model
    if best_model is not None:
        model_path = "face_shape_model.pkl"
        joblib.dump(best_model, model_path)
        print(f"💾 Model saved to: {model_path}")
        
        # Save feature names for reference
        feature_info = {
            'feature_count': X.shape[1],
            'classes': ['heart', 'oblong', 'oval', 'round', 'square'],
            'model_type': best_name,
            'accuracy': best_score
        }
        
        import json
        with open('model_info.json', 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        print("✅ Model training completed successfully!")
        return best_model
    
    return None

if __name__ == "__main__":
    print("🎯 Face Shape Classification Model Training")
    print("=" * 50)
    
    # Check if dataset exists
    dataset_path = "../faceshape-master/published_dataset"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        print("Please ensure the faceshape-master folder is in the parent directory")
        exit(1)
    
    # Train the model
    model = train_models()
    
    if model is not None:
        print("\n🎉 Training completed! You can now use the trained model in your app.")
    else:
        print("\n❌ Training failed. Please check the dataset and try again.")
