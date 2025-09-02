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
import json

def extract_consistent_features(image_path):
    """Extract consistent features for face shape classification"""
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize to standard size for consistency
        gray = cv2.resize(gray, (200, 200))
        
        # Use OpenCV face detection to get face region
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            # If no face detected, use the whole image
            face_region = gray
            x, y, w, h = 0, 0, 200, 200
        else:
            # Use the largest detected face
            face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = face
            face_region = gray[y:y+h, x:x+w]
        
        # Resize face region to standard size
        face_region = cv2.resize(face_region, (100, 100))
        
        features = []
        
        # 1. Basic geometric ratios
        h_face, w_face = face_region.shape
        aspect_ratio = h_face / w_face
        features.append(aspect_ratio)
        
        # 2. Face area analysis
        # Divide face into regions and analyze
        # Top third (forehead)
        top_third = face_region[:h_face//3, :]
        # Middle third (cheeks)
        middle_third = face_region[h_face//3:2*h_face//3, :]
        # Bottom third (jaw)
        bottom_third = face_region[2*h_face//3:, :]
        
        # Calculate width at different heights
        top_width = np.mean(np.sum(top_third > 100, axis=0))
        middle_width = np.mean(np.sum(middle_third > 100, axis=0))
        bottom_width = np.mean(np.sum(bottom_third > 100, axis=0))
        
        # Width ratios
        top_middle_ratio = top_width / (middle_width + 1e-6)
        middle_bottom_ratio = middle_width / (bottom_width + 1e-6)
        top_bottom_ratio = top_width / (bottom_width + 1e-6)
        
        features.extend([top_middle_ratio, middle_bottom_ratio, top_bottom_ratio])
        
        # 3. Symmetry analysis
        left_half = face_region[:, :w_face//2]
        right_half = face_region[:, w_face//2:]
        symmetry = np.mean(np.abs(left_half - np.fliplr(right_half)))
        features.append(symmetry)
        
        # 4. Edge analysis
        edges = cv2.Canny(face_region, 50, 150)
        edge_density = np.sum(edges > 0) / (h_face * w_face)
        features.append(edge_density)
        
        # 5. Brightness distribution
        # Horizontal brightness profile
        horizontal_profile = np.mean(face_region, axis=0)
        # Vertical brightness profile
        vertical_profile = np.mean(face_region, axis=1)
        
        # Calculate profile statistics
        h_profile_std = np.std(horizontal_profile)
        v_profile_std = np.std(vertical_profile)
        h_profile_mean = np.mean(horizontal_profile)
        v_profile_mean = np.mean(vertical_profile)
        
        features.extend([h_profile_std, v_profile_std, h_profile_mean, v_profile_mean])
        
        # 6. Contour analysis
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            # Calculate contour properties
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                circularity = 0
            features.append(circularity)
        else:
            features.append(0)
        
        # 7. Histogram features (simplified)
        hist = cv2.calcHist([face_region], [0], None, [16], [0, 256])
        # Use only the most important histogram bins
        features.extend(hist.flatten()[:8])  # Only first 8 bins
        
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
            features = extract_consistent_features(image_file)
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
    
    # Train multiple models with better parameters
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=300, 
            max_depth=15, 
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
            max_features='sqrt'
        ),
        'SVM': SVC(
            kernel='rbf', 
            C=10.0, 
            gamma='scale',
            random_state=42, 
            probability=True
        )
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
        model_path = "face_shape_model_simple.pkl"
        joblib.dump(best_model, model_path)
        print(f"💾 Model saved to: {model_path}")
        
        # Save feature names for reference
        feature_info = {
            'feature_count': X.shape[1],
            'classes': ['heart', 'oblong', 'oval', 'round', 'square'],
            'model_type': best_name,
            'accuracy': best_score,
            'version': 'simple_consistent'
        }
        
        with open('model_info_simple.json', 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        print("✅ Simple consistent model training completed successfully!")
        return best_model
    
    return None

if __name__ == "__main__":
    print("🎯 Simple Consistent Face Shape Classification Model Training")
    print("=" * 70)
    
    # Check if dataset exists
    dataset_path = "../faceshape-master/published_dataset"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        print("Please ensure the faceshape-master folder is in the parent directory")
        exit(1)
    
    # Train the model
    model = train_models()
    
    if model is not None:
        print("\n🎉 Simple consistent training completed! This should give more reliable results.")
    else:
        print("\n❌ Training failed. Please check the dataset and try again.")
