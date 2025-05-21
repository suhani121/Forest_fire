import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Paths
fire_dir = 'fire'
nofire_dir = 'nofire'

# Feature extraction: color histogram (simple)
def extract_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.resize(image, (128, 128))
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()

# Load data
X = []
y = []

for label, folder in enumerate([fire_dir, nofire_dir]):
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(folder, filename)
            features = extract_features(path)
            if features is not None:
                X.append(features)
                y.append(label)

X = np.array(X)
y = np.array(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Fire', 'Fire']))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(clf, 'fire_detection_rf_model.pkl')
print("✅ Model saved as fire_detection_rf_model.pkl")
