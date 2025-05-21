# ff1.py

import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def extract_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not read image: {image_path}")
        return None
    image = cv2.resize(image, (128, 128))
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()

X, y = [], []
for label in ['fire', 'nofire']:
    folder = os.path.join('.', label)
    for filename in os.listdir(folder):
        if filename.startswith('.'):
            continue
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(folder, filename)
            features = extract_features(path)
            if features is not None:
                X.append(features)
                y.append(label)

X = np.array(X)
y = LabelEncoder().fit_transform(y)

if len(X) == 0:
    print("❗ No valid images found!")
    exit()

kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X)
X_combined = np.hstack((X, clusters.reshape(-1, 1)))

X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

nb = GaussianNB()
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

nb.fit(X_train, y_train)
xgb.fit(X_train, y_train)

print("\n=== Naive Bayes ===")
print(classification_report(y_test, nb.predict(X_test)))

print("\n=== XGBoost ===")
print(classification_report(y_test, xgb.predict(X_test)))

