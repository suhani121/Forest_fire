# ff2.py

import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
    ('lgbm', LGBMClassifier()),
    ('cat', CatBoostClassifier(verbose=0))
]

stacked = StackingClassifier(
    estimators=base_models,
    final_estimator=RandomForestClassifier(),
    passthrough=True
)

stacked.fit(X_train, y_train)
y_pred = stacked.predict(X_test)

print("\n=== Stacked Ensemble ===")
print(classification_report(y_test, y_pred))

