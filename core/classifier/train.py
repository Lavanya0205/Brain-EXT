from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV
import joblib
from collections import Counter
from core.classifier.dataset import DATASET


# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Prepare data
texts = [item["text"] for item in DATASET]
labels = [item["label"] for item in DATASET]

# Encode text → vectors
X = embedder.encode(texts)

# Encode labels → numbers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

print("FULL DATASET LABEL DISTRIBUTION:")
print(Counter(labels))


# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train calibrated classifier
base_clf = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

classifier = CalibratedClassifierCV(
    base_clf,
    method="sigmoid",
    cv=5
)

classifier.fit(X_train, y_train)

expected_labels = {"frontal", "temporal", "parietal", "occipital"}
seen_labels = set(labels)

assert expected_labels == seen_labels, f"Missing labels: {expected_labels - seen_labels}"

# Evaluate
y_pred = classifier.predict(X_test)
print("TRAIN LABEL DISTRIBUTION:")
print(Counter(y_train))
print("TEST LABEL DISTRIBUTION:")
print(Counter(y_test))
print("Classes seen in training:", label_encoder.classes_)
print("Training samples:", len(X_train))
print("Test samples:", len(X_test))
print("Label distribution:", Counter(labels))
print(classification_report(y_test, y_pred))

# Save model + encoder
joblib.dump(classifier, "core/classifier/lobe_classifier.joblib")
joblib.dump(label_encoder, "core/classifier/label_encoder.joblib")

print("Lobe classifier trained and saved")

