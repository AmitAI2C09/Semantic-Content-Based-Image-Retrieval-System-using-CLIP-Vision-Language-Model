import os
import numpy as np
from numpy import linalg as LA

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

# ---------------------------------
# Load pretrained VGG16 model
# ---------------------------------
model = VGG16(
    weights='imagenet',
    input_shape=(224, 224, 3),
    pooling='max',
    include_top=False
)

print("VGG16 model loaded successfully...")

# ---------------------------------
# Feature Extraction Function
# ---------------------------------
def extract_vgg16_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    features = model.predict(img_array)
    features = features.flatten()          # (1,512) -> (512,)
    features = features / LA.norm(features)  # normalize

    return features


# ---------------------------------
# MAIN CBIR PIPELINE
# ---------------------------------
if __name__ == "__main__":

    database_folder = "all_images"
    query_folder = "query_images"

    # -------------------------------
    # Step 1: Load database images
    # -------------------------------
    database_features = []
    database_names = []

    print("\nProcessing database images...")

    for img_name in os.listdir(database_folder):
        if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(database_folder, img_name)
            feats = extract_vgg16_features(img_path)
            database_features.append(feats)
            database_names.append(img_name)

    database_features = np.array(database_features)

    print("Database images loaded:", len(database_features))

    # -------------------------------
    # Step 2: Load query image
    # -------------------------------
    query_images = [f for f in os.listdir(query_folder)
                    if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    if len(query_images) == 0:
        print("ERROR: No query images found!")
        exit()

    query_img_path = os.path.join(query_folder, query_images[0])
    query_features = extract_vgg16_features(query_img_path)

    print("Query image:", query_images[0])

    # -------------------------------
    # Step 3: Similarity computation
    # -------------------------------
    similarities = []

    for i, db_feat in enumerate(database_features):
        sim = np.dot(query_features, db_feat)   # cosine similarity
        similarities.append((database_names[i], sim))

    # Sort by similarity score (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)

    # -------------------------------
    # Step 4: Display results
    # -------------------------------
    print("\nTop Matching Images (CBIR Results):\n")

    for name, score in similarities[:10]:
        print(f"{name}  ---> similarity score: {score:.4f}")

    # -------------------------------
    # Step 5: Save feature database
    # -------------------------------
    np.save("database_features.npy", database_features)
    np.save("database_names.npy", np.array(database_names))

    print("\nFeature database saved:")
    print("database_features.npy")
    print("database_names.npy")

    print("\nCBIR system execution completed successfully!")
