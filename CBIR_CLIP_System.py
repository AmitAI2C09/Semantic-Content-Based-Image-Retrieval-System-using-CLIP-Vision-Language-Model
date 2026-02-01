import os
import torch
import numpy as np
from flask import Flask, request, render_template_string, send_from_directory
from PIL import Image
import clip

# ---------------------------------
# PATH CONFIG (ABSOLUTE PATH SAFE)
# ---------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATABASE_FOLDER = os.path.join(BASE_DIR, "all_images")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
TOP_K = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------------------
# Load CLIP model
# ---------------------------------
model, preprocess = clip.load("ViT-B/32", device=DEVICE)
print("CLIP semantic model loaded on", DEVICE)

# ---------------------------------
# Feature extraction
# ---------------------------------
def extract_features(img_path):
    img = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        features = model.encode_image(img)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()[0]

# ---------------------------------
# Build semantic database
# ---------------------------------
db_features = []
db_names = []

print("Building semantic feature database...")

if not os.path.exists(DATABASE_FOLDER):
    raise Exception(f"DATABASE FOLDER NOT FOUND: {DATABASE_FOLDER}")

for img_name in os.listdir(DATABASE_FOLDER):
    if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        path = os.path.join(DATABASE_FOLDER, img_name)
        feats = extract_features(path)
        db_features.append(feats)
        db_names.append(img_name)

if len(db_features) == 0:
    raise Exception("No images found in all_images folder")


db_features = np.array(db_features)
print("Semantic database built with", len(db_features), "images")

# ---------------------------------
# Flask App
# ---------------------------------
app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Semantic CBIR System</title>
    <style>
        body{font-family:Arial;background:#f2f2f2;margin:0;padding:0;}
        .box{margin:30px auto;width:85%;background:white;padding:30px;border-radius:12px;box-shadow:0 0 10px rgba(0,0,0,0.1);text-align:center;}
        img{width:180px;height:180px;object-fit:cover;margin:10px;border-radius:10px;border:1px solid #ccc;}
        .grid{display:flex;justify-content:center;flex-wrap:wrap;}
        button{padding:12px 25px;border:none;background:#4CAF50;color:white;border-radius:6px;cursor:pointer;font-size:16px;}
        button:hover{background:#45a049;}
    </style>
</head>
<body>
<div class="box">
    <h2>Semantic Content-Based Image Retrieval System</h2>

    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="image" required>
        <br><br>
        <button type="submit">Search Similar Images</button>
    </form>

    {% if query_img %}
        <h3>Query Image</h3>
        <img src="/uploads/{{query_img}}">
    {% endif %}

    {% if results %}
        <h3>Semantically Similar Images</h3>
        <div class="grid">
        {% for img,score in results %}
            <div>
                <img src="/all_images/{{img}}"><br>
                <small>{{score}}</small>
            </div>
        {% endfor %}
        </div>
    {% endif %}
</div>
</body>
</html>
"""

@app.route('/', methods=['GET','POST'])
def index():
    results = None
    query_img = None

    if request.method == 'POST':
        file = request.files['image']
        query_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(query_path)

        q_feat = extract_features(query_path)

        sims = []
        for i, db_feat in enumerate(db_features):
            sim = float(np.dot(q_feat, db_feat))
            sims.append((db_names[i], round(sim, 4)))

        sims.sort(key=lambda x: x[1], reverse=True)
        results = sims[:TOP_K]
        query_img = file.filename

    return render_template_string(HTML, results=results, query_img=query_img)

# ---------------------------------
# Static routes
# ---------------------------------
@app.route('/all_images/<filename>')
def serve_db(filename):
    return send_from_directory(DATABASE_FOLDER, filename)

@app.route('/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# ---------------------------------
# Run
# ---------------------------------
if __name__ == '__main__':
    app.run(debug=True)
