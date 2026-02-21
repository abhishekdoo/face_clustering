import os
import cv2
import base64
import random
import shutil
import numpy as np
import warnings
import sys
from mtcnn import MTCNN
from keras_facenet import FaceNet
from numpy.linalg import norm

warnings.filterwarnings("ignore")

# -----------------------------
# ARGUMENTS
# -----------------------------
if len(sys.argv) != 4:
    print("Usage:")
    print("python cluster.py <start_accuracy> <end_accuracy> <iterations>")
    print("Example:")
    print("python cluster.py 50 80 10")
    sys.exit()

start_acc = int(sys.argv[1])
end_acc = int(sys.argv[2])
ITERATIONS = int(sys.argv[3])

# -----------------------------
# CONFIG
# -----------------------------
TEST_DIR = "test"
OUTPUT_BASE = "output_clusters"
TEMP_BASE = "temp_clusters"

os.makedirs(OUTPUT_BASE, exist_ok=True)
os.makedirs(TEMP_BASE, exist_ok=True)

# -----------------------------
# INIT MODELS
# -----------------------------
detector = MTCNN()
embedder = FaceNet()

# -----------------------------
# FUNCTIONS
# -----------------------------
def detect_face(image):
    try:
        faces = detector.detect_faces(image)
    except Exception:
        return None

    for f in faces:
        x, y, w, h = f["box"]

        if w <= 0 or h <= 0:
            continue

        x, y = max(0, x), max(0, y)
        face = image[y:y+h, x:x+w]

        if face.size == 0:
            continue

        face = cv2.resize(face, (160, 160))
        return face

    return None


def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


def generate_id():
    rand_bytes = random.randbytes(4)
    b64 = base64.urlsafe_b64encode(rand_bytes).decode()
    return b64[:6]


def load_images(input_dir):
    images = []

    for img_name in os.listdir(input_dir):

        img_path = os.path.join(input_dir, img_name)

        if os.path.isdir(img_path):
            continue

        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        image = cv2.imread(img_path)
        if image is None:
            continue

        face = detect_face(image)
        if face is None:
            continue

        emb = embedder.embeddings(np.expand_dims(face, axis=0))[0]

        images.append({
            "name": img_name,
            "path": img_path,
            "embedding": emb
        })

    return images


# -----------------------------
# MULTI-THRESHOLD LOOP
# -----------------------------
for acc in range(start_acc, end_acc + 1, 10):

    threshold = acc / 100.0
    FINAL_DIR = os.path.join(OUTPUT_BASE, f"final_groups_{acc}")
    os.makedirs(FINAL_DIR, exist_ok=True)

    print(f"\n🚀 Accuracy {acc}% (threshold={threshold})")

    current_input_dir = TEST_DIR

    for iteration in range(1, ITERATIONS + 1):

        print(f"\nIteration {iteration}")

        TEMP_DIR = os.path.join(TEMP_BASE, f"acc{acc}_iter{iteration}")
        os.makedirs(TEMP_DIR, exist_ok=True)

        images = load_images(current_input_dir)

        if not images:
            print("No valid faces.")
            break

        clusters = []
        used = set()

        # -----------------------------
        # INITIAL CLUSTERING
        # -----------------------------
        for anchor in images:

            if anchor["name"] in used:
                continue

            cluster = [anchor]
            used.add(anchor["name"])

            for candidate in images:

                if candidate["name"] in used:
                    continue

                score = cosine_similarity(
                    anchor["embedding"],
                    candidate["embedding"]
                )

                if score >= threshold:
                    cluster.append(candidate)
                    used.add(candidate["name"])

            clusters.append(cluster)

        print(f"Initial clusters: {len(clusters)}")

        # -----------------------------
        # SMART MERGING LOGIC
        # -----------------------------
        merged = True

        while merged:
            merged = False

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):

                    cluster_a = clusters[i]
                    cluster_b = clusters[j]

                    print(f"\n🔍 Validating Cluster {i} vs {j}")

                    embeddings_a = [img["embedding"] for img in cluster_a]
                    centroid_a = np.mean(embeddings_a, axis=0)

                    matched = []
                    unmatched = []

                    for img_b in cluster_b:

                        score = cosine_similarity(img_b["embedding"], centroid_a)

                        if score >= threshold:
                            matched.append(img_b)
                        else:
                            unmatched.append(img_b)

                    print(f"Matched: {len(matched)} / {len(cluster_b)}")

                    # FULL MERGE
                    if len(matched) == len(cluster_b):

                        print("✅ FULL MERGE")

                        cluster_a.extend(cluster_b)
                        clusters.pop(j)

                        merged = True
                        break

                    # PARTIAL MERGE
                    elif matched:

                        print("⚡ PARTIAL MERGE")

                        cluster_a.extend(matched)
                        clusters[j] = unmatched

                        merged = True
                        break

                    else:
                        print("❌ NO MERGE")

                if merged:
                    break

        print(f"Clusters after merge: {len(clusters)}")

        # -----------------------------
        # SAVE MERGED OUTPUT
        # -----------------------------
        merged_dir = os.path.join(TEMP_DIR, "merged")
        os.makedirs(merged_dir, exist_ok=True)

        for cluster in clusters:
            for img in cluster:
                shutil.copy(img["path"], os.path.join(merged_dir, img["name"]))

        current_input_dir = merged_dir

    # -----------------------------
    # FINAL SAVE
    # -----------------------------
    print("\n💾 Saving Final Clusters")

    images = load_images(current_input_dir)
    clusters = []
    used = set()

    for anchor in images:

        if anchor["name"] in used:
            continue

        cluster = [anchor]
        used.add(anchor["name"])

        for candidate in images:

            if candidate["name"] in used:
                continue

            score = cosine_similarity(anchor["embedding"], candidate["embedding"])

            if score >= threshold:
                cluster.append(candidate)
                used.add(candidate["name"])

        clusters.append(cluster)

    for cluster in clusters:

        cluster_id = generate_id()
        cluster_folder = os.path.join(FINAL_DIR, cluster_id)
        os.makedirs(cluster_folder, exist_ok=True)

        for img in cluster:
            shutil.copy(img["path"], os.path.join(cluster_folder, img["name"]))

print("\n✅ Clustering Pipeline Complete.")