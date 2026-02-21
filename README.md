# Face Clustering Pipeline Setup

## ✅ STEP 1 --- Check Python Version

``` bash
python3.10 --version
```

Must show:

    Python 3.10.x

------------------------------------------------------------------------

## ✅ STEP 2 --- Create Virtual Environment

``` bash
python3.10 -m venv keras_env
```

------------------------------------------------------------------------

## ✅ STEP 3 --- Activate Environment

``` bash
source keras_env/bin/activate
```

You should see:

    (keras_env)

------------------------------------------------------------------------

## ✅ STEP 4 --- Upgrade pip

``` bash
pip install --upgrade pip
```

------------------------------------------------------------------------

## ✅ STEP 5 --- Install Dependencies

``` bash
pip install numpy
pip install scipy
pip install tensorflow==2.20.0
pip install opencv-python
pip install keras-facenet
pip install mtcnn
```

------------------------------------------------------------------------

## ✅ STEP 6 --- Create Project Structure

``` bash
mkdir face_cluster_project
cd face_cluster_project
```

Create folders:

``` bash
mkdir test
mkdir output_clusters
mkdir temp_clusters
```

------------------------------------------------------------------------

## ✅ STEP 7 --- Add Images

Place images inside:

    test/
        img1.jpg
        img2.jpg
        img3.jpg

✔ Flat structure\
✔ No subfolders

------------------------------------------------------------------------

## ✅ STEP 8 --- Save Script

Save your clustering script as:

    cluster.py

Inside:

    face_cluster_project/

------------------------------------------------------------------------

## ✅ STEP 9 --- Run Clustering Engine

Command format:

``` bash
python cluster.py <start_accuracy> <end_accuracy> <iterations>
```

Example:

``` bash
python cluster.py 50 80 10
```

Meaning:

✔ Thresholds → 50 / 60 / 70 / 80\
✔ Iterations → 10 refinement passes

------------------------------------------------------------------------

## ✅ STEP 10 --- Internal Processing

For each accuracy:

Iteration 1 → cluster → merge\
Iteration 2 → refine → merge\
Iteration N → final refinement

------------------------------------------------------------------------

## ✅ STEP 11 --- Output Structure

### Temp Processing (Working Data)

temp_clusters/\
acc50_iter1/\
acc50_iter2/

✔ Scratch / debug data

------------------------------------------------------------------------

### Final Clusters (Clean Result)

output_clusters/\
final_groups_50/\
final_groups_60/\
final_groups_70/\
final_groups_80/

Each contains:

final_groups_XX/\
Ab12Cd/\
img1.jpg\
img4.jpg

✔ Clustered identities

------------------------------------------------------------------------

## ✅ STEP 12 --- Threshold Tuning Strategy

If clustering is:

✔ Too aggressive → Increase accuracy (80 → 85)\
✔ Too fragmented → Lower accuracy (80 → 75)

Typical FaceNet sweet spot:

0.75 -- 0.85

------------------------------------------------------------------------

# ✅ DONE
