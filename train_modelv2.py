# USAGE
# python train_model.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle --le output/le.pickle

# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle, json
import numpy as np

# load the face embeddings
print("[INFO] loading face embeddings...")
# data = pickle.loads(open("output/embeddings.pickle", "rb").read())
data = json.load(open("output/embeddingsv2.json", "r"))

# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()

labels = le.fit_transform(data["names"])
# print(labels)
# exit()

# train the model used to accept the 128-d embeddings of the face and
# then produce the actual face recognition
print("[INFO] training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
embeddings = []
for embedding in data["embeddings"]:
    embeddings.append(np.asarray(embedding, dtype=np.float32))
# print(embeddings)
recognizer.fit(embeddings, labels)

# write the actual face recognition model to disk
f = open("output/recognizerv2.pickle", "wb")
f.write(pickle.dumps(recognizer))
f.close()

# write the label encoder to disk
f = open("output/lev2.pickle", "wb")
f.write(pickle.dumps(le))
f.close()