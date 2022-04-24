# pyhton script to cluster images with K-Means algorithm
# it is a modified version of https://github.com/rohanbaisantry/image-clustering/blob/master/image_clustering.py

import random, cv2, os, sys, shutil
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import keras
import tensorflow as tf

class image_clustering:

	def __init__(self, folder_path="data", n_clusters=10, max_examples=None, use_imagenets=False, use_pca=False):
		paths = os.listdir(folder_path)
		self.max_examples = len(paths)
		
		self.n_clusters = n_clusters
		self.folder_path = folder_path
		random.shuffle(paths)
		self.image_paths = paths[:self.max_examples]
		self.use_imagenets = use_imagenets
		self.use_pca = use_pca
		del paths 
        
		os.makedirs("KMeans_output")
        
		for i in range(self.n_clusters):
			os.makedirs("KMeans_output/cluster" + str(i))
		print("\n Object of class \"image_clustering\" has been initialized.")

	def load_images(self):
		self.images = []
		for image in self.image_paths:
			self.images.append(cv2.cvtColor(cv2.resize(cv2.imread(self.folder_path + "/" + image), (224,224)), cv2.COLOR_BGR2RGB))
		self.images = np.float32(self.images).reshape(len(self.images), -1)
		self.images /= 255
		print("\n " + str(self.max_examples) + " images from the \"" + self.folder_path + "\" folder have been loaded in a random order.")

	def clustering(self):
		model = KMeans(n_clusters=self.n_clusters, random_state=728)
		model.fit(self.images)
		predictions = model.predict(self.images)
		print(predictions)
		for i in range(self.max_examples):
			shutil.copy2(os.path.join(self.folder_path,self.image_paths[i]), os.path.join("KMeans_output", 'cluster'+str(predictions[i])))
		print("\n Clustering complete! \n\n Clusters and the respective images are stored in the \"KMeans_output\" folder.")

