from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import neighbors
import glob
import cv2
import os
import numpy as np
import re

'''
Class to run inference and train a GEI-based (Gait Energy Image) Gait recogniser
'''
class GaitRecogniser:
    def __init__(self, database_path='data/gei/GaitRecogniser'):
        self.database_path=database_path
        self.pca = PCA(n_components=10)
        self.lda = LinearDiscriminantAnalysis(n_components=20)
        self.knn = neighbors.KNeighborsClassifier(n_neighbors=3)

    def train(self):
        # load files from the database
        files = glob.glob('{}/*.png'.format(self.database_path))
        train_imgs = np.array([self.get_image(f) for f in files])
        train_labels = np.array([self.get_label(f) for f in files])

        # training
        train_reduced = self.pca.fit_transform(train_imgs)
        train_lda_reduced = self.lda.fit_transform(train_reduced, train_labels)
        self.knn.fit(train_lda_reduced, train_labels)

    def predict(self, gei_img):
        pca_reduced = self.pca.transform([gei_img])
        lda_transformed = self.lda.transform(pca_reduced)
        predictions = self.knn.predict(lda_transformed)
        return predictions[0]

    def predict_from_file(self, gei_img_path, debug=False):
        gei_img = self.get_image(gei_img_path)
        if debug:
            cv2.imshow('DEBUG', np.reshape(gei_img, (271, 170)))
            cv2.waitKey(0)
        return self.predict(gei_img)

    def get_image(self, file_path):
        # binary mask so just need 1 channel
        img = cv2.imread(file_path)[:,:,0]
        return img.flatten()

    def get_label(self, file_path):
        # file_path assumed to have the following format: /path/to/database/{label}_{id}.png
        file_name = file_path.split(os.path.sep)[-1]
        p = re.compile('(.*)\_\d+.png')
        label = p.match(file_name).group(1)
        return label

if __name__ == '__main__':
    # load up the model
    recogniser = GaitRecogniser()
    recogniser.train()

    # predict on test gei
    pred = recogniser.predict_from_file('test_gei.png')
    print(pred)