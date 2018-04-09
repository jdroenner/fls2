# -*- coding: utf-8 -*-
'''
Created on Sep 30, 2015

@author: sebastian
'''
from numpy import float32, sqrt

from sklearn.cluster.k_means_ import KMeans
from sklearn.ensemble.forest import RandomForestRegressor, RandomForestClassifier

import numpy as np
from util.read_from_file import loadSingleBandGeoTiff
from util.write_to_file import writeDataToGeoTiff


# Methoden zum Berechnen versch. Regressionsmodelle auf Basis von Machine Learning Algorithmen:
# channels_training: Input-Datensatz mit allen unabh. Variablen zum Trainieren (z.B. Meteosatkanäle)
# channels_testing:  Input-Datensatz mit allen unabh. Variablen zum Validieren (z.B. Meteosatkanäle)
# target_training:   Werte der Zielvariable fürs Trainieren (z.B. cloudtoprcr)
# target_testing:    Werte der Zielvariable fürs Validieren (z.B. cloudtoprcr)
# Random Forest:
def calcRandomForest(channels_training, channels_testing, target_training, target_testing):
    clf = RandomForestRegressor(n_estimators=500,max_features=len(channels_training[0]))
    clf = clf.fit(channels_training, target_training)
    predictions = clf.predict(channels_testing)
    comp = [predictions,target_testing]
    return clf, comp

def calcRandomForestClassifier(channels_training, channels_testing, target_training, target_testing):
    clf = RandomForestClassifier(n_estimators=500,max_features=int(sqrt(len(channels_training[0]))))
    clf = clf.fit(channels_training, target_training)
    predictions = clf.predict(channels_testing)
    comp = [predictions,target_testing,channels_testing]
    return clf, comp

data_fog = loadSingleBandGeoTiff("/home/sebastian/Documents/test/manuelle_trainingsgebiete/fog.tif").flatten()
data_nofog = loadSingleBandGeoTiff("/home/sebastian/Documents/test/manuelle_trainingsgebiete/no_fog.tif").flatten()

data01 = loadSingleBandGeoTiff("/home/sebastian/Documents/test/manuelle_trainingsgebiete/20130220_0445_01.tif").flatten().astype(float32)
data02 = loadSingleBandGeoTiff("/home/sebastian/Documents/test/manuelle_trainingsgebiete/20130220_0445_02.tif").flatten().astype(float32)
data03 = loadSingleBandGeoTiff("/home/sebastian/Documents/test/manuelle_trainingsgebiete/20130220_0445_03.tif").flatten().astype(float32)
data04 = loadSingleBandGeoTiff("/home/sebastian/Documents/test/manuelle_trainingsgebiete/20130220_0445_04.tif").flatten().astype(float32)
data05 = loadSingleBandGeoTiff("/home/sebastian/Documents/test/manuelle_trainingsgebiete/20130220_0445_05.tif").flatten().astype(float32)
data06 = loadSingleBandGeoTiff("/home/sebastian/Documents/test/manuelle_trainingsgebiete/20130220_0445_06.tif").flatten().astype(float32)
data07 = loadSingleBandGeoTiff("/home/sebastian/Documents/test/manuelle_trainingsgebiete/20130220_0445_07.tif").flatten().astype(float32)
data08 = loadSingleBandGeoTiff("/home/sebastian/Documents/test/manuelle_trainingsgebiete/20130220_0445_08.tif").flatten().astype(float32)
data09 = loadSingleBandGeoTiff("/home/sebastian/Documents/test/manuelle_trainingsgebiete/20130220_0445_09.tif").flatten().astype(float32)
data10 = loadSingleBandGeoTiff("/home/sebastian/Documents/test/manuelle_trainingsgebiete/20130220_0445_10.tif").flatten().astype(float32)
data11 = loadSingleBandGeoTiff("/home/sebastian/Documents/test/manuelle_trainingsgebiete/20130220_0445_11.tif").flatten().astype(float32)
data12 = loadSingleBandGeoTiff("/home/sebastian/Documents/test/manuelle_trainingsgebiete/20130220_0445_sza.tif").flatten().astype(float32)

channels_training = []
target_training = []

for i in range(len(data_fog)):
    if data_fog[i]>0.:
        if not np.any(np.isnan([data01[i],data02[i],data03[i],data04[i],data05[i],data06[i],data07[i],data08[i],data09[i],data10[i],data11[i],data12[i]])):
            channels_training.append([data01[i],data02[i],data03[i],data04[i],data05[i],data06[i],data07[i],data08[i],data09[i],data10[i],data11[i],data12[i]])
            target_training.append(1.)

for i in range(len(data_nofog)):
    if data_nofog[i]>0.:
        if not np.any(np.isnan([data01[i],data02[i],data03[i],data04[i],data05[i],data06[i],data07[i],data08[i],data09[i],data10[i],data11[i],data12[i]])):
            channels_training.append([data01[i],data02[i],data03[i],data04[i],data05[i],data06[i],data07[i],data08[i],data09[i],data10[i],data11[i],data12[i]])
            target_training.append(0.)

channels_testing = []
target_testing = []

for i in range(len(data_fog)):
    if not np.any(np.isnan([data01[i],data02[i],data03[i],data04[i],data05[i],data06[i],data07[i],data08[i],data09[i],data10[i],data11[i],data12[i]])):
        channels_testing.append([data01[i],data02[i],data03[i],data04[i],data05[i],data06[i],data07[i],data08[i],data09[i],data10[i],data11[i],data12[i]])
        target_testing.append(5.)
    else:
        channels_testing.append([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
        target_testing.append(5.)

channels_training = np.asarray(channels_training,dtype=float32)
target_training = np.asarray(target_training,dtype=float32)
channels_testing = np.asarray(channels_testing,dtype=float32)
target_testing = np.asarray(target_testing,dtype=float32)

print len(channels_testing)

print "los gehts..."

#samples = np.transpose(([channels_testing[:,3],channels_testing[:,8],channels_testing[:,9]]))
#kmeans = KMeans(n_clusters=12)
#kmeans.fit(samples)
#centroids = kmeans.cluster_centers_
#labels = kmeans.labels_
#ergebnis = np.reshape(labels,(510,767))
#writeDataToGeoTiff(ergebnis,path="/home/sebastian/Documents/test/manuelle_trainingsgebiete/ergebnis_unsupervised_6_classes.tif")


clf, comp = calcRandomForestClassifier(channels_training, channels_testing, target_training, target_testing)
ergebnis = np.reshape(comp[0],(510,767))
writeDataToGeoTiff(ergebnis,path="/home/sebastian/Documents/test/manuelle_trainingsgebiete/ergebnis_2_sza.tif")
