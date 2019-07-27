#PROJECT
To develop a machine learning model which can classify gamma and hadron rays coming out of
charged cosmic particles.

# UNDERSTANDING THE PROJECT
In this project we make use of the dataset collected through the Cherenkov Telescope array(CTA).With its ability to view the highest 
energy processes in the Universe,CTA is able to detect the gamma rays about 10 million times more energetic than visible light.
Cherenkov gamma telescope observes high energy gamma rays, taking advantage of the radiation emitted by charged particles produced inside 
the electromagnetic showers initiated by the gammas, and developing in the atmosphere. This Cherenkov radiation  leaks through the 
atmosphere and gets recorded in the detector, allowing reconstruction of the shower parameters. The available information consists of 
pulses left by the incoming Cherenkov photons on the photomultiplier tubes, arranged in a plane, the camera. Depending on the energy of 
the primary gamma, a total of few hundreds to some 10000 Cherenkov photons get collected, in patterns (called the shower image), allowing 
to discriminate statistically those caused by primary gammas (signal) from the images of hadronic showers initiated by cosmic rays in the 
upper atmosphere (background).
Typically, the image of a shower after some pre-processing is an elongated cluster.A principal component analysis is performed in the camera plane, which results in a correlation axis and defines an ellipse.
The characteristic parameters of this ellipse (often called Hillas parameters) are among the image parameters that can be used for 
discrimination. The energy depositions are typically asymmetric along the major axis, and this asymmetry can also be used in discrimination.

###ATTRIBUTE INFORMATION
1. fLength: continuous # major axis of ellipse [mm]
2. fWidth: continuous # minor axis of ellipse [mm]
3. fSize: continuous # 10-log of sum of content of all pixels [in #phot]
4. fConc: continuous # ratio of sum of two highest pixels over fSize [ratio]
5. fConc1: continuous # ratio of highest pixel over fSize [ratio]
6. fAsym: continuous # distance from highest pixel to center, projected onto major axis [mm]
7. fM3Long: continuous # 3rd root of third moment along major axis [mm]
8. fM3Trans: continuous # 3rd root of third moment along minor axis [mm]
9. fAlpha: continuous # angle of major axis with vector to origin [deg]
10. fDist: continuous # distance from origin to center of ellipse [mm]
11. class: g,h # gamma (signal), hadron (background) 

#OVERVIEW
Basically, we use the MAGIC TELESCOPE dataset to build a machine learning classifier model which will train on the dataset and will be able to classify whether or not some energy is either Gamma Radiatin or Hadron Radiation.
We will train the model using the application of different algorithms(Logistics Regression, Neural Network,KNN,SVM,Random Forests) and will analyse the accuracy obtained using these algorithms.

#DEPENDENCIES
* Numpy
* Pandas
* scikit-learn
* Matplotlib





