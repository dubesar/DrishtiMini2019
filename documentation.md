# PROJECT 
To develop machine learning models that can classify cosmic particles into gamma and hadron rays.

# OVERVIEW

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

Our objective is to train different machine learing models on this data and try to find the best model for hadron-gamma clasiffication.
 
### ATTRIBUTE INFORMATION

The following are the different attributes of the data:

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

# DATA CLEANING 
The data we recived containained missing values along with strings randomly inserted into columns that one would expect to have contained only numeric values. So it was essential 
for us to first clean the data before we could run any model on it. We first located the said stings in the data by running 2 seperate 'for' loops through the features and the target 
columns respectively. If a row contained any string in any of the 'features' columns, we droped it. We found positions of the said rows by running a loop that individually checked 
each element of a column and stored the outliers. For the target column however we can't use the same code as it already consisted of strings. So our approch was to find any string 
that didn't match either 'h' or 'g', as these were what the values were supposed to be. The outliers found in this loop were added to the previous list, finally giving us a list 
containing all outliers. All the rows stored in this list were latter dropped.

# ALGORITHMS USED TO MODEL THE DATA
We implimented logistic regresstion and neural network algorithms on the dataset.

# ACCURACY ACHIEVED ON THE APPLICATION OF THE ALGORITHMS

# ISSUES FACED DURING:
* OBTAINING THE DATA
* DATA CLEANING
* APPLICATION OF THE ALGORITHM
* MISCELLANEOUS

# CONCLUSION
