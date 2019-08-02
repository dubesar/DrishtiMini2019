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
The main aim of Data Cleaning is to identify and remove errors & duplicate data, in order to create a reliable dataset. This improves the quality of the training data for analytics and enables accurate decision-making.Needless to say, data cleansing is a time-consuming process and most data scientists spend an enormous amount of time in enhancing the quality of the data. However, there are various methods to identify and classify data for data cleansing.

Data Cleaning consists of two basic stages, first is error identification and second is error solving. For any data cleaning activity, the first step is to identify the anomalies.

There are various method to clean data but in our project we are using pandas in python to clean data.
There are some kind we faced while we are cleaning the data.They are:
1.missing values
2.numeric strings
3.alphabetical strings
4.special characters..etc

The data we recived containained missing values along with strings randomly inserted into columns that one would expect to have contained only numeric values. So it was essential 

for us to first clean the data before we could run any model on it. We first located the said stings in the data by running 2 seperate 'for' loops through the features and the target columns respectively. 

If a row contained any string in any of the 'features' columns, we droped it. We found positions of the said rows by running a loop that individually checked 

each element of a column and stored the outliers. For the target column however we can't use the same code as it already consisted of strings. So our approch was to find any string 

that didn't match either 'h' or 'g', as these were what the values were supposed to be. The outliers found in this loop were added to the previous list, finally giving us a list containing all outliers. All the rows stored in this list were latter dropped.

For missing values first store the index of the data in one array and us fillna function in pandas to fill the the empty spaces with mean/median/mode of the respectively colomns or drop the row of missing value using dropna function.

to remove alphabetical strings we can use is.alpha() or is.alnum() and store the data index.drop the index values.we can do in another way by defineing a function it type cast r]the number if it changes then it return True if its show value error the return False and drop the false rows.
 
Conclusion:
Data Cleaning is a critical process for the success of any machine learning function. For most machine learning projects, about 80 percent of the effort is spent on data cleaning. 


# ALGORITHMS USED TO MODEL THE DATA
We implimented logistic regresstion and neural network algorithms on the dataset.

# ACCURACY ACHIEVED ON THE APPLICATION OF THE ALGORITHMS
The simple classification accuracy is not meaningful for this data, since classifying a background event as signal is worse than classifying a signal event as background. where the probability of accepting a background event as signal is below one of the following thresholds: 0.01, 0.02, 0.05, 0.1, 0.2 depending on the required quality of the sample of the accepted events for different experiments.

Logistic-regression:\
for 0.01 Accuracy from scratch: 0.357906494335407673\
for 0.02 Accuracy from scratch: 0.3636508696345939\
for 0.05 Accuracy from scratch: 0.3965214616243817\
for 0.1 Accuracy from scratch: 0.4573161002074358\
for 0.2 Accuracy from scratch: 0.565980532950375\
for 0.5 Accuracy from scratch: 0.781554172650391
# ISSUES FACED DURING:
* OBTAINING THE DATA
* DATA CLEANING
* APPLICATION OF THE ALGORITHM
* MISCELLANEOUS

# CONCLUSION
