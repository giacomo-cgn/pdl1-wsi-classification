# pdl1-wsi-classification
Code used for classifying Whole Slide Images (WSI) of tumors to decide if they are positive or negative to the brown biomarker PD-L1 (brown staining on the tumor area), based on a WSI level label. 

The model developed is divided in 2 independent parts, Region of Interest (ROI) identification and PD-L1 classification. The notebooks contain portions of code both for training the models and for testing it.

Due to the large dimension of WSIs and the presence of only WSI-level labeling, 2 different feature extraction are used: histogram of the pixel distance from brown of a WSI and the aggregation (through clustering and average) of autoencoder-obtained tile embeddings.

2 datasets of WSIs of breast tumors stained with the VENTANA PD-L1 (SP142)  are used for this project. The first dataset consists of 39 WSIs, 20 positive and 19 negative to PD-L1 staining.
The second dataset is composed of 25 WSIs, 21 negative and 4 positive, with similar characteristics to the first dataset (it is referred in the code as external or ext dataset). 
In both datasets, each slide has a full resolution in the range of approximately 50.000 to 150.000 pixels both in width and height.


DISCLAIMER:
This code is intended to show the modus operandi behind the draft paper "PD-L1 classification of weakly-labeled whole slide
images of breast cancer" by Cignoni et al.; it is not intended to be runnable code as it is. 
This is just a collection of the algorithms used for said article, files could have been executed in different environments, this implies that files paths and dependencies may not be correct. Final results may also differ slightly from the ones indicated in the paper because code re-runs may have occurred.
