# Cloud Segmentation with Unet

This repository contains my coursework as 2nd year BSc in HSE. The objective was to create an app for cloud segmentation using 7 bands of Landsat 8 satellite. The proposed solution takes as input first 7 bands from the Landsat 8 and builds a segmentatino mask, using a bland of Unet and DeconvNet architectures.

[]!(images/arch.png)



## Instructions
1. Create "input" folder in the main directory and place input files into the folder.
2. If necessary, install packages in `art/requirements.txt`.
3. Run `art/FinalSolution.ipynb` to generate predictions for the test set (Our solution was developed for track A only)
    - It takes 5-6 hours to predict for 300 satellites.
4. You could check our other notebooks as well, these were our ideas that did not succeed.
