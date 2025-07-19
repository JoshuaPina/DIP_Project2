# DIP Side Project for Machine Learning

## Overview

The purpose of this script was to enable my ML team to test our models with a cleaner dataset, and in turn optimize the time spent.

While creating the script and learning + testing multiple libraries, I realized that this script may be of use to others. Please feel free to copy the script and optimize it. 

If there are any changes that you would like to recommend, I am open to making the script as short and concise as possible! 

**AI Disclaimer:** I did utilize AI to check and optimize certain parts of the script to reduce space&time complexity, as that is very important when dealing with increasingly large datasets. The updates section explicitly states areas of AI utilization.

If you are an outside source, I ask that you do mention/reference where you got the script from. Thanks for reading, updates and creator info can be found below. 

## General Information 

The purpose of this script is related to my group Machine Learning project. We obtained a 6GB dataset of CT Scans, due to the number of images, and the number of folders these images were separated into, certain scripts took a while to get through everything...especially when utilizing things like OneDrive for storage. 

To help reduce the time spent, I created a script (and then used some AI assistance for optimization) to do the following:

- Checked that the ID attached to Image Folders had a matching row in our CSV. If it did not, then this image folder was removed for ML consideration, as it would be devoid of labels.
- Checked that all CSV rows had a matching Image Folder
- Resize images from 512x512 to 256x256 (or 128x128) with a configuration at the top of the script to easily swap when running the script.
- Convert images from png to jpg, as jpg takes up less storage.
- A configuration was added to the top of the script to toggle between different output resolutions depending on how much detail is needed.
- Copy and output a new folder that contains all of the proccessed images and csv data.
  
***This in turn will allow me to test different ML models on the dataset in a much quicker fashion.***

## Updates

### ->Future Update<-

- This section will eventually be converted into a true changelog
- The next update will utilize Streamlit for basic UI/UX

### July 19, 2025

This update:
- Adds viz to show code improvement.
- Adds viz to show output.
- AI was used to find these edge cases and pain-points that I did not originally consider:
  - Someone else using the code may not have png, input file is now configurable.
  - Vastly improving my timer.
- I noted that the new timer gives me UTC, which led to incorrect time for my usecase. 
  - I manually added timezone to fix this. 
    
### July 18, 2025

This update:
- Adds a true timer to measure how long the process takes.
- Adds batching to further increase script speed.
- Adds configuration for batch and jpeg/jpg quality.
- Converts the output to a table.
- AI was used to find these edge cases and pain-points that I did not originally consider:
  - Additional error-handling (AI assistance helped my find additional pain-points).
  - Further conversion options, for when RGB is not needed.
  - Configuration logging, as this project was how I learned config.

## Piña, The Creator

Joshua Piña
~ A Theoretical Master of None ~
Data Science - Senior, Georgia State University  
Background in Python, Applied Probability & Statistics, Data Analytics, Big Data Programming, DIP, and Machine Learning. 
Experienced Program Manager in government contracting and former U.S. Army Combat Medic.

[Github](https://www.github.com/joshuadpina) | [LinkedIn](https://www.linkedin.com/ln/joshuapina) | [Kaggle](https://www.kaggle.com/joshuapina)
