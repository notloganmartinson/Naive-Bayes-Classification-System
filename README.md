# Project 1: Naïve Bayes Classification and Performance Evaluation
Built entirely with AI coding
## Description
This program is a custom implementation of a Naïve Bayes classification system. It is built entirely from scratch without the use of external machine learning libraries. The classifier calculates probabilities based on training data and smooths the data by adding one to all counts to prevent zero-probability errors. 

The system features an interactive menu that allows users to perform tasks repeatedly and in any reasonable order. The user can:
* Train the model using a metadata file and a training data file through a single menu item.
* Classify new datasets (with or without labels) and output the predicted classifications in the exact format as the training data.
* Evaluate model accuracy on a testing dataset, generating a confusion matrix alongside precision, recall, and F-measure statistics.
* Perform k-fold cross-validation to assess the model's reliability, printing the accuracy of each fold and the overall average accuracy.

## Compilation Instructions
This project is written in standard Python 3 and does not require a traditional compilation step. It is designed to run seamlessly on Linux environments, including WSL and the ISU terminal server.

**Requirements:**
* Python 3.x installed on your system.
* Standard Python libraries (no external machine learning libraries are used).

**To execute the program:**
1. Open your terminal or connect to the ISU terminal server.
2. Navigate to the directory containing the project files.
3. Run the script using the following command:
   python3 naive_bayes.py

## Usage Instructions
Upon running the program, you will be presented with a main menu. You can perform these operations in any reasonable order and repeat them as often as desired without needing to restart the program. 

### Menu Options
* **1. Train the model:** The system will ask for the metadata file (e.g., car.meta) and the training data file (e.g., car.train). There is no separate menu item for the metadata file. You must complete this step before classifying or evaluating data. You can train on new files as often as desired.
* **2. Classify a dataset:** The system will ask for an input file and an output file name. It will read the input instances, ignore any existing labels, compute the classifications, and save the results to your specified output file. You can classify as many different files as desired before retraining.
* **3. Test model accuracy & print confusion matrix:** The system will prompt you for a test data file (e.g., car.test). It will compare the model's predictions against the actual labels, then print the accuracy, confusion matrix, precision, recall, and F-measure to the screen.
* **4. K-fold cross-validation:** This operates independently of the main training/testing flow. The system will ask for the metadata file, a data file, and an integer k. It will perform k-fold cross-validation and print the accuracy of each fold, followed by the average accuracy.
* **5. Quit:** Safely exits the program.

**Important Notes:**
* Ensure your metadata file lists attributes and their possible values, ending with the classification.
* Data files must contain comma-separated examples that match the order of the metadata file.
