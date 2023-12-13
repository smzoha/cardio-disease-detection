# Cardiovascular Disease Detection

This project was completed as a requirement of an Advanced Artificial Intelligence
coursework, for the semester of Fall 2023 at BRAC University.

The project is intended to compare Naive Bayes Classifiers (Multinomial and Gaussian)
against other common Machine Learning algorithms, such as Logistic Regression, Random
Forest, and Support Vector Machine (SVM). The Gaussian Naive Bayes & Machine Learning models
are then analysed with LIME (Local Interpretable Model-agnostic Explanations) to perform model explanation (XAI).

The dataset used in this project was obtained for Kaggle and contains demographic data
of individuals, as well as medical data, for around 70,000 instances.

The dataset can be found here: [Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)

The dataset contained 11 features and one target variable - a boolean value that indicates
if the patient has Cardiovascular Disease or not.

The project can be divided into three parts - data analysis & visualization, cleanup, and
model training.

There are five Python scripts in the project, each forming different segments of the project:

* **data-fetch.py:** Fetch the dataset directly from Kaggle and store in the `./data` directory
* **data-analysis.py:** Print summary of dataset and visualize every attribute
  * For categorical data, bar chart was drawn; for rest, histogram was the chosen approach
* **data-cleanup.py:** Involves data cleanup, engineering, and storing the updated values in a separate file
* **train-nb.py:** Train MultinomialNaiveBayes & GaussianNaiveBayes models, and obtain training/testing metrics, along with analysing the latter with LIME
* **train-ml.py:** Same as above with LogisticRegression, RandomForestClassifier and LinearSVC, with the latter being analysed with LIME

Prerequisite:
* Python 3
* Pip
* (Option) Kaggle Account - for running script to fetch dataset
* (Option) Jupyter Notebook - to run the notebook version of the project

Instruction:
1. If you wish to download the dataset using the provided script, a Kaggle account is required. Please create an account and download the API key, before placing it under `~/.kaggle` directory
2. Run the scripts one by one in the order mentioned above

Libraries used:
* Pandas
* Numpy
* Sci-kit Learn
* Matplotlib
* Kaggle
* [mRMR](https://github.com/smazzanti/mrmr)
* [LIME](https://github.com/marcotcr/lime)
