# SpaceX Launch Prediction

## Overview
SpaceX advertises Falcon 9 rocket launches on its website with a cost of 62 million dollars; other providers cost upward of 165 million dollars each. Much of the savings is because SpaceX can reuse the first stage of the rocket. This project aims to create a machine learning model to predict whether the first stage of the Falcon 9 rocket will successfully land, which can significantly impact the overall cost of a launch.

## Features

-   **Data Preprocessing:** Cleaning and preparing the initial data for model training.
-   **Feature Engineering:** Creating new features from existing data to improve model performance.
-   **Model Training:** Using various machine learning algorithms (such as Logistic Regression, SVM, Decision Tree, and KNN) to create predictive models.
-   **Model Evaluation:** Evaluating model performance using various metrics (such as Accuracy, Precision, Recall, and Confusion Matrix).
-   **Model Comparison:** Comparing the performance of different models to select the best one.
-   **Visualization:** Displaying results using charts and images.

## Algorithms

The following machine learning algorithms are implemented and compared in this project:

-   **K-Nearest Neighbors (KNN):** A non-parametric algorithm that classifies data based on the majority class among its k nearest neighbors.
-   **Decision Tree:** A tree-like model that makes decisions based on feature values.
-   **Support Vector Machine (SVM):** A powerful algorithm that finds the optimal hyperplane to separate data into different classes.
-   **Logistic Regression:** A linear model that predicts the probability of a binary outcome.


## Dependencies

-   **Python:** The primary programming language for the project.
-   **Pandas:** For data management and analysis.
-   **Matplotlib:** For creating charts and images.
-   **Seaborn:** For creating visually appealing statistical graphs.
-   **Scikit-learn:** For training and evaluating machine learning models.


To install the required libraries, use the following command:

```bash
pip install pandas matplotlib seaborn scikit-learn