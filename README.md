# Digital Sommelier: Classifying Wines by Common Varieties

**CAP4770 Introduction to Data Science**  
Spring 2025  
Group 10: Julian Ubico, Renzo Vallejos, Zack Whalen, Athena Wrenn

## Overview

Digital Sommelier is a machine learning project that predicts a wine’s variety based solely on its written description. Using a dataset of 130,000 professional wine reviews from Kaggle (WineEnthusiast, 2017), we focused on the 20 most common wine varieties to address class imbalance and improve accuracy.

## Approach

- **Data**: 130k+ wine reviews, each with description, variety, and other attributes.
- **Preprocessing**: Filtered to top 20 varieties, removed uninformative words, and used TF-IDF vectorization.
- **Models Tested**: Multinomial Naive Bayes, Decision Tree, Random Forest, and XGBoost.
- **Final Model**: XGBoost, chosen for its accuracy and efficiency.
- **Class Imbalance**: Addressed using sample weighting to ensure fair training across all varieties.

## Usage

1. Ensure `winemag-data-130k-v2.csv` is in the project directory.
2. Run `improved_model.py` to train or use the model.
3. Enter a wine description to receive a predicted variety.

## References

- [Wine Reviews Dataset (Kaggle)](https://www.kaggle.com/datasets/zynicide/wine-reviews)
