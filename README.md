# Multi-Layer Perceptron Generation from NBA Box 2019 Data
## Application of First-Classification-Then-Regression Analysis Coupled with Stochastic Gradient Boosting
### Steps to success:
- Create an estimate MLP classifier using MLP_E-Classifier.py
- Create a rank MLP classifier using MLP_R-Classifier.py
- Create an overall MLP regressor using MLP_Creator.py
- Apply the overall MLP regressor to holdout data using Final_Applier.py
### Miscellanies:
- MLP_Tester.py: Reads in the pickled MLP and testing sets from MLP_Creator, determines MAPE, and plots the results
- Feature_Selector.py: Determine which features are best for use when creating MLP
- Data_Viewer.py: View a 2D line graph comparing one feature to the engagements
- Good_Words.py: Find the most commonly-used words in captions of pictures in the top 25% of engagements
### Data:
- training_set.csv: Training data with engagements filled in (7766 lines)
- holdout_set.csv: Testing data with engagements empty (1000 lines)
