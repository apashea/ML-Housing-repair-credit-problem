# ML-Housing-repair-credit-problem
(*Note: For those unfamiliar with R or github, click on the 'Fig' image files above to see visualizations and table output from R. 'apashea_housing_credit' contains the R script from which they were produced as well as line-by-line comments describing and commenting on what is happening in each section.*)

This project uses machine-learning methods, including model comparisons and metrics, for binary classification to help a fictional public institution better allocate their resources. The data and parts of the code are based on the assignment (no answer key provided) in Chapter 6 of Ken Steif's excellent textbook *Public Policy Analytics: Code & Context for Data Science in Government*, 2021 (https://urbanspatial.github.io/PublicPolicyAnalytics/). My project does not restrict itself to the methods taught in the chapter, however; I use other machine-learning algorithms and sampling methods in addition to logistic regression as found in the chapter's example churn problem. This project demonstrates uses of the following in R:
- Data cleaning and pre-processing, including one-hot encoding and normalization
- Feature engineering and recursive feature elimination
- Logit models
- (Repeated) cross-validation
- Random forest models
    - Repeated cross-validation
    - Grid search
- Down-/up-sampling and other methods for tackling imbalanced data
- Ensemble model training and comparison (Random forest, glmnet, Support Vector Machine)
- Confusion matrices (and how to interpret them for use in cost-benefit analyses)
- ROC curves / Area under the curve calculation and visualization
- Model threshold optimization and cost-benefit optimization with visualization
- Cost-benefit analyses and tables for comparing total social benefit and campaign losses between using the final chosen model versus not using a model


This project imagines a city department who runs 'marketing' campaigns by allocating resources for contacting and working with eligible homeowners who might accept a home repair tax credit. In this scenario, previous campaigns have unfortunately gone poorly due to no selectivity over which homeowners to contact: only around 11% ultimately accepted the credit last period, with an assumed 75% of those who accepted being homeowners who already planned to apply for the credit regardless of the campaign, and so many marketing campaign resources were counted as losses. Meanwhile, the department has collected data on eligible homeowners prior to contacting them and hires its first data science team to see what can be done to improve resource allocation.

This is treated as a binary classification problem of predicting if someone will or will not accept the credit so that the department can exercise more discretion in whom it contacts, and thus in whom it allocates resources that otherwise might be lost. The project tries and evaluates a small variety of models using different techniques and sampling methods to find one better suited for classifying this data. At the end, the project undertakes a cost-benefit analysis with visualization to help the department understand how using this model can help them improve their resource allocation.

Future plans/notes for improvement: 
The final chosen model in this project, a random forest using repeated cross-validation and down-sampling, could be improved by being combined with grid search and higher k-folds and repetitions of cross-validation. While the latter were kept small for the sake of quicker computing, the model (and others tried in the script) could easily be improved with methods like these just mentioned.
