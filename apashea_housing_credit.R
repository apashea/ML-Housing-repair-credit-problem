library(tidyverse)
library(caret)
library(knitr)
library(pscl)
library(plotROC)
library(pROC)
library(scales)


# Importing our data
orig_data <- "https://raw.githubusercontent.com/urbanSpatial/Public-Policy-Analytics-Landing/master/DATA/Chapter6/housingSubsidy.csv" %>%
  read.csv(stringsAsFactors = TRUE)

housedf <- orig_data

# Fig - Data structure and summary
# Reviewing our variables, counts, summary information
str(orig_data)
summary(orig_data)

# Initial modifications
# Our taxLien variable has only one count of a 'yes' response, which will not be generalizable between test and training sets. 
# Because there is only one count of 'yes', it is safe enough to leave it out. The same goes for 'illiterate' under education.
housedf <- orig_data %>%
  mutate(y_num = as.factor(ifelse(y == "yes",1,0))) %>%
  select(y_num, y, everything(), -y_numeric, -X) %>%
  na.omit() %>%
  filter(taxLien != "yes") %>%
  filter(education != "illiterate") %>%
  mutate(taxLien = droplevels(taxLien), education = droplevels(education))
# Viewing the structure again to review the changes
str(housedf)
# Checking for missing values. There are none.
any(is.na(housedf))

# Viewing counts for who accepted the credit (451) and who did not (3666).
# We also see that our data is imbalanced, with less than 11% people who accepted the credit.
# This makes it more difficult to find a model which can accurately predict both
# 'yes' and 'no'. Because more data is available for the latter, a baseline
# model is expected to have much less accuracy in predicted 'yes'. Later in
# this script an assortment of typical best-practice methods will be used to
# adjust for this issue and gain higher accuracy for either category.
housedf %>% count(y)
(y_count <- housedf %>% count(y))
y_count[2,2] / (y_count[1,2] + y_count[2,2])

# Fig - continuous variables
# Visualizing accepting the credit versus our available continuous variables.
housedf %>%
  select(y, age, campaign, previous, unemploy_rate, cons.price.idx, cons.conf.idx, inflation_rate, spent_on_repairs) %>%
  gather(Variable, value, -y) %>%
  ggplot(aes(y, value, fill=y)) +
  geom_bar(position = "dodge", stat = "summary", fun = "mean") +
  facet_wrap(~Variable, scales = "free") +
  #   scale_fill_manual(values = palette2) +
  labs(x="Accepted credit", y="Mean",
       title = "Feature associations with likelihood",
       subtitle = "(Continuous outcomes)") +
  #    plotTheme() + 
  theme(legend.position = "none")

# Fig - categorical variables visualization 1
# Visualizing relationships between who accepted the credit and the low-level categorical variables.
housedf %>%
  select(y, marital, taxLien, mortgage, taxbill_in_phl, contact, poutcome) %>%
  gather(Variable, value, -y) %>%
  count(Variable, value, y) %>%
  ggplot(aes(value, n, fill = y)) +
  geom_bar(position = "dodge", stat = "identity") +
  facet_wrap(~Variable, scales = "free") +
  labs(x = "Accepted credit", y = "Count",
       title = "Feature associations with likelihood of accepting credit",
       subtitle = "") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))      # Ignore the warning; coercing our factors of different levels into character vectors here has no impact on visualization. 

# Fig - categorical variables visualization 2
# Visualizing relationships between who accepted the credit and the higher-level categorical variables.
housedf %>%
  select(y, job, education, month, day_of_week, pdays) %>%
  gather(Variable, value, -y) %>%
  count(Variable, value, y) %>%
  ggplot(aes(value, n, fill = y)) +
  geom_bar(position = "dodge", stat = "identity") +
  facet_wrap(~Variable, scales = "free") +
  labs(x = "Accepted credit: yes or no", y = "Count",
       title = "Feature associations with counts of accepting credit",
       subtitle = "") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


# Baseline model:
# Partitioning a 65/35 training/test split of our data.
library(caret)
set.seed(555)
trainIndex <- createDataPartition(housedf$y, p = 0.65,
                                  list = FALSE,
                                  times = 1)
dfTrain <- housedf[trainIndex,]
dfTest <- housedf[-trainIndex,]

summary(dfTrain)

# Building a logistic regression ('logit') model as an initial baseline model
# where all variables are included.
yreg1 <- glm(y ~ .,
             data = dfTrain %>% select(-y_num),
             family = "binomial" (link = "logit"))
summary(yreg1) 
# Viewing McFadden metric, which is one common metric for logit models.
pR2(yreg1)[4]
testProbs <- data.frame(Outcome = as.factor(dfTest$y_num),
                        Probs = predict(yreg1, dfTest, type = "response"))
testProbs <- testProbs %>% mutate(predOutcome = as.factor(ifelse(testProbs$Probs > 0.5, 1, 0)))

# Fig - baseline logit confusion matrix
# Producing a confusion matrix from our baseline logit.
# This model has an accuracy of around 89%, which is quite good, as well as a
# Specificity, i.e. true negative rate or its ability to predict when the homeowner
# will decline the credit, of around 97% meaning it is nearly perfect at predicting
# a 'no'. If the institution's main priority is to constrain the number of people
# it contacts in order to further save on its own resources, this would be very
# desireable. However, the model's Sensitivity, i.e. true positive rate or ability
# to predict when a client will accept the credit, is only around 27%. This model
# when left to a typical probability threshold of 50% for 'yes' or 'no' will be
# very conservative and frequently 
confusionMatrix(testProbs$predOutcome, testProbs$Outcome, positive = "1")

# Fig - baseline logit ROC curve 
# Receiver Operating Characteristic Curve for evaluating our model. The higher
# the area under the curve (AUC), the more the model has a capacity to make correct
# predictions. 
# The grey diagonal line here respresents a 'coinflip', or choosing
# at random, prediction. Despite our data being imbalanced and not yet transformed 
# to help with this issue, this model already strongly outperforms a 'coinflip'
# and would be beneficial when no other model is available.
ggplot(testProbs, aes(d = as.numeric(testProbs$Outcome), m = Probs)) +
  geom_roc(n.cuts = 50, labels = FALSE, colour = "#FE9900") +
  style_roc(theme = theme_grey) +
  geom_abline(slope = 1, intercept = 0, size = 1.5, color = 'grey') +
  labs(title = "ROC Curve - yreg1",
       subtitle = paste0("AUC =", auc(testProbs$Outcome, testProbs$Probs)))

# Area Under the Curve: 0.7401
auc(testProbs$Outcome, testProbs$Probs)


################## DATA PREPROCESSING ##########################################



# Checking for near-zero variance, creating a new data frame, and dropping variables with <= 5%.
# The variable 'campaign', representing the number of times an individual has been contacted
# during the current campaign, drastically falls as number of times increase, with the majority
# of individuals having only been contacted once or twice (1764 people and 1039 people, respectively).
# Those contacted more than often strongly decline, and every individual was contacted at least once.
# We could transform this variable via binning, but in the case the variance its variance is so
# small given 5% as the threshold that in this script it will be left out altogether.
(nzv95 <- nearZeroVar(housedf %>% select(-y_num), freqCut = 95/5,uniqueCut = 10))
colnames(housedf[nzv95])
summary(housedf)

housedfp <- housedf
housedfp <- housedfp %>% select(-nzv95, -y_num) %>% mutate(y = housedf$y)
str(housedfp)
summary(housedfp)

library(caret)
set.seed(555)
trainIndexp <- createDataPartition(housedfp$y, p = 0.65,
                                  list = FALSE,
                                  times = 1)
dfTrainp <- housedfp[trainIndexp,]
dfTestp <- housedfp[-trainIndexp,]

# One-hot encoding using package 'vtreat', which allows us to retain our same
# treatment plan for future data sets.
library(vtreat)
library(magrittr)
outcome <- "y"
(vars <- colnames(housedfp %>% select(-y)))
treatplan <- designTreatmentsZ(housedfp, vars)
(scoreFrame <- treatplan %>%
    use_series(scoreFrame) %>%
    select(varName, origName, code))
(newvars <- scoreFrame %>%
    filter(code %in% c("clean", "lev")) %>%
    use_series(varName))
# New training data
dfTrainpx <- prepare(treatplan, dfTrainp, varRestriction = newvars)
dfTrainpx <- dfTrainpx %>% mutate(y = dfTrainp$y) %>% select(y, everything())
# New testing data
dfTestpx <- prepare(treatplan, dfTestp, varRestriction = newvars)
dfTestpx <- dfTestpx %>% mutate(y = dfTestp$y) %>% select(y, everything())
# New full dataframe
housedfpx <- rbind(dfTrainpx, dfTestpx)
# Normalizing predictors.
rangeModel <- preProcess(dfTrainpx, method = "range")
dfTrainpx <- predict(rangeModel, newdata = dfTrainpx)
rangeModel2 <- preProcess(dfTestpx, method = "range")
dfTestpx <- predict(rangeModel2, newdata = dfTestpx)

# Recursive feature elimination
# RFE can be used to help determine which variables are most useful in predicting
# our data while eliminating those with less significance.
# While this script will not heed this section's results due to time constraint
# and better practice of keeping as much of our data as possible when dealing
# with this amount of observations, this script could be run again in the future
# if more data were collected or if a different model and further tuning were desired.
subsets <- c(1:5, 10, 15, 20)
set.seed(555)
rfeCtrl <- rfeControl(functions = rfFuncs,
                      method = "cv",
                      number = 5,
                      verbose = TRUE)
rfProfile <- rfe(x = dfTrainpx[,2:ncol(dfTrainpx)],
                 y = dfTrainpx$y,
                 sizes = subsets,
                 rfeControl = rfeCtrl)
# Top 5 variables: spent_on_repairs, inflation_rate, pdays, unemploy_rate, poutcome_lev_x_success
rfProfile

# Reviewing available algorithms from the 'caret' package
names(getModelInfo())

# Training a baseline random forest for variable importance
set.seed(555)
rfCtrl <- trainControl(number = 5, verboseIter = TRUE)
rf <- train(y ~ ., data = dfTrainpx, method = "rf", trControl = rfCtrl)
(varimp_RF <- varImp(rf))

# Fig - baseline random forest variable importance
# Plotting variable importance
plot(varimp_RF, main = "Credit Acceptance Variable Importance (Random Forest)")
# Predictions from baseline random forest
fitted <- predict(rf, dfTestpx)
# Confusion matrix and area under the curve
# Sensitivity = 0.15287
# Specificity = 0.99143
# AUC         = 0.7235
# The baseline rf is a bit worse than our baseline logit. Though the false negative rate
# is nearly perfect, its true positive rate is more or less useless.
confusionMatrix(dfTestpx$y, data = fitted, mode = "everything", positive = "yes")
rfProbs <- data.frame(Outcome = dfTestpx$y,
                        Probs = predict(rf, dfTestpx, type = "prob"))
rfProbs <- rfProbs %>% mutate(Probs = Probs.yes) %>% select(Outcome, Probs)
rfProbs <- rfProbs %>% mutate(predOutcome = as.factor(ifelse(rfProbs$Probs > 0.5, 1, 0)))
auc(rfProbs$Outcome, rfProbs$Probs)

# Constructing a more complex random forest
# The 'caret' package allows us to specify the aim of our model and adjust it
# accordingly. Here 'twoClassSummary' adjusts a random forest model for the purpose
# of binary classification, in our case 'yes' or 'no' for accepting the credit.
# Repeated cross-validation is used, which is a method of sub-sampling the data which helps avoid 
# overfitting (where the model is perfectly trained to its training data and
# cannot actually equally make accurate prediction from new data).
# Unfortunately, this model is performatively not much different from the baseline.
# Increasing the number of folds for cross-validation will typically help, but
# the imbalanced data issue is still present and will likely skew the results of
# most models without other measures being taken.
twoClassCtrl <- trainControl(method = "repeatedcv",
                             number = 5,
                             repeats = 2,
                             savePredictions = "final",
                             classProbs = TRUE,
                             summaryFunction = twoClassSummary,
                             verboseIter = TRUE)
set.seed(555)
rfTL <- train(y ~., data = dfTrainpx, method = "rf", metric = "ROC",
              trControl = twoClassCtrl, tuneLenth = 10)
rfTL
fittedTL <- predict(rfTL, dfTestpx)
confusionMatrix(reference = dfTestpx$y, data = fittedTL, mode = "everything", positive = "yes")

# One last model here before directly confronting the imbalanced issue is a grid
# search model. The training process involves our twoClassSummary element and
# cross-validation from the last model and adds a 'grid' of hyperparameters to
# try, deciding on the best one. Adding the grid search brings the accuracy
# up to 89% and Specificity up to just over 97%, marginally better than previous
# models. However, with Sensitivity as 19% it now seems to be time to handle
# the imbalanced data issue.
# Grid search:
# Accuracy    = 0.8861
# Sensitivity = 0.19108
# Specificity = 0.97116
rfGrid <- data.frame(mtry = c(3,5,7,9,10,11,12,13,15,17,19))
set.seed(555)
rfTG <- train(y ~ ., data = dfTrainpx, method = "rf", metric = "ROC",
              trControl = twoClassCtrl, tuneGrid = rfGrid)
print(rfTG)
fittedTG <- predict(rfTG, dfTestpx)
confusionMatrix(reference = dfTestpx$y, data = fittedTG, mode = "everything", positive = "yes")

# Down-sampling is a common method for approaching imbalanced data. This method
# reduces the number of observations used from the majority class (in this case,
# the 'yes' category) to balance this class with the other class ('no'). 
# Down-sampling
# With cv = 5:
# Though 5% less accurate overall than our grid rf, the down-sampled 
# rf dramatically improves on Sensitivity while losing respectively less
# Specificity. A better balance between these two metrics is preferred to
# a small loss in accuracy, so down-sampling seems best thus far.
# Accuracy    = 0.8306
# Sensitivity = 0.56051
# Specificity = 0.86360
downCtrl <- trainControl(method = "boot",
                         number = 2,                      
                         savePredictions = "final",
                         classProbs = TRUE,
                         summaryFunction = twoClassSummary,
                         sampling = "down",
                         verboseIter = TRUE)
set.seed(555)
rfDown <- train(y ~ ., data = dfTrainpx, method = "rf", metric = "ROC", 
                trControl = downCtrl, tuneLength = 10)
fittedDown <- predict(rfDown, dfTestpx)
confusionMatrix(reference = dfTestpx$y, data = fittedDown, mode = "everything", positive = "yes")

# Up-sampling
# Alternatively, one can use up-sampling which in our case will increase the 
# number of observations in the minority class ('no') as opposed to down-sampling
# which decreased the number of majority class ('yes') observations. Unlike
# down-sampling, you do not exclude any of your data. On the other hand,
# the additional minority class observations from up-sampling are synthesized
# from currently existing data. This means there can be distortions in the model.
# Up-sampling would not be preferred to down-sampling when working with
# a very large dataset. Here, however, it is worth trying and comparing it to
# other methods.
# This model is nearly equivalent to our down-sampled model.
# Accuracy =  = 0.8444
# Sensitivity = 0.54140
# Specificity = 0.88153
upCtrl <- trainControl(method = "boot",
                       number = 5,
                       savePredictions = "final",
                       classProbs = TRUE,
                       summaryFunction = twoClassSummary,
                       sampling = "up",
                       verboseIter = TRUE)
set.seed(555)
rfUp <- train(y ~ ., data = dfTrainpx, method = "rf", metric = "ROC", 
              trControl = upCtrl, tuneLength = 10)
rfUp
fittedUp <- predict(rfUp, dfTestpx)
confusionMatrix(reference = dfTestpx$y, data = fittedUp, mode = "everything", positive = "yes")
set.seed(555)


# Alternate boundaries for classifier thresholds.
# Setting a 33% probability of 'yes' to classification as 'yes' for
# our first two-class rf. Accuracy hardly changes as the true positive
# rate improves by around 11% while the true negative rate drop by only
# around 2%. Without down- or up-sampling, we have returned to
# imbalance issue.
# Accuracy: 0.8944
# Sensitivity = 0.26115
# Specificity = 0.97194
# Pos Pred Value = 0.53247         
# Neg Pred Value = 0.91489 
fittedProb <- predict(rfTL, dfTestpx, type = "prob")
alteredProb <- fittedProb$yes
alteredProb <- factor(ifelse(alteredProb >= 0.333, "yes","no"))
confusionMatrix(reference = dfTestpx$y, data = alteredProb, mode = "everything", positive = "yes")

# Setting a 33% threshold for 'yes' for our down-sampling rf model.
# True positive rate improves by almost 13% while true negative rate
# drops around 13% and Accuracy drops around 11%. 
# The two true rates are nearly balanced, the losses may not be worthwhile.
# Accuracy =    0.7264
# Sensitivity = 0.6879
# Specificity = 0.7311
# Pos Pred Value : 0.2384          
# Neg Pred Value : 0.9504
fittedProb <- predict(rfDown, dfTestpx, type = "prob")
alteredProb <- fittedProb$yes
alteredProb <- factor(ifelse(alteredProb >= 0.333, "yes","no"))
confusionMatrix(reference = dfTestpx$y, data = alteredProb, mode = "everything", positive = "yes")

# Fig - ROC curve for down-sample random forest with one-third threshold <----switch out for Steif's ROC curve code
# Plot ROC curve for down-sampled random forest with 0.333 threshold
# library(ROCR)
# predictions <- fittedProb$yes
# labels <- dfTestpx$y
# pred <- prediction(predictions, labels)
# pred
# # Required objects:
# # Performance metrics (Senstivity and Specificity)
# perf <- performance(pred, "tpr", "fpr")
# # Visualization
# plot(perf, avg = "threshold", colorize = TRUE)

# Training, resampling and ensembling over multiple models
# One more way to explore models is to use ensemble methods where multiple models
# are trained simultaneously. This section tries three different models, in caret
# listed under 
# Training control, which includes down-sampling given 
# its superior results in the prior models' comparisons.:
methodCtrl <- trainControl(method = "repeatedcv",
                           number = 2,
                           repeats = 2,
                           # Ensuring the sampling indices are the same for all models
                           index = createFolds(dfTrainpx$y, 5),
                           savePredictions = "final",
                           classProbs = TRUE,
                           sampling = "down",
                           summaryFunction = twoClassSummary,
                           verboseIter = TRUE)
# List of models to be used:
methodList <- c("rf", "glmnet", "svmRadial")
# Train the ensemble
library(caretEnsemble)
library(kernlab)
set.seed(555)
ensemble <- caretList(y ~ ., data = dfTrainpx, metric = "ROC", 
                      trControl = methodCtrl, 
                      # Here I will not actually use the methodList and instead
                      # manually add each model for additional tuning. tuneLength,
                      # as used earlier, will tune multiple models and choose 
                      # the best one. Random forests (rf) have already had good results,
                      # and glmnet models seem well-suited, so this code will return
                      # twice as many possible models for either of them as the 
                      # radial kernel support vector machine, which is worth
                      # trying given its common use in classification problems.
                      #methodList = methodList,
                      tuneList = list(
                        rf = caretModelSpec(method = "rf", tuneLength = 10),
                        glmnet = caretModelSpec(method = "glmnet", tuneLength = 10),
                        svm = caretModelSpec(method = "svmRadial", tuneLength = 5)
                      ))

# Comparing model performance, particularly by Median and Mean ROC, true positive
# rate, and true negative rate.
# Our rf and glmnet models are the two best, with glmnet winning in ROC and true
# negative rate but rf being superior in true positive rate. Their difference in ROC
# median and mean is less than 0.22 percentage points, leaving them nearly equivalent in 
# ROC. We will thus stay with the rf model given its higher Sensitivity.
resampledList <- resamples(ensemble)
summary(resampledList)

# Cost-benefit analysis using chosen model
# Creating a binary 1 or 0 vector for the outcome of whether or not someone accepted the credit,
# where 'no' becomes 0 and 'yes' becomes 1.
y_numTrain <- dfTrainpx$y
y_numTrain <- ifelse(y_numTrain == "no", 0, 1)
y_numTest <- dfTestpx$y
y_numTest <- ifelse(y_numTest == "no", 0, 1)

# Creating a dataframe containing actual outcome, the model's predicted probability for the outcome being
# a 'yes', and the model's predicted outcome with a 50% threshold (where if Prob >= 0.5, it predicts 1, i.e. 'yes')
# were we to use this model.
testProbs <- data.frame(
  Outcome = as.factor(y_numTest),
  Probs = predict(
    # Chosen model, test data, prediction method:
    rfDown,
    dfTestpx, 
    type= "prob"))

testProbs <- testProbs %>% 
  mutate(predOutcome  = as.factor(ifelse(testProbs$Probs.yes > 0.5 , 1, 0)),
         Probs = Probs.yes) %>%
  select(Outcome, Probs, predOutcome)
head(testProbs)

# Creating a cost-benefit table
cost_benefit_table <-
  testProbs %>%
  count(predOutcome, Outcome) %>%
  summarize(True_Negative = sum(n[predOutcome==0 & Outcome==0]),
            True_Positive = sum(n[predOutcome==1 & Outcome==1]),
            False_Negative = sum(n[predOutcome==0 & Outcome==1]),
            False_Positive = sum(n[predOutcome==1 & Outcome==0])) %>%
  gather(Variable, Count) %>%
  mutate(Revenue =
           case_when(Variable == "True_Negative"  ~ Count * 0,
                     Variable == "True_Positive"  ~ ((-2850 - 5000 + 10000 + 56000) * (Count * .25)) + 
                       (-2850 * (Count * .75)),
                     Variable == "False_Negative" ~ Count * 0,
                     Variable == "False_Positive" ~ (-2850) * Count)) %>%
  bind_cols(data.frame(Description = c(
    "Predicted correctly homeowner would not take the credit, no marketing resources were allocated, and no credit was allocated. 
$0",
    "Predicted correctly homeowner would take the credit; allocated the marketing resources, and 25% took the credit. 
25% were impacted by the marketing campaign: -$2,850 -$5,000 + $10,000 + $56,000;
75% were already going to apply for the credit, no impact from marketing: -$2,850",
    "Predicted incorrectly homeowner would not take the credit. These are likely homeowners who signed up for reasons unrelated to the marketing campaign. Thus, we '0 out' this category, assuming the cost/benefit of this is $0.",
    "Predicted incorrectly homeowner would take the credit; allocated marketing resources; no credit allocated.
-$2,850")))

# Fig - Cost-benefit table for model at default threshold
library(formattable)
formattable(cost_benefit_table %>% select(Description, everything()), align = c("l", rep("r", NCOL(cost_benefit_table) - 1)))


# Defining a function to iterate thresholds
# This function is drawn from Ken steif, https://urbanspatial.github.io/PublicPolicyAnalytics/ (2021) and 
# https://github.com/urbanSpatial/Public-Policy-Analytics-Landing .
iterateThresholds <- function(data, observedClass, predictedProbs, group) {
  #"This function takes as its inputs, a data frame with an observed binomial class (1 or 0); a vector of predicted 
  #probabilities; and optionally a group indicator like race. It returns accuracy plus counts and rates of confusion matrix 
  #outcomes."
  observedClass <- enquo(observedClass)
  predictedProbs <- enquo(predictedProbs)
  group <- enquo(group)
  x = .01
  all_prediction <- data.frame()
  if (missing(group)) {
    while (x <= 1) {
      this_prediction <- data.frame()
      
      this_prediction <-
        data %>%
        mutate(predclass = ifelse(!!predictedProbs > x, 1,0)) %>%
        count(predclass, !!observedClass) %>%
        summarize(Count_TN = sum(n[predclass==0 & !!observedClass==0]),
                  Count_TP = sum(n[predclass==1 & !!observedClass==1]),
                  Count_FN = sum(n[predclass==0 & !!observedClass==1]),
                  Count_FP = sum(n[predclass==1 & !!observedClass==0]),
                  Rate_TP = Count_TP / (Count_TP + Count_FN),
                  Rate_FP = Count_FP / (Count_FP + Count_TN),
                  Rate_FN = Count_FN / (Count_FN + Count_TP),
                  Rate_TN = Count_TN / (Count_TN + Count_FP),
                  Accuracy = (Count_TP + Count_TN) / 
                    (Count_TP + Count_TN + Count_FN + Count_FP)) %>%
        mutate(Threshold = round(x,2))
      all_prediction <- rbind(all_prediction,this_prediction)
      x <- x + .01
    }
    return(all_prediction)
  }
  else if (!missing(group)) { 
    while (x <= 1) {
      this_prediction <- data.frame()
      this_prediction <-
        data %>%
        mutate(predclass = ifelse(!!predictedProbs > x, 1,0)) %>%
        group_by(!!group) %>%
        count(predclass, !!observedClass) %>%
        summarize(Count_TN = sum(n[predclass==0 & !!observedClass==0]),
                  Count_TP = sum(n[predclass==1 & !!observedClass==1]),
                  Count_FN = sum(n[predclass==0 & !!observedClass==1]),
                  Count_FP = sum(n[predclass==1 & !!observedClass==0]),
                  Rate_TP = Count_TP / (Count_TP + Count_FN),
                  Rate_FP = Count_FP / (Count_FP + Count_TN),
                  Rate_FN = Count_FN / (Count_FN + Count_TP),
                  Rate_TN = Count_TN / (Count_TN + Count_FP),
                  Accuracy = (Count_TP + Count_TN) / 
                    (Count_TP + Count_TN + Count_FN + Count_FP)) %>%
        mutate(Threshold = round(x,2))
      all_prediction <- rbind(all_prediction,this_prediction)
      x <- x + .01
    }
    return(all_prediction)
  }
}
 # Producing a table where each row contains a unique threshold and its respective goodness of fit indicators.
whichThreshold <- 
  iterateThresholds(
    data=testProbs, observedClass = Outcome, predictedProbs = Probs)

whichThreshold[1:5,]

# The following plots and information aim to describe only the additional costs 
# and benefits derived from the campaign.


# Producing a plot of revenue based on threshold versus revenue, where functions
# are defined for true negatives, true positives, false negatives, and false
# positives based on known information.
# This first involves writing the function for each part of the confusion matrix,
# as seen above. In this script's example, it is assumed that 25% of the true
# positives are people who were actually impacted by the campaign and deciding
# to accept the credit thanks to resources and information given to them, while
# the other 75% are those who accepted the credit but were already planning to
# apply for the credit anyway.

# Remember that this analysis is only focused on the marketing campaign; thus
# for the false negatives who accepted the credit but no marketing resources 
# were allocated, this had no relationship with the campaign itself. The costs
# and benefits are thus not included here, and Count_FN is reflected as a 
# straight line equal to zero in the plot. This line also is equivalent to--and
# covers up in the plot--the line for true negatives where no resources were
# devoted to someone predicted to decline the credit.
# On the other hand, the false positives are defined by the marketing resources
# allocated--and lost--to these homeowners who ultimately did not accept the 
# credit despite contact. 
# Knowing that the threshold determines if a homeowner will be predicted to
# say 'yes' based on the probability produced by the machine-learning model,
# the lower the threshold the more likely it will be predicted as 'yes'. This
# is reflected in the dueling true positive line of threshold points (starting
# high as more people are predicted 'yes'and so  have more resources allocated to them,
# and social benefit is derived from those who accept the credit) versus the
# false positive line whereby the more homeowners are incorrectly predicted to
# say 'yes' the more resources are misallocated and social costs add up.
# As the threshold nears 1.00, where all homeowners would be predicted to say
# say 'no', all points reach zero as no resources are allocated and no benefit
# can be derived.

# Fig - Social benefit by confusion matrix type and threshold
whichThreshold <- 
  whichThreshold %>%
  dplyr::select(starts_with("Count"), Threshold) %>%
  gather(Variable, Count, -Threshold) %>%
  mutate(Revenue =
           case_when(Variable == "Count_TN"  ~ Count * 0,
                     Variable == "Count_TP"  ~ ((-2850 - 5000 + 10000 + 56000) * (Count * .25)) +
                       (-2850 * (Count * .75)),
                     Variable == "Count_FN"  ~ Count * 0,
                     Variable == "Count_FP"  ~ (-2850) * Count))
options(scipen = 999)
whichThreshold %>%
  ggplot(.,aes(Threshold, Revenue, colour = Variable)) +
  geom_point() +  
  labs(title = "Social benefit by confusion matrix type and threshold",
       y = "Social benefit") +
  guides(colour=guide_legend(title = "Confusion Matrix")) +
  theme_bw()


# Calculating losses for campaign versus total social benefit
# Here we focus on both the false positives and the 75% of true positives
# assumed to have accepted the credit regardless of the campaign.
# For both of these groups marketing resources were allocated 
# at a loss, i.e. did not impact the homeowners' decision. These
# two groups are aggregated below into 'Loss_Count'.
whichThreshold_benefit <- 
  whichThreshold %>% 
  # All clients who actually accepted the tax credit
  mutate(Loss_Count = ifelse(Variable == "Count_TP", (Count * .75),
                     ifelse(Variable == "Count_FP", Count, 0))) %>%  
  group_by(Threshold) %>% 
  summarize(Total_Benefit = sum(Revenue),
            Loss_Rate = sum(Loss_Count) / sum(Count),
            # - $2,850 in marketing resources per person in Loss_Count group
            Campaign_Loss =  sum(Loss_Count * -2850)) 
whichThreshold_benefit[1:5,]
whichThreshold_benefit %>% arrange(desc(Total_Benefit)) %>% head(n=5)

# Fig - Optimal threshold characteristics
# Displaying the optimal threshold from above, which is 0.63. 
# At this threshold, total social benefit as a function of individual and
# aggregate social premiums derived from use of the housing repair credit
# is equal to $645,150. This exceeds the dead loss resources due to
# incorrect prediction, leading to a net value of $156,375.
# This optimal threshold will be kept for future when planning to use
# the model to make predictions from new data.
(optimal_threshold <- pull(arrange(whichThreshold_benefit, -Total_Benefit)[1,1]))
(optimal_threshold_full <- whichThreshold_benefit %>% filter(Threshold == optimal_threshold))

# Fig - Total social benefit and campaign resources loss by threshold
# Plotting the total social benefit by threshold, with optimization at a
# threshold 0.63. Campaign resource loss is included as well to give a sense
# of its change with threshold; it begins to stabilize approximately around
# a threshold of 0.33.
whichThreshold_benefit %>%
  dplyr::select(Threshold, Total_Benefit, Campaign_Loss) %>%
  gather(Variable, Value, -Threshold) %>%
  ggplot(aes(Threshold, Value, colour = Variable)) +
  geom_point() +
  geom_vline(xintercept = pull(arrange(whichThreshold_revenue, -Total_benefit)[1,1])) +
  scale_y_continuous(breaks = round(seq(min(-6000000), max(3000000), by = 500000),10)) +
  labs(title = "Total social benefit with campaign resources lost by threshold",
       subtitle = paste0("Vertical line denotes optimal threshold of ", optimal_threshold),
       caption = "Campaign_Loss: Resource loss from misprediction
       Total_Benefit: Social benefit minus resource and credit allocation") +
  theme_bw()

# Comparison between no model and model
# Our model is based on the population in our test data, so this test data
# acts as a hypothetical 'next round' of homeowners for the campaign. 
# We know from the original data that marketing resources were allocated to each 
# individual without any discretion from using a predictive model.
# Therefore total spent on marketing without using a model is $4,104,000:
(orig_campaign_cost <- nrow(dfTestpx) * -2850)
# Again we assume that only 25% of homeowners who said 'yes' were actually 
# convinced via the campaign. Thus benefit derived from the campaign itself is 
# only from this 25%. 
# Number of 'yes'
test_yes <- dfTestpx %>% filter(y == 'yes') %>% select(y) %>% nrow()
# Number of 'no'
test_no <- dfTestpx %>% filter(y == 'no') %>% select(y) %>% nrow()

# Fig - Comparison between no model and model scenario
# Without a model, using our test data as our representation of homeowners,
# we know that:
# 1. Marketing resources were allocated to all individuals
# 2. Only considering the campaign itself, just as with using model, 
#    the only benefit considered here comes from the 25% of individuals who
#    accepted the credit because they were contacted by the campaign.
# 3. When considering only the campaign itself, the sum of all marketing
#    resources allocated for homeowners outside of this constraint are losses. 
#    This is reflected below when calculating Campaign_Loss for a scenario 
#    with no model.
# 4. As stated before, 'yes' only makes up around 11% of the data, making
#    the other around 89% 'no' all of whom had resources allocated to them
#    so we imagine Campaign_Loss to be very large.
# Total_Benefit and Campaign_Loss are thus calculated for the 'no model'
# scenario just as they were with the model.
#
#               Total_Benefit     Campaign_Loss
# No model  -----    -1709750          -3992138
# Model (at -----      645150           -488775
# optimum
# threshold)
#
#
compare_table <- data.frame(
  Scenario = c("No model", "Model at optimal threshold"),
  Description = c("No model is used, campaign marketing resources allocated to all eligible homeowners",
                  "Resources allocated based on model predictions at optimal threshold"),
  Total_Benefit = c(
    (-2850 - 5000 + 10000 + 56000) * (test_yes * .25) + (-2850 * (test_yes * .75)) + (test_no * -2850),
    optimal_threshold_full$Total_Benefit),
  Campaign_Loss = c(
    (test_no + (test_yes * .75)) * -2850, 
    optimal_threshold_full$Campaign_Loss))
compare_total <- data.frame(
  Scenario = "Gains from using model",
  Description = "Increase in total social benefit by $2,354,900 and becomes net positive; \n Increase in campaign account (decrease in losses due to error) by $3,503,362.50",
  Total_Benefit = paste0("+", compare_table[2,3] - compare_table[1,3]),
  Campaign_Loss = paste0("+", compare_table[2,4] - compare_table[1,4])
)
compare_table <- rbind(compare_table, compare_total)
formattable(compare_table, align = c("l", rep("r", NCOL(compare_table) - 4)))



