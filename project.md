# Practical Machine Learning - Course Project

##Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

##Objectives
The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.

##Load required libraries
**Here make use of doParallel library to enable caret to use multiple cores in the system**

```r
library(caret)
library(doParallel)
cl <- makeCluster(detectCores())
registerDoParallel(cl)
```

## Data Preparation
**Download and read data**

```r
trainingURL = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
trainingFile = "ml-training.csv"
testingURL = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
testingFile = "pml-testing.csv"
if (!file.exists(trainingFile)) {
    download.file(trainingURL,trainingFile)
}
if (!file.exists(testingFile)) {
    download.file(testingURL,testingFile)
}
trainingData = read.csv(trainingFile, header = TRUE, sep = ",", quote = "\"", na.strings=c("","NA","#DIV/0!"))
testingData = read.csv(testingFile, header = TRUE, sep = ",", quote = "\"", na.strings=c("","NA","#DIV/0!"))
```

**Creating training and testing dataset**

Partition the data, training set would have 60% of data and 40% for testing set. Columns such as user_name, timestamps, num_windows are not related to the measures and they are removed. Also, a zero variance analysis is performed to remove columns with zero or near zero variance. At last, all columns with NA values are also excluded.

```r
set.seed(20160521)
trainingSet <- createDataPartition(y=trainingData$classe, p=0.6, list=FALSE)
training <- trainingData[trainingSet,]
testing <- trainingData[-trainingSet,]

#Remove zero variance predictors
training <- training[,-nearZeroVar(training)]
#Remove id, username, timestamps and num_window which are not relevant to the reading
training <- training[,c(-1,-2,-3,-4,-5,-6)]

#Remove columns with NA values
training <- training[colSums(is.na(training))==0]
```

## Start training - Using Random forests
Use random forests to train the model, with cross validation

```r
rf <- train(classe ~., data=training, method="rf", prox=TRUE, trControl=trainControl(method="cv"))
rf$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry, proximity = TRUE) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.93%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3337    8    2    0    1 0.003285544
## B   21 2250    8    0    0 0.012724879
## C    0   15 2027   12    0 0.013145083
## D    0    0   26 1901    3 0.015025907
## E    0    2    4    8 2151 0.006466513
```

##Perform prediction and check for accurary
After the final model is established, examine the confusion matric with the testing set splitted from training set

```r
rfPredict <- predict(rf, testing)
rfConfusionMatric <- confusionMatrix(rfPredict, testing$classe)
rfConfusionMatric
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2229   15    0    0    0
##          B    3 1496    5    1    0
##          C    0    7 1358   27    1
##          D    0    0    5 1256    2
##          E    0    0    0    2 1439
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9913         
##                  95% CI : (0.989, 0.9933)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.989          
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9987   0.9855   0.9927   0.9767   0.9979
## Specificity            0.9973   0.9986   0.9946   0.9989   0.9997
## Pos Pred Value         0.9933   0.9940   0.9749   0.9945   0.9986
## Neg Pred Value         0.9995   0.9965   0.9985   0.9954   0.9995
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2841   0.1907   0.1731   0.1601   0.1834
## Detection Prevalence   0.2860   0.1918   0.1775   0.1610   0.1837
## Balanced Accuracy      0.9980   0.9920   0.9936   0.9878   0.9988
```
The accuracy of the prediction is 99.13% while out of sample error is 0.87%. This indicate the prediction is accurate.

##Perform prediction with testing data
**Use the trained random forest to predict the testing data**

```r
testingData$classe <- predict(rf, testingData)
testingData[,c(160,161)]
```

```
##    problem_id classe
## 1           1      B
## 2           2      A
## 3           3      B
## 4           4      A
## 5           5      A
## 6           6      E
## 7           7      D
## 8           8      B
## 9           9      A
## 10         10      A
## 11         11      B
## 12         12      C
## 13         13      B
## 14         14      A
## 15         15      E
## 16         16      E
## 17         17      A
## 18         18      B
## 19         19      B
## 20         20      B
```


