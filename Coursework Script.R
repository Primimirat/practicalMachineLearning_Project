setwd("D:\\Dropbox\\R Work\Coursera\\Practical Machine Learning")
training <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", na.strings=c("NA",""), header=TRUE)
testing <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", na.strings=c("NA",""), header=TRUE)
#https://rpubs.com/sandrine1551/39319

#training <- read.csv("pml-training.csv", na.strings=c("NA",""), header=TRUE)
#testing <- read.csv("pml-testing.csv", na.strings=c("NA",""), header=TRUE)

#Load Packages
library(caret)
library(ggplot2)
library(randomForest)
library(rpart)
library(rpart.plot)

set.seed(1234)


#Confirm there are few complete cases and they do not favour one classe over another
Complete <- training[complete.cases(training),]
table(Complete$classe)

#Remove columns with NA's
trainingNoNA <- training[, colSums(is.na(training)) == 0]
testingNoNA <- testing[, colSums(is.na(training)) == 0]

#Remove first 7 columns that are just time stamps
testToUse <- testingNoNA[, -c(1:7)]
trainToUse <- trainingNoNA[, -c(1:7)]

#Confirm columns are the same in the test and training set ignoring the final column
colnamesTrain <- colnames(trainToUse)
colnamesTest <- colnames(testToUse)
all.equal(colnamesTrain[1:length(colnamesTrain)-1], colnamesTest[1:length(colnamesTest)-1])

#Subset the training set for Cross Validation
trainSubSet <- createDataPartition(y=trainToUse$classe, p=0.70, list=FALSE)
trainingSubSet <- trainToUse[trainSubSet,]
validationSubSet <- trainToUse[-trainSubSet,]

#Exploratory Data Analysis
graph <- ggplot(trainToUse, aes(x=classe))
graph2 <- graph + geom_histogram() + ylab("Number of entries") + xlab("Class of lift")
plot(graph2)
dev.off()

#Decision Tree
decisiontree <- rpart(classe~., data=trainingSubSet, method="class")
pdf("Decision Tree Plot.pdf")
decisiontreeplot <- rpart.plot(decisiontree, main="Decision Tree Plot", under=TRUE, extra=102, faclen=0)
dev.off()


#Random Forest
randommodel <- randomForest(classe~., data=trainingSubSet, method="class")

#Run Decision Tree on Validation Set
validatedecisiontree <- predict(decisiontree, validationSubSet, type="class")
DecisionTreePredictionResults <- confusionMatrix(validatedecisiontree, validationSubSet$classe)
DecisionTreePredictionResults
#DecisionTreePredictCorrect <- validatedecisiontree == validationSubSet$classe


#Run Random Tree on Validation Set
validaterandommodel <- predict(randommodel, validationSubSet, type="class")
RandomForestPredictionResults <- confusionMatrix(validaterandommodel, validationSubSet$classe)
RandomForestPredictionResults
#RandomForestPredictCorrect <- validaterandommodel == validationSubSet$classe


#Choose best Predictor
table(validatedecisiontree, validationSubSet$classe)
table(validaterandommodel, validationSubSet$classe)

#In and Out of Sample Error Rates
#Much better on the Random Forest so will use that for the Testing
#The insample error is....
#Therefore the out of sample error will be higher this

#Run on test data
Test.Random.Forest <- predict(randommodel, testToUse, type="class")
table(Test.Random.Forest)

#Write the files for submission
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(Test.Random.Forest)
