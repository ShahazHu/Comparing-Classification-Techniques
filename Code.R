install.packages("dplyr")
library(dplyr)

install.packages("RWeka")
library(RWeka)

install.packages("partykit")
library(partykit)

install.packages("caret")
library(caret)

install.packages("ROCR")
library(ROCR)

install.packages("class")
library(class)


##setwd("C:/Users/Shaha/Desktop/ITM 618 - Business Intelligence and Analytics/Group Project") /// FOR PERSONAL USE, NOT APART OF CODE // 

bank = read.csv("bank.csv", sep=";")  #Importing the data, specifying the separator as, stringsAsFactors=TRUE shows the factor levels;

str(bank)
dim(bank)           #dimensions of the data frame 
head(bank)          #
str(bank)           #structure of data frame
summary(bank)       #



mean(bank$age)      # 40.93621
mean(bank$balance)  # 1362.272
mean(bank$duration) # 258.1631
mean(bank$previous) #0.5803234


median(bank$age)                # 39
median(bank$job)                # "management"
median(bank$education)          #"secondary"
median(bank$loan)               #"no"


nrow(bank)*ncol(bank) #Total number of values
table(bank == "unknown") #statistics of unknown


boxplot(bank$age, 
        ylab = "age", 
        col = "light green", 
        horizontal = TRUE)

plot(table(bank$job), 
     ylab = "job", 
     col = "light green")

hist(bank$age, col = "light blue")


nrow(bank)*ncol(bank) #total number of cells. 


bank = read.csv("bank.csv", sep=";", stringsAsFactors=TRUE) #stringsAsFactors=TRUE shows the factor levels;

df <- select(bank, -c(poutcome))                                      #remove poutcome
df <- filter(df, across(.cols = everything(), ~ !grepl("unknown",.)))                           #across all rows, 
                                                                      #we search every column which depends on grepL returning true for rows 
                                                                      #that do not have the value of unknown
 

sum(bank == "unknown")  #number of occurrences of "unknown" in original data frame. 
nrow(bank) - nrow(df)   #number of rows containing "unknown" 
sum(df == "unknown")    #showing that there are no more unknowns in df 

mean(df$age)      # 40.93621
mean(df$balance)  # 1362.272
mean(df$duration) # 258.1631
mean(df$previous) #0.5803234

median(df$age)                # 39
median(df$job)                #"management"
median(df$education)          #"secondary"
median(df$loan)               #"no"
dim(df)

#CTREE BELOW

set.seed(123)
myIndex <- sample(1:nrow(df), 0.8*nrow(df))
myIndex
trainData <- df[myIndex, ]    #80% of DF rows   
nrow(trainData)               #number of rows for training, 
testData <- df[-myIndex, ]    #inverse of 80% which gives us 20%
nrow(testData)                #number of rows for testing

str(trainData) #showing structure. 
myFormula <- y ~ . #setting the target class and using all parameters. 
CTree <- ctree(myFormula, data=trainData)
testCTree <- predict(CTree, newdata = testData) # applying ctree
CTreeResult <- table(testCTree, testData$y)   #confusion matrix from ctree analysis 

confusionMatrix(table(testCTree, testData$y)) #analysis of ctree

precision <- posPredValue(table(testCTree, testData$y)) #calculates precision, using a table, positive = yes 
recall <- sensitivity(table(testCTree, testData$y)) #sensitivity function, to calculate recall
f1 <- (2 * precision * recall )/(precision+recall) # f1 using precision and recall



#Linear Regression - MODEL BELOW

LMtrainData <- trainData #I am making a copy of the data frame to change the y column to a numeral value
LMtestData <- testData

LMtrainData$y<-ifelse(LMtrainData$y =="yes",1,0) #lm can only use numeric values for the target class, so we define y as = 1 and no = 0 
LMtestData$y<-ifelse(LMtestData$y =="yes",1,0)   #lm can only use numeric values for the target class, so we define y as = 1 and no = 0 

logitmod <- lm(y ~ .,  data=LMtrainData)         #lm = linear model, predict 'Class' ,using all paremeters, use the 'LMtrainData set'
pred <- predict(logitmod, newdata = LMtestData, type = "response")        #predict function, use logicmodel, apply it on the test data, and type is response. 
y_predicted <- ifelse(pred > 0.5, 1, 0)                                 #apply a threshold, if greater than .5, class is 1 "yes" , otherwise it is 0. 


y_actual <- LMtestData$y #getting actual from testData#Class, comparing from predicted. seeing how many were correctly classified. 
Accuracy = sum(y_predicted == y_actual)/length(y_actual) #Create a sum where you see 'y predicted' == 'y actual' divided by the length of the y_actual(how many instances we had). 

confusionMatrix(table(y_predicted, y_actual))#to generate confusion matrix, use y predicted and y actual

precision1 <- posPredValue(table(y_predicted, y_actual)) #calculates precision, using a table, positive = yes 
recall1 <- sensitivity(table(y_predicted, y_actual)) #sensitivity function, to calculate recall
f11 <- (2 * precision1 * recall1 )/(precision1+recall1) # f1 using precision and recall

#J48 TREE

J48Tree <- J48(myFormula, data = trainData)
testJ48Tree <- predict(J48Tree, newdata = testData)
table(testJ48Tree, testData$y)

confusionMatrix(table(testJ48Tree, testData$y))

precision2 <- posPredValue(table(testJ48Tree, testData$y)) #calculates precision, using a table, positive = yes 
recall2 <- sensitivity(table(testJ48Tree, testData$y)) #sensitivity function, to calculate recall
f12 <- (2 * precision2 * recall2 )/(precision2+recall2) # f1 using precision and recall

#KNN Training Model

KNNtrainData <- trainData #I am making a copy of the data frame to change the y column to a numeral value, more for aesthetic purposes. 
KNNtestData <- testData


str(KNNtestData) # we see that we need to convert the variables with factor levels to integers for knn to work. 

columns <- c("job", "marital", "education", "default","housing", "loan", "contact", "month", "y") # list of all columns that are factors

KNNtrainData[,columns] <- lapply(KNNtrainData[,columns], as.integer) #conversion to int
KNNtestData[,columns] <- lapply(KNNtestData[,columns], as.integer) #conversion to int
row_labels = KNNtrainData[,16] #we take the label rows for the cl parameter in knn

sqrt(nrow(trainData)) #sqrt of the number of obs to get K 
result= knn(KNNtrainData, KNNtestData, row_labels, k = 157)
testTarget = KNNtestData$y #set our actual values here 

table(result,testTarget) #matrix 

confusionMatrix(table(result,testTarget)) #confusion matrix analysis for KNN


confusionMatrix(table(testCTree, testData$y))   #CTree
confusionMatrix(table(y_predicted, y_actual))   #Linear Regression
confusionMatrix(table(testJ48Tree, testData$y)) #J48Tree
confusionMatrix(table(result,testTarget)) #k-NN


precision3 <- posPredValue(table(result,testTarget)) #calculates precision, using a table, 
recall3 <- sensitivity(table(result,testTarget)) #sensitivity function, to calculate recall
f13 <- (2 * precision3 * recall3 )/(precision3+recall3) # f1 using precision and recall

preCompare <- c(precision, precision1, precision2, precision3)
barplot(preCompare,
        main = "Classifier Comparision Based on Precision",
        xlab = "Classifiers",
        ylab = "Precision",
        ylim = c(0,1),
        names.arg = c("CTree", "Linear Regression", "J48 TREE", "KNN Training Model"),
        col = "lightgreen")


accCompare <- c(0.8784, 0.8678, 0.8708, 0.8572)
barplot(accCompare,
        main = "Classifier Comparision Based on Accuracy",
        xlab = "Classifiers",
        ylab = "Precision",
        ylim = c(0,1),
        names.arg = c("CTree", "Linear Regression", "J48 Tree", "KNN Training Model"),
        col = "lightblue")

errCompare <- c(1-0.8784, 1-0.8678, 1-0.8708, 1-0.8572)
barplot(errCompare,
        main = "Classifier Comparision Based on Error Rate",
        xlab = "Classifiers",
        ylab = "Error Rate",
        ylim = c(0,1),
        names.arg = c("CTree", "Linear Regression", "J48 Tree", "KNN Training Model"),
        col = "peachpuff")


recCompare <- c(recall,recall1,recall2, recall3)
barplot(recCompare,
        main = "Classifier Comparision Based on Recall",
        xlab = "Classifiers",
        ylab = "Recall",
        ylim = c(0,1),
        names.arg = c("CTree", "Linear Regression", "J48 Tree", "KNN Training Model"),
        col = "azure2")


fCompare <- c(f1,f11,f12,f13)
barplot(fCompare,
        main = "Classifier Comparision Based on F1 Score",
        xlab = "Classifiers",
        ylab = "F1-Score",
        ylim = c(0,1),
        names.arg = c("CTree", "Linear Regression", "J48 Tree", "KNN Training Model"),
        col = "mistyrose2")

