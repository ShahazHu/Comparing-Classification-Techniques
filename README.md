# Comparing-Classification-Techniques

The purpose of this project is to identify the best classifier method for a large data set to predict if a bank’s client has subscribed to a term deposit. The four classifier methods tested were Logistic Regression, J48, CTree, and K Nearest Neighbour (k-NN). After cleaning and preparing the dataset, the classifiers were then used and compared against each other. The experiments concluded that CTree was most effective at predicting the dataset, and therefore was declared the best classifier for this experiment. 

## Introduction

As technology continues its unprecedented growth and become a bigger part of businesses all around the world, vast amounts of data are generated everyday from these processes. Sectors like finance, to biology, produce huge amounts of information, known as ‘big data, which, when properly analyzed, can lead to invaluable insights to solve problems. With many ways to classify data, the objective is to identify the best classifier out of four, given a particular dataset. 

The dataset that will be analysed is from a Portuguese banking institution . The data is concerning direct marketing campaigns. Our problem is to identify the best classification method for determining the output variable ‘y’; if the client has subscribed to a term deposit. 

Some types of datasets can be used with classification learning techniques to make predictive models for determining unknown values. Four classification models which will be used to make an analysis of the data are CTree, J48 tree, Linear Classifier, & K Nearest Neighbour (k-NN).

CTree, also known as conditional inference tree, is a decision tree which predicts a relationship using statistical method known as BRP (binary recursive partitioning) .

J48 tree is also another decision tree, using an algorithm: C4.5, to generate a decision tree. In R, depending on the package used, this tree can be either pruned or unpruned. 

Linear classification uses a ‘line of best fit’ to classify the data. For analysis of this particular dataset, logistic regression modeling will be used.  Both can be used interchangeably, as they are a type of linear classifier. In terms of R, logistic regression target variable is continuous. 

K Nearest Neighbor it relies on a ‘k’ value and uses Euclidean distance training set vectors, and then the classification of each data point is decided by a vote to which k it belongs in. 


## Methodology
To complete the experiment, we first needed to understand the data that is given to us, as we already understand that the business wants to identify ways to correctly predict if the client subscribed a term deposit. Looking through the given dataset, we see that it’s saved as a csv file ordered but not correctly into columns, so it was noted to indicate the data was separated with “;” when importing into a data frame. We then begin to preprocess the data, by removing “poutcome”, and all unknown values, from the data frame. We format the data frame to fit individual classification method, as each required that the data had to be formatted a certain way, such as all columns needed to be an integer, some had to be factors, etc. We then did an 80-20 split for training and testing the dataset. Finally, we calculate the results and compare to see which classifier is best. 

## Experiment Design

The experiment requires the data frame is be sorted and prepared for each classifier, so the objective was to clean the data set, and make an 80-20 split.

### Description of the given dataset

The dataset’s initial dimensions are 45,211 observations by 17 variables. Variables 1-7 refer to the bank client data, 9-12 are related to the last contact of the current campaign, 13-16 are other attributes, 17 is the output variable, in this case, our target variable. The variables are:  

```
1.	age (numeric)
2.	job : type of job (categorical)
3.	marital : marital status (categorical)
4.	education (categorical)
5.	default: has credit in default? (binary)
6.	balance: average yearly balance, in euros (numeric) 
7.	housing: has housing loan? (binary)
8.	loan: has personal loan? (binary)
9.	contact: contact communication type (categorical)
10.	day: last contact day of the month (numeric)
11.	month: last contact month of year (categorical)
12.	duration: last contact duration, in seconds (numeric)  
13.	campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
14.	pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)
15.	previous: number of contacts performed before this campaign and for this client (numeric)
16.	poutcome: outcome of the previous marketing campaign (categorical)
17.	y - has the client subscribed a term deposit? (binary)

```

We can look at some of the means for different variables:

```
•	Mean of age = 40.93621
•	Mean of balance = 1362.272
•	Mean of duration = 258.1631
•	Mean of previous = 0.5803234
```

Some of the medians include: 
```
•	Median of age: 39
•	Median of job: “management”
•	Median of education: “secondary”
•	Median of loan: “no”
```
For the “unknown” entries, there are 52,124 entries out of a total of 716,463, which is approximately 7.275% of total observations. There are 14,304 rows containing “unknown”.
![Figure 1](https://github.com/ShahazHu/Comparing-Classification-Techniques/assets/61039853/09b355ea-d6fa-4e73-9209-093cf39f3e5e)

Figure 1 above is the boxplot for age. We can see that a majority takes place between 30 and 50. With outlies above 65~. 

![Figure2](https://github.com/ShahazHu/Comparing-Classification-Techniques/assets/61039853/ce4338e5-8bb7-42d0-a5ea-f1434a2c13b7)

Figure 2 above is another way and arguably a better visualization of understanding the frequency of age across the age category. 

### Preprocessing the data
Preprocessing the data begins with removing the unknown entries. We first start with removing the poutcome column. To remove the poutcome column, we can use: 
```
df <- select(bank, -c(poutcome)),
```
which takes all columns except poutcome. 

Then we go through each row and remove any row that contains unknown. To do those, we can use:
```
df <- filter(df, across(.cols = everything(), ~ !grepl("unknown",.)))
```
which states that we are going to filter the data frame across all columns in every variable, depending on grep to see if unknown is not there, in the entire data frame. 
The dataset is now removed of all known values. The new dimensions of the data frame are now 30,807 observations by 16 variables. The means and medians are the same. 

### Preparing the Training and Test Datasets

To prepare the training and testing data set, we start by setting the seed to ‘123’. We create an index vector which chooses 80% of the rows of our data frame. We then create two new data frames, applying 80% of the data from our original data frame, and using it as our training dataset, and setting the 20% remainder the data frame as our test dataset. 

From the original data frame, which has 30,807 observations the training data set has 24,735 observations, and the testing dataset has 6,182 observations for testing the model. 

## Results

The following results for each classification will be presented by showing the confusion matrix, accuracy, error rate, precision, recall and f1-score. These classifications use all variables to compute the final value. We use the confusionMatrix method from RStudio’s ‘caret’ package to help compute these terms. 

### CTree Classification
The confusion matrix of CTree classification generated from RStudio’s ‘caret’ package: 
![image](https://github.com/ShahazHu/Comparing-Classification-Techniques/assets/61039853/b4b2dfe1-f497-4276-adfc-4f4d86c6612f)

The accuracy noted above is 87.84%. This is calculated as:
((4964 + 466))/((4964 + 466 + 299 + 466))

Using this value, the error rate is 1-0.8784 = 0.1216 = 12.16%

Precision (pos Pred Value) = 4964/(4964+453) = 0.916374377
Recall (sensitivity) =  4964/(4964+299)= 0.943188296
F1_Score: (2*0.916374377*0.943188296.)/(0.916374377+ 0.943188296)= 0.929588

### Logistic Regression
![image](https://github.com/ShahazHu/Comparing-Classification-Techniques/assets/61039853/ffbc9103-4e46-463e-bfed-3da42389dcd3)

The accuracy noted above is 86.78%. This is calculated using   ((5150+704))/((5150+704+113+215)). Using this value, the error rate is 1-0.8678 = 0.1322 = 13.22%
Precision (pos Pred Value) = 5150/(5150+704)= 0.879740
Recall (sensitivity) =  4964/(4964+299)= 0.978529
F1_Score: (2*0.916374377*0.943188296.)/(0.916374377+ 0.943188296)= 0.926509

### J48 Tree

![image](https://github.com/ShahazHu/Comparing-Classification-Techniques/assets/61039853/a1c09d83-092f-4852-8ede-0ef4bb0c39f0)

The accuracy noted above is 87.08%. This is calculated using   ((4952+431))/((4952+488+311+431)). Using this value, the error rate is 1-0.8708 = 0.1292 = 12.92%
Precision (pos Pred Value) = 4952/(4952+488)= 0.910294
Recall (sensitivity) =  4952/(4952+311)= 0.940908
F1_Score: (2*0.910294*0.940908)/(0.910294 + 0.940908)= 0.925348


### k-NN Classification
![image](https://github.com/ShahazHu/Comparing-Classification-Techniques/assets/61039853/56a861a9-0986-49f2-87c3-4359f6c5c77d)

The accuracy is 85.72%. This is calculated using   ((4952+431))/((4952+488+311+431)). Using this value, the error rate is 1-0.8572 = 0.1428 = 14.28%
Precision (pos Pred Value) = 4952/(4952+488)= 0.866098.
Recall (sensitivity) =  4952/(4952+311)= 0.984420.
F1_Score: (2*0.910294*0.940908)/(0.910294 + 0.940908)= 0.921476.


## Discussion on the Results
### Classifier Comparison Based on Accuracy 
![image](https://github.com/ShahazHu/Comparing-Classification-Techniques/assets/61039853/37ce19cd-310e-41b2-924f-c4280a6e5526)

Ranked in order by accuracy: 
```
1.	CTree
2.	J48 Tree
3.	Logistic Regression
4.	k-NN
```
Based on the graphics, the important to note that the accuracy between these classifiers may seem small. 0.8784, 0.8678, 0.8708, 0.8572, on average, having around 1% difference, but this can be significant when applied to very large datasets. 

### Classifier Comparison Based on Precision
![image](https://github.com/ShahazHu/Comparing-Classification-Techniques/assets/61039853/f54968da-9768-487f-bb33-e31c721216f2)

We see above that the J48 classifier has the highest precision out of all the classifiers. Ranking in order, it is:
```
1.	CTree 
2.	J48 Tree 
3.	Logistical Regression
4.	k-NN. 
```
The most significant percentage difference is between the top 2 versus the bottom two, having greater than 3% difference. 

### Classifier Comparison Based on Recall
![image](https://github.com/ShahazHu/Comparing-Classification-Techniques/assets/61039853/53d8df9b-7d2d-46cf-8032-1bdf174b83dc)

Ranked in order of highest recall to lowest: 
```
1.	k-NN
2.	Logistic Regression
3.	CTree
4.	J48 Tree
```
Percentage wise, there’s a significant percentage between the top two and the bottom two. Between 2) and 3), there is a ~3.5% difference, while 1) and 2) have a ~0.589% difference, 3) and 4) have a ~ 0.228 difference. 

### Classifier Comparison Based on Error Rate
![image](https://github.com/ShahazHu/Comparing-Classification-Techniques/assets/61039853/29305a12-4931-4436-95c2-3eb81497af16)


Ranking from lowest error rate to highest, we have: 
```
1)	CTree
2)	J48 Tree
3)	Logistic Regression
4)	k-NN
```
K-NN had the highest error rate, having 2.12% more errors than CTree. 

### Classifier Comparison Based on F1-Score
![image](https://github.com/ShahazHu/Comparing-Classification-Techniques/assets/61039853/75158ea0-e56c-4008-8716-7f9e81bcb228)

The F1-scores were very close, ranking from highest to lowest: 
```
1.	CTree = 0.9295880
2.	Logistic Regression = 0.9265090
3.	J48 = 0.9253480
4.	k-NN = 0.9214762
```

The percentage difference between them were ranging from ~0.1% to ~0.3% respectively. 
To conclude, on ranking the overall classifiers, 
```
1.	CTree
2.	J48 
3.	Logistic Regression
4.	k-NN. 
```
We use the F1 Score of each classifier to along with all the other comparisons, to determine which is the best. Overall, we see that CTree has performed the best.

## Conclusion
To conclude, CTree was the best classifier. This does not mean that CTree is superior in every dataset, but in this case, for the purpose of this dataset and to solve the given problem, CTree had the highest accuracy. 







## References: 

  [Moro et al., 2011] S. Moro, R. Laureano and P. Cortez. Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology. 
  In P. Novais et al. (Eds.), Proceedings of the European Simulation and Modelling Conference - ESM'2011, pp. 117-121, Guimarães, Portugal, October, 2011. EUROSIS.
  Available at: [pdf] http://hdl.handle.net/1822/14838
                [bib] http://www3.dsi.uminho.pt/pcortez/bib/2011-esm-1.txt
  Information on CTrees : https://www.rdocumentation.org/packages/partykit/versions/1.2-7/topics/ctree
  Information on J48L https://www.rdocumentation.org/packages/RWeka/versions/0.4-43/topics/Weka_classifier_trees
  Information on Linear Modeling: https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/lm
  k-NN information: https://www.rdocumentation.org/packages/class/versions/7.3-19/topics/knn
