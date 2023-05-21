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

## References: 

  [Moro et al., 2011] S. Moro, R. Laureano and P. Cortez. Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology. 
  In P. Novais et al. (Eds.), Proceedings of the European Simulation and Modelling Conference - ESM'2011, pp. 117-121, Guimarães, Portugal, October, 2011. EUROSIS.
  Available at: [pdf] http://hdl.handle.net/1822/14838
                [bib] http://www3.dsi.uminho.pt/pcortez/bib/2011-esm-1.txt
  Information on CTrees : https://www.rdocumentation.org/packages/partykit/versions/1.2-7/topics/ctree
  Information on J48L https://www.rdocumentation.org/packages/RWeka/versions/0.4-43/topics/Weka_classifier_trees
  Information on Linear Modeling: https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/lm
  k-NN information: https://www.rdocumentation.org/packages/class/versions/7.3-19/topics/knn
