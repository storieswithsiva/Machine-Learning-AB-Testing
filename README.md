# A/B Testing with Machine Learning - A Step-by-Step Tutorial

[![Makes people smile](https://forthebadge.com/images/badges/makes-people-smile.svg)](https://github.com/iamsivab)

## Machine-Learning-AB-Testing

[![Generic badge](https://img.shields.io/badge/Datascience-Beginners-Red.svg?style=for-the-badge)](https://github.com/iamsivab/Machine-Learning-AB-Testing) 
[![Generic badge](https://img.shields.io/badge/LinkedIn-Connect-blue.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/iamsivab/) [![Generic badge](https://img.shields.io/badge/Python-Language-blue.svg?style=for-the-badge)](https://github.com/iamsivab/Machine-Learning-AB-Testing) [![ForTheBadge uses-git](http://ForTheBadge.com/images/badges/uses-git.svg)](https://GitHub.com/)

#### The goal of this project is to implement Machine Learning [#DataScience](https://github.com/iamsivab/Machine-Learning-AB-Testing) for A/B Testing step-by-step.

[![GitHub repo size](https://img.shields.io/github/repo-size/iamsivab/Machine-Learning-AB-Testing.svg?logo=github&style=social)](https://github.com/iamsivab) [![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/iamsivab/Machine-Learning-AB-Testing.svg?logo=git&style=social)](https://github.com/iamsivab/)[![GitHub top language](https://img.shields.io/github/languages/top/iamsivab/Machine-Learning-AB-Testing.svg?logo=python&style=social)](https://github.com/iamsivab)

#### Few popular hashtags - 
### `#ABTesting` `#Machine Learninig` `#Linear Regression`
### `#Learning Recommendation` `#Decision Tree` `#XGBoost`

### Motivation

With the rise of digital marketing led by tools including Google Analytics, Google Adwords, and Facebook Ads, a key competitive advantage for businesses is using A/B testing to determine effects of digital marketing efforts. Why? In short, small changes can have big effects.

` https://www.business-science.io/business/2019/03/11/ab-testing-machine-learning.html ` 

This is why A/B testing is a huge benefit. A/B Testing enables us to determine whether changes in landing pages, popup forms, article titles, and other digital marketing decisions improve conversion rates and ultimately customer purchasing behavior. A successful A/B Testing strategy can lead to massive gains - more satisfied users, more engagement, and more sales - Win-Win-Win.


### About the Project

A major issue with traditional, statistical-inference approaches to A/B Testing is that it only compares 2 variables - an experiment/control to an outcome. The problem is that customer behavior is vastly more complex than this. Customers take different paths, spend different amounts of time on the site, come from different backgrounds (age, gender, interests), and more. This is where Machine Learning excels - generating insights from complex systems.

In this article you will experience how to implement Machine Learning for A/B Testing step-by-step. After reading this post you will:

``` Understand what A/B Testing is

Understand why Machine Learning is a better approach for performing A/B Testing versus traditional statistical inference (e.g. z-score, t-test)

Get a Step-by-Step Walkthrough for implementing machine learning for A/B Testing in R using 3 different algorithms:
Linear Regression
Decision Trees
XGBoost
Develop a Story for what contributes to the goal of gaining Enrollments

Get a Learning Recommendation for those that want to learn how to implement machine learning following best practices for any business problem. 

```


#### Steps involved in this project

[![Made with Python](https://forthebadge.com/images/badges/made-with-python.svg)](https://github.com/iamsivab/Machine-Learning-AB-Testing) [![Made with love](https://forthebadge.com/images/badges/built-with-love.svg)](https://www.linkedin.com/in/iamsivab/) [![ForTheBadge built-with-swag](http://ForTheBadge.com/images/badges/built-with-swag.svg)](https://www.linkedin.com/in/iamsivab/)

#### 1.0 What is A/B Testing?

A/B Testing is a tried-and-true method commonly performed using a traditional statistical inference approach grounded in a hypothesis test (e.g. t-test, z-score, chi-squared test). In plain English, 2 tests are run in parallel:

``` 
Treatment Group (Group A) - This group is exposed to the new web page, popup form, etc.

Control Group (Group B) - This group experiences no change from the current setup.

```

The goal of the A/B is then to compare the conversion rates of the two groups using statistical inference.

The problem is that the world is not a vacuum involving only the experiment (treatment vs control group) and effect. The situation is vastly more complex and dynamic. Consider these situations:

- Users have different characteristics: Different ages, genders, new vs returning, etc

- Users spend different amounts of time on the website: Some hit the page right away, others spend more time on the site

- Users are find your website differently: Some come from email or newsletters, others from web searches, others from social media

- Users take different paths: Users take actions on the website going to different pages prior to being confronted with the event and goal

Often modeling an A/B test in this vacuum can lead to misunderstanding of the true story.

#### 2.0 Why use Machine Learning?

Unlike statistical inference, Machine Learning algorithms enable us to model complex systems that include all of the ongoing events, user features, and more. There are a number of algorithms each with strengths and weaknesses.

` An attractive benefit to Machine Learning is that we can combine multiple approaches to gain insights. ` 

Rather than discuss in abstract, we can use an example from Udacity’s A/B Testing Course, but apply the applied Machine Learning techniques from our Business Analysis with R Course to gain better insights into the inner-workings of the system rather than simply comparing an experiment and control group in an A/B Test.

#### 3.0 A/B Test Using Machine Learning: Step-By-Step Walkthrough

# Experiment Name: “Free Trial” Screener

In the experiment, Udacity tested a change where if the student clicked “start free trial”, they were asked how much time they had available to devote to the course.

If the student indicated 5 or more hours per week, they would be taken through the checkout process as usual. If they indicated fewer than 5 hours per week, a message would appear indicating that Udacity courses usually require a greater time commitment for successful completion.

Why Implement the Form?

The goal with this popup was that this might set clearer expectations for students upfront, thus reducing the number of frustrated students who left the free trial because they didn’t have enough time.

However, what Udacity wants to avoid is “significantly” reducing the number of students that continue past the free trial and eventually complete the course.

### Project Goal

In this analysis, we will investigate which features are contributing enrollments and determine if there is an impact on enrollments from the new “Setting Expectations” form.

The users that experience the form will be denoted as “Experiment = 1”
The control group (users that don’t see the form) will be denoted as “Experiment = 0”.

### 3.1 Get the Data

``` 
The data set for this A/B Test can be retrieved from Kaggle Data Sets.

Control Data(https://www.kaggle.com/tammyrotem/control-data)

Experiment Data(https://www.kaggle.com/tammyrotem/experiment-data)

```

### Investigate the Data

We have 5 columns consisting of:

Date: a character formatted Day, Month, and Day of Month
Pageviews: An aggregated count of Page Views on the given day
Clicks: An aggregated count of Page Clicks on the given day for the page in question
Enrollments: An aggregated count of Enrollments by day.
Payments: An aggregated count of Payments by day.

### Key Points

37 total observations in the control set and 37 in the experiment set

Data is time-based and aggregated by day - This isn’t the best way to understand complex user behavior, but we’ll go with it

We can see that Date is formatted as a character data type. This will be important when we get to data quality. We’ll extract day of the week features out of it.

Data between the experiment group and the control group is in the same format. Same number of observations (37 days) since the groups were tested in parallel.

### 3.5 Data Quality Check

Next, let’s check the data quality. We’ll go through a process involves:

` Check for Missing Data - Are values missing? What should we do? `

Check Data Format - Is data in correct format for analysis? Are all features created and in the right class?

### 3.5.1 Check for Missing Data

Key Point: We have 14 days of missing observations that we need to investigate

Key Point: The count of missing data is consistent (a good thing). We still need to figure out what’s going on though.

Key Point: We don’t have Enrollment information from November 3rd on. We will need to remove these observations.

### 3.5.2 Check Data Format

Key Points:

Date is in character format. It doesn’t contain year information. Since the experiment was only run for 37 days, we can only realistically use the “Day of Week” as a predictor.

The other columns are all numeric, which is OK. We will predict the number of Enrollments (regression) (taught in Week 6 of Business Analysis with R course)

Payments is an outcome of Enrollments so this should be removed.

### 3.6 Format Data

Now that we understand the data, let’s put it into the format we can use for modeling. We’ll do the following:

- Combine the control_tbl and experiment_tbl, adding an “id” column indicating if the data was part of the experiment or not
- Add a “row_id” column to help for tracking which rows are selected for training and testing in the modeling section
- Create a “Day of Week” feature from the “Date” column
- Drop the unnecessary “Date” column and the “Payments” column
- Handle the missing data (NA) by removing these rows.
- Shuffle the rows to mix the data up for learning
- Reorganize the columns

### 3.7 Training and Testing Sets

With the data formatted properly for analysis, we can now separate into training and testing sets using an 80% / 20% ratio. We can use the initial_split() function from rsample to create a split object, then extracting the training() and testing() sets.

### 3.8 Implement Machine Learning Algorithms

We’ll implement the new parsnip R package. For those unfamiliar, here are some benefits:

Our strategy will be to implement 3 modeling approaches:

``` 
Linear Regression - Linear, Explainable (Baseline)
Decision Tree
Pros: Non-Linear, Explainable.
Cons: Lower Performance
XGBoost
Pros: Non-Linear, High Performance
Cons: Less Explainable
```

### 3.8.1 Linear Regression (Baseline)

Key Points:

Our model is on average off by +/-19 enrollments (means absolute error). The test set R-squared is quite low at 0.06.

We investigated the predictions to see if there is anything that jumps out at us. The model had an issue with observation 7, which is likely throwing off the R-squared value.

We investigated feature importance. Clicks, Pageviews, and Experiment are the most important features. Experiment is 3rd, with a p.value 0.026. Typically this is considered significant.

We can also see the term coefficient for Experiment is -17.6 indicating as decreasing Enrollments by -17.6 per day when the Experiment is run.

### 3.8.2 Helper Functions
### 3.8.3 Decision Trees

Key Points:

Our new model has roughly the same accuracy to +/-19 enrollments (MAE) as the linear regression model.

Experiment shows up towards the bottom of the tree. The rules indicate a when Experiment >= 0.5, there is a drop in enrollments.



### Libraries Used

![R Studio](https://img.shields.io/badge/R-dplyr-blue.svg?style=flat&logo=r&logoColor=white) 
![R Studio](https://img.shields.io/badge/R-stringr-blue.svg?style=flat&logo=r&logoColor=white)
![R Studio](https://img.shields.io/badge/R-readtext-blue.svg?style=flat&logo=r&logoColor=white) 
![R Studio](https://img.shields.io/badge/R-e1071-blue.svg?style=flat&logo=r&logoColor=white) 
![R Studio](https://img.shields.io/badge/R-mlr-blue.svg?style=flat&logo=r&logoColor=white)
![R Studio](https://img.shields.io/badge/R-caret-blue.svg?style=flat&logo=r&logoColor=white) 
![R Studio](https://img.shields.io/badge/R-randomForest-blue.svg?style=flat&logo=r&logoColor=white) 


### Installation

- Install **randomForest** using pip command: `install.packages("randomForest")`
- Install **caret** using pip command: `install.packages("caret")`
- Install **mlr** using pip command: `install.packages("mlr")`
- Install **MASS** using pip command: `install.packages("MASS")`

### How to run?

[![R Studio](https://img.shields.io/badge/R-clean_data.R.-lightgrey.svg?logo=R&style=social)](https://github.com/iamsivab/Machine-Learning-AB-Testing/tree/master/src)


### Project Reports

[![report](https://img.shields.io/static/v1.svg?label=Project&message=Report&logo=microsoft-word&style=social)](https://github.com/iamsivab/Machine-Learning-AB-Testing/blob/master/Sivasubramanian-Text%20Mining%20Report.pdf)

- [Download](https://github.com/iamsivab/Machine-Learning-AB-Testing/blob/master/Sivasubramanian-Text%20Mining%20Report.pdf) for the report.

### Useful Links

1. [Why Term Frequency is better than TF-IDF for text classification](https://www.quora.com/Why-does-TF-term-frequency-sometimes-give-better-F-scores-than-TF-IDF-does-for-text-classification)
 
### Related Work

[![Sentiment Analysis](https://img.shields.io/static/v1.svg?label=Text&message=Mining&color=lightgray&logo=linkedin&style=social&colorA=critical)](https://www.linkedin.com/in/iamsivab/) [![GitHub top language](https://img.shields.io/github/languages/top/iamsivab/Machine-Learning-AB-Testing.svg?logo=php&style=social)](https://github.com/iamsivab/)

[Text Mining Analyzer](https://github.com/iamsivab/Machine-Learning-AB-Testing) - A Detailed Report on the Analysis


### Contributing

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?logo=github)](https://github.com/iamsivab/Machine-Learning-AB-Testing/pulls) [![GitHub issues](https://img.shields.io/github/issues/iamsivab/Machine-Learning-AB-Testing?logo=github)](https://github.com/iamsivab/Machine-Learning-AB-Testing/issues) ![GitHub pull requests](https://img.shields.io/github/issues-pr/viamsivab/Machine-Learning-AB-Testing?color=blue&logo=github) 
[![GitHub commit activity](https://img.shields.io/github/commit-activity/y/iamsivab/Machine-Learning-AB-Testing?logo=github)](https://github.com/iamsivab/Machine-Learning-AB-Testing/)

- Clone [this](https://github.com/iamsivab/Machine-Learning-AB-Testing/) repository: 

```bash
git clone https://github.com/iamsivab/Machine-Learning-AB-Testing.git
```

- Check out any issue from [here](https://github.com/iamsivab/Machine-Learning-AB-Testing/issues).

- Make changes and send [Pull Request](https://github.com/iamsivab/Machine-Learning-AB-Testing/pull).
 
### Need help?

[![Facebook](https://img.shields.io/static/v1.svg?label=follow&message=@iamsivab&color=9cf&logo=facebook&style=flat&logoColor=white&colorA=informational)](https://www.facebook.com/iamsivab)  [![Instagram](https://img.shields.io/static/v1.svg?label=follow&message=@iamsivab&color=grey&logo=instagram&style=flat&logoColor=white&colorA=critical)](https://www.instagram.com/iamsivab/) [![LinkedIn](https://img.shields.io/static/v1.svg?label=connect&message=@iamsivab&color=success&logo=linkedin&style=flat&logoColor=white&colorA=blue)](https://www.linkedin.com/in/iamsivab/)

:email: Feel free to contact me @ [balasiva001@gmail.com](https://mail.google.com/mail/)

[![GMAIL](https://img.shields.io/static/v1.svg?label=send&message=balasiva001@gmail.com&color=red&logo=gmail&style=social)](https://www.github.com/iamsivab) [![Twitter Follow](https://img.shields.io/twitter/follow/iamsivab?style=social)](https://twitter.com/iamsivab)


### License

MIT &copy; [Sivasubramanian](https://github.com/iamsivab/Machine-Learning-AB-Testing/blob/master/LICENSE)

[![](https://sourcerer.io/fame/iamsivab/iamsivab/Machine-Learning-AB-Testing/images/0)](https://sourcerer.io/fame/iamsivab/iamsivab/Machine-Learning-AB-Testing/links/0)[![](https://sourcerer.io/fame/iamsivab/iamsivab/Machine-Learning-AB-Testing/images/1)](https://sourcerer.io/fame/iamsivab/iamsivab/Machine-Learning-AB-Testing/links/1)[![](https://sourcerer.io/fame/iamsivab/iamsivab/Machine-Learning-AB-Testing/images/2)](https://sourcerer.io/fame/iamsivab/iamsivab/Machine-Learning-AB-Testing/links/2)[![](https://sourcerer.io/fame/iamsivab/iamsivab/Machine-Learning-AB-Testing/images/3)](https://sourcerer.io/fame/iamsivab/iamsivab/Machine-Learning-AB-Testing/links/3)[![](https://sourcerer.io/fame/iamsivab/iamsivab/Machine-Learning-AB-Testing/images/4)](https://sourcerer.io/fame/iamsivab/iamsivab/Machine-Learning-AB-Testing/links/4)[![](https://sourcerer.io/fame/iamsivab/iamsivab/Machine-Learning-AB-Testing/images/5)](https://sourcerer.io/fame/iamsivab/iamsivab/Machine-Learning-AB-Testing/links/5)[![](https://sourcerer.io/fame/iamsivab/iamsivab/Machine-Learning-AB-Testing/images/6)](https://sourcerer.io/fame/iamsivab/iamsivab/Machine-Learning-AB-Testing/links/6)[![](https://sourcerer.io/fame/iamsivab/iamsivab/Machine-Learning-AB-Testing/images/7)](https://sourcerer.io/fame/iamsivab/iamsivab/Machine-Learning-AB-Testing/links/7)


[![GitHub license](https://img.shields.io/github/license/iamsivab/Machine-Learning-AB-Testing.svg?style=social&logo=github)](https://github.com/iamsivab/Machine-Learning-AB-Testing/blob/master/LICENSE) 
[![GitHub forks](https://img.shields.io/github/forks/iamsivab/Machine-Learning-AB-Testing.svg?style=social)](https://github.com/iamsivab/Machine-Learning-AB-Testing/network) [![GitHub stars](https://img.shields.io/github/stars/iamsivab/Machine-Learning-AB-Testing.svg?style=social)](https://github.com/iamsivab/Machine-Learning-AB-Testing/stargazers) [![GitHub followers](https://img.shields.io/github/followers/iamsivab.svg?label=Follow&style=social)](https://github.com/iamsivab/)
