#!/usr/bin/env python
# coding: utf-8

# ## Analyze A/B Test Results
# 
# This project will assure you have mastered the subjects covered in the statistics lessons.  The hope is to have this project be as comprehensive of these topics as possible.  Good luck!
# 
# ## Table of Contents
# - [Introduction](#intro)
# - [Part I - Probability](#probability)
# - [Part II - A/B Test](#ab_test)
# - [Part III - Regression](#regression)
# 
# 
# <a id='intro'></a>
# ### Introduction
# 
# A/B tests are very commonly performed by data analysts and data scientists.  It is important that you get some practice working with the difficulties of these 
# 
# For this project, you will be working to understand the results of an A/B test run by an e-commerce website.  Your goal is to work through this notebook to help the company understand if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.
# 
# **As you work through this notebook, follow along in the classroom and answer the corresponding quiz questions associated with each question.** The labels for each classroom concept are provided for each question.  This will assure you are on the right track as you work through the project, and you can feel more confident in your final submission meeting the criteria.  As a final check, assure you meet all the criteria on the [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).
# 
# <a id='probability'></a>
# #### Part I - Probability
# 
# To get started, let's import our libraries.

# In[1]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)


# `1.` Now, read in the `ab_data.csv` data. Store it in `df`.  **Use your dataframe to answer the questions in Quiz 1 of the classroom.**
# 
# a. Read in the dataset and take a look at the top few rows here:

# In[2]:


df=pd.read_csv("ab_data.csv")
df.head(10)


# b. Use the below cell to find the number of rows in the dataset.

# In[3]:


total_rows=len(df.axes[0])
total_rows


# c. The number of unique users in the dataset.

# In[4]:


df['user_id'].nunique()


# d. The proportion of users converted.

# In[5]:


x=df.groupby('user_id').mean()[['converted']]
x.mean()*100


# e. The number of times the `new_page` and `treatment` don't line up.

# In[6]:


number_of_times= df[( (df['landing_page']== 'new_page') != (df['group']=='treatment') )]
number_of_times.shape[0]


# f. Do any of the rows have missing values?

# In[7]:


df.isna().sum()


# `2.` For the rows where **treatment** is not aligned with **new_page** or **control** is not aligned with **old_page**, we cannot be sure if this row truly received the new or old page.  Use **Quiz 2** in the classroom to provide how we should handle these rows.  
# 
# a. Now use the answer to the quiz to create a new dataset that meets the specifications from the quiz.  Store your new dataframe in **df2**.

# In[8]:


df2=df
df2=df[~( (df['landing_page']== 'new_page') != (df['group']=='treatment') )]


# In[9]:


# Double Check all of the correct rows were removed - this should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]


# `3.` Use **df2** and the cells below to answer questions for **Quiz3** in the classroom.

# a. How many unique **user_id**s are in **df2**?

# In[10]:


df2['user_id'].nunique()


# b. There is one **user_id** repeated in **df2**.  What is it?

# In[11]:


df2[df2.duplicated('user_id')]


# c. What is the row information for the repeat **user_id**? 

# In[12]:


df2[df2['user_id']==773192]


# d. Remove **one** of the rows with a duplicate **user_id**, but keep your dataframe as **df2**.

# In[13]:


df2.drop( 1899 , inplace=True)


# In[14]:


df2[df2['user_id']==773192]


# `4.` Use **df2** in the below cells to answer the quiz questions related to **Quiz 4** in the classroom.
# 
# a. What is the probability of an individual converting regardless of the page they receive?

# In[15]:


converted=df2['converted']
converted.mean()*100


# b. Given that an individual was in the `control` group, what is the probability they converted?

# In[16]:


control = df2[(df2['group']=='control')]['converted']
control.mean()*100


# c. Given that an individual was in the `treatment` group, what is the probability they converted?

# In[17]:


treatment=df2[(df2['group']=='treatment')]['converted']
treatment.mean()*100


# d. What is the probability that an individual received the new page?

# In[18]:


(df2['landing_page']=='new_page').mean()


# In[50]:


(df2['landing_page']=='old_page').mean()


# e. Consider your results from a. through d. above, and explain below whether you think there is sufficient evidence to say that the new treatment page leads to more conversions.

# no there isnt enough evidence because the new page had a lower conversion rate than the old page but the difference is small

# <a id='ab_test'></a>
# ### Part II - A/B Test
# 
# Notice that because of the time stamp associated with each event, you could technically run a hypothesis test continuously as each observation was observed.  
# 
# However, then the hard question is do you stop as soon as one page is considered significantly better than another or does it need to happen consistently for a certain amount of time?  How long do you run to render a decision that neither page is better than another?  
# 
# These questions are the difficult parts associated with A/B tests in general.  
# 
# 
# `1.` For now, consider you need to make the decision just based on all the data provided.  If you want to assume that the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, what should your null and alternative hypotheses be?  You can state your hypothesis in terms of words or in terms of **$p_{old}$** and **$p_{new}$**, which are the converted rates for the old and new pages.

# **null hypothesis
# H0:pnew−pold≤0** 
# 
# 

# 
# The null hype is when the difference between the population conversion rate of users given old or new page is equal or lower than zero
# 

# **Alternative hypothesis H1:pnew−pold>0**

# The null hype is when the difference between the population conversion rate of users given old or new page is equal or lower greater

# `2.` Assume under the null hypothesis, $p_{new}$ and $p_{old}$ both have "true" success rates equal to the **converted** success rate regardless of page - that is $p_{new}$ and $p_{old}$ are equal. Furthermore, assume they are equal to the **converted** rate in **ab_data.csv** regardless of the page. <br><br>
# 
# Use a sample size for each page equal to the ones in **ab_data.csv**.  <br><br>
# 
# Perform the sampling distribution for the difference in **converted** between the two pages over 10,000 iterations of calculating an estimate from the null.  <br><br>
# 
# Use the cells below to provide the necessary parts of this simulation.  If this doesn't make complete sense right now, don't worry - you are going to work through the problems below to complete this problem.  You can use **Quiz 5** in the classroom to make sure you are on the right track.<br><br>

# a. What is the **convert rate** for $p_{new}$ under the null? 

# In[19]:


p_new=df2['converted'].mean()
p_new*100


# b. What is the **convert rate** for $p_{old}$ under the null? <br><br>

# In[20]:


p_old=df2['converted'].mean()
p_old*100


# c. What is $n_{new}$?

# In[21]:


n_new = df2.query("group=='treatment'").shape[0]
n_new


# d. What is $n_{old}$?

# In[22]:


n_old = df2.query("group=='control'").shape[0]
n_old


# e. Simulate $n_{new}$ transactions with a convert rate of $p_{new}$ under the null.  Store these $n_{new}$ 1's and 0's in **new_page_converted**.

# In[23]:


new_page_converted = np.random.choice([1, 0], size=n_new, p = [p_new, (1-p_new) ])
new_page_converted.mean()


# f. Simulate $n_{old}$ transactions with a convert rate of $p_{old}$ under the null.  Store these $n_{old}$ 1's and 0's in **old_page_converted**.

# In[24]:


old_page_converted=np.random.choice([1,0], size=n_old, p=[p_old, (1-p_old)])
old_page_converted.mean()


# g. Find $p_{new}$ - $p_{old}$ for your simulated values from part (e) and (f).

# In[25]:


x=new_page_converted.mean()
y=old_page_converted.mean()
x-y


# h. Simulate 10,000 $p_{new}$ - $p_{old}$ values using this same process similarly to the one you calculated in parts **a. through g.** above.  Store all 10,000 values in a numpy array called **p_diffs**.

# In[26]:


p_diffs = []
for i in range(10000):
    new_page_converted = np.random.choice([1, 0], size=n_new, p=[p_new, (1-p_new)])
    old_page_converted = np.random.choice([1, 0], size=n_old, p=[p_old, (1-p_old)])
    p_diff = new_page_converted.mean()-old_page_converted.mean()
    p_diffs.append(p_diff)
p_diffs=np.array(p_diffs)


# i. Plot a histogram of the **p_diffs**.  Does this plot look like what you expected?  Use the matching problem in the classroom to assure you fully understand what was computed here.

# In[27]:


plt.hist(p_diffs)
plt.show()


# j. What proportion of the **p_diffs** are greater than the actual difference observed in **ab_data.csv**?

# In[28]:


#obs_diff = treatment_prob - control_prob
obs_diff=treatment.mean()-control.mean()
obs_diff


# In[51]:


(p_diffs > obs_diff).mean()


# In[29]:


plt.hist(p_diffs);
plt.axvline(obs_diff, color='red');


# k. In words, explain what you just computed in part **j.**  What is this value called in scientific studies?  What does this value mean in terms of whether or not there is a difference between the new and old pages?

# 
# **it represents the sample and if null hypothesis is true**
# **the J part is the probabilty**

# the p-value is almost 90% and we can see that the new_page isnt doing any better than the old_page 

# this means that we cant reject the null or accept the alternative hypothesis because the p_value is large

# l. We could also use a built-in to achieve similar results.  Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance. Fill in the below to calculate the number of conversions for each page, as well as the number of individuals who received each page. Let `n_old` and `n_new` refer the the number of rows associated with the old page and new pages, respectively.

# In[30]:


import statsmodels.api as sm

convert_old = len(df2.query('converted == 1 & landing_page == "old_page"'))
convert_new = len(df2.query('converted == 1 & landing_page == "new_page"'))
n_old = len(df2.query('landing_page=="old_page"'))
n_new = len(df2.query('landing_page=="new_page"'))


# m. Now use `stats.proportions_ztest` to compute your test statistic and p-value.  [Here](http://knowledgetack.com/python/statsmodels/proportions_ztest/) is a helpful link on using the built in.

# In[57]:


z_score, p_value = sm.stats.proportions_ztest([convert_new,convert_old], [n_new, n_old], alternative ='larger')
print(z_score, p_value)


# n. What do the z-score and p-value you computed in the previous question mean for the conversion rates of the old and new pages?  Do they agree with the findings in parts **j.** and **k.**?

# **it seems that the difference betwwen the lines is 1.31 standaed deviations**

# the p_value is 90% so there isnt enough evidence to reject the null or accept the alternative

# z-score is less than the critical value, so we will have to accept the null hypothesis

# <a id='regression'></a>
# ### Part III - A regression approach
# 
# `1.` In this final part, you will see that the result you acheived in the previous A/B test can also be acheived by performing regression.<br><br>
# 
# a. Since each row is either a conversion or no conversion, what type of regression should you be performing in this case?

# **logistic regression** because Y is one of two possibilities

# b. The goal is to use **statsmodels** to fit the regression model you specified in part **a.** to see if there is a significant difference in conversion based on which page a customer receives.  However, you first need to create a column for the intercept, and create a dummy variable column for which page each user received.  Add an **intercept** column, as well as an **ab_page** column, which is 1 when an individual receives the **treatment** and 0 if **control**.

# In[32]:


def ab_page(x):
    if x=='control':
        return 0
    else:
        return 1                   
        
df2['ab_page']=df2['group'].apply(ab_page)
df2['intercept']=1
df2.head()


# c. Use **statsmodels** to import your regression model.  Instantiate the model, and fit the model using the two columns you created in part **b.** to predict whether or not an individual converts.

# In[33]:


import statsmodels.api as sm
x=df2[['intercept','ab_page']]
y=df2['converted']
model=sm.Logit(y,x)
results=model.fit()


# d. Provide the summary of your model below, and use it as necessary to answer the following questions.

# In[34]:


results.summary()


# e. What is the p-value associated with **ab_page**? Why does it differ from the value you found in **Part II**?<br><br>  **Hint**: What are the null and alternative hypotheses associated with your regression model, and how do they compare to the null and alternative hypotheses in the **Part II**?

# The null hypothesis is that when ab_page = 1, converted = 0

# The alternative hypothesis is that when ab_page = 1, converted = 1

# the p_value is higher but with a small value this is due to the model is trying to predict if the user is going to convert the page

# h0:pnew=pold=0
# H1:pnew−pold>0

# we dont see enough evidence to reject the null

# f. Now, you are considering other things that might influence whether or not an individual converts.  Discuss why it is a good idea to consider other factors to add into your regression model.  Are there any disadvantages to adding additional terms into your regression model?

# it doesnt seem that treatment or control page has such a high influence on the converts so maybe we will need more features

# lets keep in mind that only one feature was helpful so there is no problem with adding a couple of features like the time spent looking at the page

# g. Now along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives. You will need to read in the **countries.csv** dataset and merge together your datasets on the approporiate rows.  [Here](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.join.html) are the docs for joining tables. 
# 
# Does it appear that country had an impact on conversion?  Don't forget to create dummy variables for these country columns - **Hint: You will need two columns for the three dummy variables.** Provide the statistical output as well as a written response to answer this question.

# In[35]:


countries = pd.read_csv('countries.csv')
countries.head()


# In[37]:


df3=pd.merge(df2,countries,on='user_id' )
df3.head()


# h. Though you have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if there significant effects on conversion.  Create the necessary additional columns, and fit the new model.  
# 
# Provide the summary results, and your conclusions based on the results.

# In[38]:


### Create the necessary dummy variables
df3[['CA', 'UK', 'US']] = pd.get_dummies(df3['country'])


# In[39]:


import statsmodels.api as sm
x=df3[['intercept','ab_page','CA', 'UK']]
y=df3['converted']
model=sm.Logit(y,x)
results=model.fit()


# In[40]:


results.summary()


# In[42]:


df3['CA_ab_page'] = df3['CA']*df3['ab_page']
df3['UK_ab_page'] = df3['UK']*df3['ab_page']
df3['US_ab_page'] = df3['US']*df3['ab_page']


# In[44]:


df3.head()


# In[59]:


x=df3[['intercept','CA','UK','ab_page', 'CA_ab_page', 'UK_ab_page']]
y=df3['converted']
model=sm.Logit(y,x)
results=model.fit()


# In[60]:


results.summary()


# Based on the above the canadians would be more likely to convert with 1.08 

# this doesnt hold much practical data, so it holds a statistical significance so we still dont have enough evidence to reject the null hypothesis

# there is no need to switch to the new page as the old page is doing well

# <a id='conclusions'></a>
# ## Conclusions
# 
# based on the p_values it doesnt appear that the countries have influenced the conversion. therefore we cant reject the null on the given data.
# 
# it was found that this wasnt dependent on countries with conversion rates because even if CA is little bit high but US and UK are the same in the terms of the conversion rates once again on the given data we cant reject the null hypothesis
# 
# ### Gather Submission Materials
# 
# Once you are satisfied with the status of your Notebook, you should save it in a format that will make it easy for others to read. You can use the __File -> Download as -> HTML (.html)__ menu to save your notebook as an .html file. If you are working locally and get an error about "No module name", then open a terminal and try installing the missing module using `pip install <module_name>` (don't include the "<" or ">" or any words following a period in the module name).
# 
# You will submit both your original Notebook and an HTML or PDF copy of the Notebook for review. There is no need for you to include any data files with your submission. If you made reference to other websites, books, and other resources to help you in solving tasks in the project, make sure that you document them. It is recommended that you either add a "Resources" section in a Markdown cell at the end of the Notebook report, or you can include a `readme.txt` file documenting your sources.
# 
# ### Submit the Project
# 
# When you're ready, click on the "Submit Project" button to go to the project submission page. You can submit your files as a .zip archive or you can link to a GitHub repository containing your project files. If you go with GitHub, note that your submission will be a snapshot of the linked repository at time of submission. It is recommended that you keep each project in a separate repository to avoid any potential confusion: if a reviewer gets multiple folders representing multiple projects, there might be confusion regarding what project is to be evaluated.
# 
# It can take us up to a week to grade the project, but in most cases it is much faster. You will get an email once your submission has been reviewed. If you are having any problems submitting your project or wish to check on the status of your submission, please email us at dataanalyst-project@udacity.com. In the meantime, you should feel free to continue on with your learning journey by beginning the next module in the program.

# In[ ]:




