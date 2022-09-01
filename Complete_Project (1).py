#!/usr/bin/env python
# coding: utf-8

# > **Tip**: Welcome to the Investigate a Dataset project! You will find tips in quoted sections like this to help organize your approach to your investigation. Once you complete this project, remove these **Tip** sections from your report before submission. First things first, you might want to double-click this Markdown cell and change the title so that it reflects your dataset and investigation.
# 
# # Project: Investigate The - [Movies Dataset]
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# ### Dataset Description 
# 
# > **Tip**: This data set contains information about 10,000 movies collected from The Movie Database (TMDb), including user ratings and revenue.
# 
# 
# ### Question(s) for Analysis
# >**Tip**: Some of the Questions that we might ask about this data would be : 
# - What is the most common genre in this data set ?
# - What Genres provide the highest revenue ?
# - What Genres provide the highest profit ?
# - What are the top profitable movies ?
# - What is the year with the most movies produced ?
# - What months have the most and least movie production ?
# - Which directors directed the highest number of movies?
# - What are the characteristics of a good movie?
# - What movie has the highest\lowest ratings ?
# - Dose The run time of movies changed overtime (Years) ? What factors affect the incline\decline in the runtime ?
# 

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.stats import trim_mean


# In[2]:


# Upgrade pandas to use dataframe.explode() function. 
get_ipython().system('pip install --upgrade pandas==0.25.0')


# In[60]:


get_ipython().system('pip install -U seaborn')


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# > **Tip**: In this section of the report, you will load in the data, check for cleanliness, and then trim and clean your dataset for analysis. Make sure that you **document your data cleaning steps in mark-down cells precisely and justify your cleaning decisions.**
# 
# 
# ### General Properties
# > **Tip**: You should _not_ perform too many operations in each cell. Create cells freely to explore your data. One option that you can take with this project is to do a lot of explorations in an initial notebook. These don't have to be organized, but make sure you use enough comments to understand the purpose of each code cell. Then, after you're done with your analysis, create a duplicate notebook where you will trim the excess and organize your steps so that you have a flowing, cohesive report.

# In[3]:


# Load your data and print out a few lines. Perform operations to inspect data
#   types and look for instances of missing or possibly errant data.
pd.options.display.max_rows = 9999
mov=pd.read_csv('tmdb-movies.csv')


# In[4]:


#checking the first rows of data
mov.head()


# In[5]:


#Coloumns of the Data set
mov.columns


# In[6]:


#Look at the types of Data and the missing col
mov.info()


# Notice That there are a lot of missing data in columns (homepage,tagline,keywords,production companies) some of those rows we actually don't need so we might drop them in the cleaning #Reminder for me

# In[7]:


#How many rows and columns in the dataset
mov.shape


# In[8]:


#Describe the data set
mov.describe()


# 
# ### Data Cleaning
# > **Tip**: Make sure that you keep your reader informed on the steps that you are taking in your investigation. Follow every code cell, or every set of related code cells, with a markdown cell to describe to the reader what was found in the preceding cell(s). Try to make it so that the reader can then understand what they will be seeing in the following cell(s).
#  

# In[9]:


#First we don't need the columns homepage,tagline,keywords,overview,imbd_id in our analysis so we are dropping them
mov.drop(['imdb_id','cast','homepage','tagline','keywords','overview','production_companies'],inplace=True,axis=1)


# In[10]:


#check for columns
mov.columns


# __Missing Values Treatment__ : First we need to check using info

# In[11]:


mov.info()


# __There are 44 movies that dosen't have the director name so we are droping them as they are segnificant & There are 33 rows in genres that we are droping too , we can drop null values if they aren't going to affect our analysis if it small number of rows like 33 and 44 from 10000 rows__

# In[12]:


#Handlling missing Values
mov.dropna(inplace=True)
#this is done to make the indecies back from 0-n
mov.reset_index(drop=True,inplace=True)


# In[13]:


#check
mov.info()


# __Fixing The Data Types__ : budget, Revenue and Runtime columns are int but they should be float as it is an amount that might have percision and release_date column should be date 

# In[14]:


mov[['budget','revenue','runtime']]=mov[['budget','revenue','runtime']].astype(float)


# In[15]:


mov['release_date']=pd.to_datetime(mov['release_date'])


# In[16]:


#Check
mov.info()


# In[17]:


#check for duplicates 
mov.duplicated().sum()


# In[18]:


#we have one duplicated row so we are dropping it
mov.drop_duplicates(inplace=True)


# In[19]:


#check
mov.duplicated().sum()


# In[20]:


mov['revenue'].mode()
mov['budget'].mode()


# All the coloumns names are not capitalized and consistant, all the data types are fixed except the release_year and i want it to be int, no missing data , no duplicates  but there is somthing in the revenue and budget column which is going to affect our analysis we have a lot of zeroes in both columns so we might take the median or the mean and fill in the missing data
# the mode is zeros so the we cannt take the mode

# In[21]:


rev_col=mov['revenue']
rev_col.replace(to_replace = 0, value = rev_col.mean(), inplace=True)
budget_col=mov['budget']
budget_col.replace(to_replace = 0, value = budget_col.mean(), inplace=True)


# In[22]:


#check
mov[mov['revenue']<=0.0]
mov[mov['budget']<=0.0]
#it returns empty data frame we are Good now !


# # Some Data Visualiztion to under stand the data

# In[48]:


mov['revenue'].plot(kind='hist')


# In[49]:


mov['budget'].plot(kind='hist')


# # Questions

# ### What is the most common genre in this data set ?

# In[23]:


mov['genres']


# __First we need to get the first genre of each movie as The genres column contian multiple values for one movie separated by "|"
# so we will take only the first genre__
# 

# In[24]:


mov['genre_1']=mov['genres'].apply(lambda text: text.split('|')[0])
#now we don't need the generes column so i am dropping it
mov.drop('genres',axis=1,inplace=True)
#check
mov.columns


# In[25]:


#get the most common genre how many movies have this genre
mov.groupby('genre_1')['original_title'].count().sort_values(ascending=False)


# __It looks like Drama is the highest genre in our data
# so lets visualize and see this with our eyes better than numbers__

# In[26]:


mov_genre=mov.groupby('genre_1')['original_title'].count().sort_values(ascending=False)
mov_genre.plot(kind='bar',title='Most common genres')
plt.xlabel('Genre')
plt.ylabel('Number of movies under genre')


# __Now  we know that Drama has the highest amount of movies but we want to know more Does drama provide the highest amount of money or in another words__

# # What genre provides the highest revenue ?

# In[27]:


mov_rev=mov.groupby('genre_1')['revenue'].sum().sort_values(ascending=False)
mov_rev


# __What a surprise although Drama was the most common genre it is not the highest revenue , Action is the highest one !
# Lets better see this with our eyes to believe__

# In[28]:


mov_rev.plot(kind='bar',title='Genres With The Highest Revenues')
plt.xlabel('Genre')
plt.ylabel('Revenue in hundred billions')


# __Ok now let's search for more surprises lets see if action is also the most profitable movie or is another genre__ 

# # What is the most profitable genre ?

# In[29]:


#We need a profit coloumn to answer this question
mov['profit']=mov['revenue'].sub(mov['budget'])


# In[30]:


profit_per_gen=mov.groupby('genre_1')['profit'].sum().sort_values(ascending=False)
profit_per_gen


# __Ok no more surprises action is the most profitable genre as we expected !__

# In[31]:


profit_per_gen.plot(kind="bar",title='Most Profitable Genres')
plt.xlabel('Genre')
plt.ylabel('Profit in Ten billions')


# __Now lets see the most profitable movies i think they might be in the action genre or maybe comedy lets see__

# # What are the top profitable movies ?

# In[32]:


top_profitable_mov=mov[['original_title','genre_1','profit','budget','vote_average','popularity','runtime','director']].sort_values(by='profit',ascending=False)
top_15_mov=top_profitable_mov[0:15]
top_7_mov=top_profitable_mov[0:7]
top_7_mov


# __Looks like Avatar movie is killing it far from the seconed best movie by more than 0.3 billion dollars lets visualize the top ten movies of all time according to our data set__

# In[33]:


sns.barplot(x="original_title",y='profit',data=top_7_mov,hue="genre_1")
sns.set(rc={"figure.figsize":(30, 10)})
plt.title('Top 15 Movies With Respect to Profit')
plt.xlabel('Movie Title')
plt.ylabel('Profit in Billions')
plt.show()


# __We See that avatar is the highest movie that generated more than 2.5 billion Dollars profit That's alot of money i mean 10 zeros , Great job !! , We Could see also why the action is the most profitable : 4 out of the top 7 movies are in the action genre but how are those movies so successful we need to know a reason__

# # What are the characteristics of the Most successful Movies ? 

# In[34]:


#First lets get the mean profit of all movies
mean_profit=mov['profit'].mean()
mean_profit
#40 Million is a big number , it might be the outliers so lets clear the top 5 movies and try again
mean_trim=trim_mean(mov['profit'],0.05)
mean_trim


# __Now 26 million Dolars is a more reasonable than 40 Million Dolars for a movie to be successful we need to know what movies are higher than this number and then identify their characteristics__

# In[35]:


successful_movies=mov[mov["profit"] >=mean_trim]
successful_movies.shape


# __There are 2823 successful movie that's a Good number out of the 10k movies according to our standards which is the profit so what is the mean ,median budget spent on a Good movie and what is the average Rating of those movies and popularity lets check those questions__

# # What is the average budget and the median budget for a good movie ?

# In[36]:


mean_bud=trim_mean(successful_movies['budget'],0.05)
median_bud=successful_movies['budget'].median()
mean_bud
median_bud


# __We are talkking about 30 Million Dolar budget for a good movie according to the mean and nearly 15 million according to the median What about the Top 5 movies what was their budget__

# In[37]:


budget_of_top_5=top_7_mov[0:5][['original_title','budget']]
mean_5=budget_of_top_5.mean()
mean_5
budget_of_top_5.plot(kind='bar',x='original_title')
plt.title('Top 5 Movies With Respect to budget')
plt.xlabel('Movie Title')
plt.ylabel('budget in 100 Million')


# __To be successful is good but to be in the Top 5 you need to spend about 195400000 Dollars That's the mean for the Top Films,
# Cost of Reaching the peak Right :) ? Thats why we needed to exclude those movies from the mean calculations__

# # What is the Average Rating for a Successful movie and a Top movie  ?

# In[38]:


average_rating=trim_mean(successful_movies['vote_average'],0.05)
average_rating


# __You Need To have higher than 6.1 Average Rating in order to be successful movie What about the Top movies__

# In[39]:


Top_avg_rating=top_7_mov[0:5]['vote_average'].mean()
Top_avg_rating


# In[104]:


sns.barplot(x=top_7_mov['original_title'],y=top_7_mov['vote_average'],data=top_7_mov)
plt.show()


# __You Need To have higher than 7.1 Average Rating in order to be in Top movies__

# __Popularity of the successful and the Top movies :__

# In[40]:


trim_mean(successful_movies['popularity'],0.05)


# __You Need on average 1.1 popularity in order to be successful__

# In[41]:


top_7_mov['popularity'].mean()


# __You Need on average 11.5 popularity in order to be at the peak__

# # What movie has the highest-lowest ratings ?

# In[126]:


highest_rated_movie=mov[mov['vote_average']==mov['vote_average'].max()][['original_title','vote_average','budget','profit']]
lowest_rated_movie=mov[mov['vote_average']==mov['vote_average'].min()][['original_title','vote_average','budget','profit']]
highest_rated_movie


# The highest rated movie is : The Story of Film: An Odyssey

# In[125]:


lowest_rated_movie


# The lowest rated movies are : Transmorphers	, Manos: The Hands of Fate

# # Who directed the Most successful movies ?

# In[42]:


mov['1st_director']=mov['director'].apply(lambda text: text.split('|')[0])
mov['1st_director']


# In[43]:


top_15_mov.groupby('director')['original_title'].count().sort_values(ascending=False)


# __We have Two Directors that seems to be peak persons which are Joss Whedon and James Cameron , They Directed not 1 but 2 of the most successful movies of all time according to our Data__

# # Who Directed the Most Number of movies ?
# 

# In[143]:


top_dir=mov.groupby('1st_director')['original_title'].count().sort_values(ascending=False)

#Woody allen directed the most movies lets see that through viz
top_dir[0:5].plot(kind='bar',title='Most Number of Movies for a Director')
plt.xlabel('Director')
plt.ylabel('Count of Movies Directed')


# In[87]:


xd=mov['budget']
yd=mov['revenue']
sns.regplot(x=xd, y=yd);
plt.title("Scatter plot between two variables budget and revenue") 


# __There is a postive relationship between the budget and profit that insures our analyis as the Top 10 movies budget were very far from the mean__

# In[88]:


np.corrcoef(xd, yd)


# In[92]:


xa=mov['vote_average']
ya=mov['profit']
sns.regplot(x=xa, y=ya,logx=True);
plt.title("Scatter plot between two variables vote_average and profit") 


# __Seems like those with vote average higher than seven have got a better chance to be at the peak of the movies with respect to profit , but it is not neccessarly that if you have the highest Vote you will have the highest profit as we saw up in the average voting barplot but you must have higher than 6.5 in order to have that high profit__

# # What is the year with the most movies produced ?

# In[94]:


mov.groupby('release_year')['original_title'].count().sort_values(ascending=False).plot(kind="bar",title="Years with the most movie production ")
plt.xlabel('Year')
plt.ylabel('Number of movies produced')
plt.show()


# __it seems that the production of movies reach its peak in 2014 with almost 700 movies produced, this is ordaniry as the movie bussniess is becoming widely known year by year__

# __What months have the most and least movie production ?__

# In[103]:


mov['month']=mov['release_date'].dt.strftime('%b')
data_by_month=mov.groupby('month')['original_title'].count().sort_values(ascending=False)
data_by_month.plot(kind='bar',title='Highest Release Month')
plt.show()


# __September is the highest month that movies are released in by over 1200 movies released in this month,
# his is primarily so because, during the summer months of Sep, movies are promoted as events
# Schools are closed, children are looking for entertainment and parents are also less fatigued as many take vacations. If the cinema halls will be full, naturally, the producers and distributors will make more money.__

# # What is the mean Run time of Movies ?

# In[137]:


runtime=mov[mov['runtime']>40]['runtime']
mean_runtime=runtime.mean()
mean_runtime


# In[142]:


runtime.hist(bins=[0,25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,400])
plt.vlines(x = mean_runtime, ymin = 0,ymax=8000,
           colors = 'purple',
           label = 'vline_multiple - full height',
           linestyles='dashed')
plt.text(70,8100,'Mean run time = 104.2 mins  (1.74 hr)')
plt.title('The Distribution of The Runtime Variable')


# __The Curve is Right skeewed meaning most have a runtime greater tahn zero and 150 Mins and less than 200 Mins __

# __Run Time of The Top 15 Movies :__

# In[139]:


Top_movies_runtime=top_15_mov['runtime'].mean()
Top_movies_runtime


# __Looks like the Top 15 movies are higher than the mean even in the Run time length Not only their profit and Rating__

# In[145]:


mov.plot(x='release_year',y='runtime')
plt.title("Runtime Vs Release Year")


# __The Relation between this two variables is not clear and needs further analysis__

# <a id='conclusions'></a>
# ## Conclusions
# 
# > **Finally**: 
# -What is the most common genre in this data set ?
# 
#   Drama
#   
# __What Genres provide the highest revenue ?__
# 
#   Action
#   
# __What Genres provide the highest profit ?__
# 
#   Action
#   
# __What are the top profitable movies ?__
# 
#    Avatar, Star Wars:The Force Awakens ,Titanic, Jurassic world, furious 7 , Harry Potter and the Deathly Hallows:Part 2 , The Avengers		
#     
# __What is the year with the most movies released ?__
# 
# 2014 Was the highest Release year for films and this is simply because the movie bussiness is getting bigger by time but this needs further analysis .
# 
# __What months have the most and least movie production ?__
# 
# September , because this time of the year is vacation for both childrens and adults coming back from their jobs and is last month in summer so people would be less fatuge to go out but we need further anlysis to make sure
# 
# __Which directors directed the highest number of movies?__
# 
# Woody Allen                          46 Movies
# Clint Eastwood                       34 Movies
# Martin Scorsese                      30 Movies
# Steven Spielberg                     29 Movies
# Ridley Scott                         23 Movies
# Ron Howard                           22 Movies
# Steven Soderbergh                    22 Movies
# 
# __What are the characteristics of a good movie?__
# 
#   They have mean profit of 26 million or higher 30 Million Dolars budget or more ,You Need To have higher than 6.1 Average       Rating in order to be a successful movie and more Than 1.1 popularity but Great movies have other specs that i mentioned in     the markk down cells up-top
#   
# __What is the mean Run time of Movies ?__
# 
#  104.2 Mins , and the movies runtime lies between 60 to 150 mins or less than 200 mins
# 
# __Did the mean time of movies changed over time ?__
#  This needs further analysis
# __What movie has the highest\lowest ratings ?__
# 
#  The highest rated movie is : The Story of Film: An Odyssey
#  The lowest rated movies are : Transmorphers , Manos: The Hands of Fate
# 

# In[ ]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])

