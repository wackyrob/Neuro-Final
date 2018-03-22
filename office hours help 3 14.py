
# coding: utf-8

# In[3]:


get_ipython().magic(u'matplotlib inline')
import numpy as np
import pandas as pd


# In[2]:


#importing the data on schizophrenia from the cross study 
#"Identification of risk loci with shared effects on five major psychiatric disorders: a genome-wide analysis."
#this study looks at five diseases, but I'm just using schizophrenia as an example, we can 
#compare the diseases later and also incorporate data from other papers if we like
df=pd.read_csv('/home/nbuser/library/schizophrenia.csv')
df.head()


# In[4]:


df.iloc[:, 0:8]
df.head()
#this dropped the columns with data that are not particularly useful to us 


# In[5]:


df['log_or'] = np.log10(df['oddsratio'])
df.hist('log_or',bins=100)
#this displayes the odds ratio in a nice histogram 
#https://youtu.be/09ZaCKfdwzM for info on what the odds ratio shows
#basically we care about values that are not close to 1 (log of 1 is zero, like on histogram)


# In[6]:


data ={}
for i in range(1,17):
    data[i]=df[:][df['hg18chr']==i]
#now we can select data by what chromosome the snp is on, otherwise there is too much data and we end up with crowded plots
#we can use other characteristics as well 


# In[7]:


data[1]
#this is just the data from snps on the first chromosome


# In[8]:


data[16].plot.scatter(y='pval',x='log_or',s=0.1)
#this uses data from the 16th chomosome as an example
#plotting the log of odds ratio vs p value
#we are able to see that most snps are not significant (p<0.05) and have no association with schizophrenia


# In[9]:


df.plot.scatter(y='pval',x='log_or',s=0.1)
#this is the data for all chromosomes, seems we could do some PCA or clustering?


# In[10]:


df_extreme=df[(abs(df['log_or'])>1.2) & (df['pval']<0.1)]
df_extreme.head()
#df_extreme is snps that are more significant and associated with disease
#these are the snps we would want to examine 


# In[11]:


df_extreme.plot.scatter(y='pval',x='log_or',s=0.1)


# In[12]:


df_extreme.plot.scatter(y='pval',x='oddsratio',s=0.1)

