# *** import & display dataframe
# -----------------------------------------
# import package
import pandas as pd
# -----------------------------------------
# create dataframe w/dictionary
dic = {
    "PassengerId": [1, 3, 5],
    "name": ['Braund, Mr. Owen Harris', 'Heikkinen, Miss. Laina', 'Allen, Mr. William Henry'], 
    "sex": ['male', 'female', 'male'],
}
dic_df = pd.DataFrame(dic)
dic_df
# -----------------------------------------
# import dataframe
train_df = pd.read_csv('train.csv')

# advance usage (use chunksize to import large size data, prevent memory error)
train = pd.read_csv('train.csv', chunksize=200)
train_df = pd.DataFrame()
for chunk in train:
    train_df = pd.concat([train_df, chunk], ignore_index=True)
# -----------------------------------------
# crawl dataframe (use pandas to extract table from html of webpage)                                                                                                             
url = 'https://en.wikipedia.org/wiki/History_of_Python'
dfs = pd.read_html(url)
## first table found in html 
dfs[0]
# -----------------------------------------
# view dataframe
train_df.head()
train_df.tail()

# to display all row/columns completely
pd.set_option("display.max_row", None)
pd.set_option("display.max_columns", None)

# view info
train_df.info()

# statistics of numerical variables in dataFrame
train_df.describe()


# *** select & display column
# -----------------------------------------
# choose column
train_df.Cabin
train_df['Cabin']
# -----------------------------------------
# all distinct value in column
train_df['Cabin'].unique()
# -----------------------------------------
# all distinct value and counts in column
train_df['Cabin'].value_counts()
# -----------------------------------------
# all distinct value and counts in column
train_df['Cabin'].value_counts()


# *** process dataframe
# -----------------------------------------
# slice dataframe w/select
train_df[['PassengerId','Cabin']]

# slice dataframe w/loc -> [row range, column range]
train_df.loc[:, 'Name':'Fare']

# add column w/select
train_df['new_col'] = 1

# add column w/loc -> [:, new column name]
train_df.loc[:,'new_col2'] = 2

# delete column
train_df = train_df.drop(columns=['new_col', 'new_col2', 'SibSp', 'Parch'])

# fill empty value with constant value
train_df['Age'] = train_df['Age'].fillna(20)

# replace value with apply & def function
def get_personal_title(s):
    s = s.split(',')[1]
    s = s.split('.')[0]
    s = s.strip()
    return s
train_df['personal_title'] = train_df['Name'].apply(get_personal_title) 

# replace value with apply & lambda function
median_fare = train_df['Fare'].median()
train_df['fare_grade'] = train_df['Fare'].apply(lambda x: 'low' if x < median_fare else 'high')

# replace value with mask function
train_df['chinese_sex'] = train_df['Sex'].mask(train_df['Sex'] == 'male', '男生')
train_df['chinese_sex'] = train_df['chinese_sex'].mask(train_df['Sex'] == 'female', '女生')

# change value type
train_df['Age'] = train_df['Age'].astype(int)

# grouping continuous value to discrete value in new column 
labels = [f'{i}年齡層' for i in ['低', '中', '高']]
train_df['Age_group'] = pd.cut(x=train_df['Age'], bins=3, labels=labels)

# aggregate all values to list in condition with groupby function
train_df.groupby(['Pclass', 'Sex'])['Name'].apply(list).reset_index()

# statistics values in condition with groupby function
train_df.groupby(['Sex', 'Age_group']).count()[['PassengerId']].reset_index()


# *** merge dataframe
# -----------------------------------------
# merge dataframe with concat function (df2 will equal to df1)
train_df = train_df[['PassengerId', 'Name', 'Sex', 'Age']]
upper_df = train_df.loc[0:400, :]
lower_df = train_df.loc[401:, :].reset_index(drop=True)

train_df2 = pd.concat([upper_df, lower_df], ignore_index=True)

# merge dataframe with merge function
left_df = train_df[['PassengerId', 'Name']]
right_df = train_df[['PassengerId', 'Sex', 'Age']]

full_df = pd.merge(left_df, right_df, on="PassengerId")