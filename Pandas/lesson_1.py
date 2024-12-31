# import library
import pandas as pd
# import sklearn

# dataframe build example
df1 = pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'],
              'Sue': ['Pretty good.', 'Bland.']},
             index=['Product A', 'Product B'])

# dataframe series example
series = pd.Series([30, 35, 40], index=['2015 Sales', '2016 Sales', '2017 Sales'], name='Product A')
# print(df1)
# print(series)

path = "C:/Users/johnf/"
data = pd.read_csv(path + "data.csv")

# data.shape
# output (observations, variables)
# data.head() # out put first 5 rows
