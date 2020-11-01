import pandas as pd
students = [('jack', 34, 'Sydeny', 'Australia'),
            ('Riti', 30, 'Delhi', 'India'), ('Vikas', 31, 'Mumbai', 'India'),
            ('Neelu', 32, 'Bangalore', 'India'),
            ('John', 16, 'New York', 'US'), ('Mike', 17, 'las vegas', 'US')]
df = pd.DataFrame(students,
                  columns=['Name', 'Age', 'City', 'Country'],
                  index=['a', 'b', 'c', 'd', 'e', 'f'])
df
col_names=df.columns.values
col_names
df.columns
list_cols=list(col_names)
list_cols
index_names=df.index.values
df.index
index_names
