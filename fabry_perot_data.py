import matplotlib.pyplot as plt
import pandas as pd

# reading in the csv
df = pd.read_csv('scope_1.csv')
print(df.head())
# converting the strings from the csv into numbers
df['x-axis'] = pd.to_numeric(df['x-axis'], errors='coerce')
df['1'] = pd.to_numeric(df['1'], errors='coerce')
df['2'] = pd.to_numeric(df['2'], errors='coerce')
# dropping the rows that have NaN
df = df.dropna(subset=['x-axis', '1', '2'])
print(df.head())


plt.figure()
plt.plot(df['x-axis'], df['2'])
plt.plot(df['x-axis'], df['1'])
plt.show()


#################################################################
# I will now include various methods for finding the spacing b/w the peaks
# they should move from less complex to more complex

#################################################################
