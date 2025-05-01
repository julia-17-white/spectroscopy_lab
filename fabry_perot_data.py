import matplotlib.pyplot as plt
import pandas as pd
import scipy

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
plt.xlabel('time (s)')
plt.ylabel('voltage')
# plt.plot(df['x-axis'], df['1'])


#################################################################
# I will now include various methods for finding the spacing b/w the peaks
# they should move from less complex to more complex

#################################################################
# using the scipy peak finding algorithm to locate the peaks
peak_width = 10
peaks = scipy.signal.find_peaks(df['2'], width=peak_width)
print(peaks[1])
plt.figure()
plt.plot(df['x-axis'], df['2'])
plt.scatter(df['x-axis'][peaks[0] + 8], peaks[1]['prominences'] + 0.36, c = 'm', s=15)
# the +8 is to account for the NaN indices which were removed
# the +36 is a y-axis normalization
plt.xlabel('time (s)')
plt.ylabel('voltage')


#################################################################
# now i'll take the two middle peaks and calculate delta t with them
time_diff = df['x-axis'][peaks[0][int(len(peaks[0])/2)] + 8] - df['x-axis'][peaks[0][int(len(peaks[0])/2) - 1] + 8]
print(f'the time difference is {time_diff:.4f} seconds')

#################################################################
# instead i can plot the times vs frequency and take the linear part of the polynomial
plt.figure()
plt.scatter(df['x-axis'][peaks[0][2:-1]], peaks[1]['prominences'][2:-1] + 0.36, s=15)





plt.show()
