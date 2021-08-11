# IMPORT PACKAGES
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# IMPORT DATA
fishdf = pd.read_csv("D:\KaggleData\Data\ish.csv")

# LOOK AT THE TOP 5 AND BOTTOM 5 ROWS OF THE DATA SET
print(fishdf.head())
print(fishdf.tail())

# CHECK THE DATA TYPES
print(fishdf.dtypes)

# TOTAL ROWS AND COLUMNS
print(fishdf.shape)

# CHECK FOR DUPLICATE ROWS
duplicatedf = fishdf[fishdf.duplicated()]
print("Number of duplicate rows:", duplicatedf.shape)
# NO DUPLICATES

# CHECK FOR MISSING OR NULL VALUES
print(fishdf.isnull().sum())
# NO NULLS

# TAKE OUT NON NUMERICAL DATA
fish_num = fishdf.drop(['Species'], axis=1)

# SET UP SNS STANDARD AESTHETICS FOR PLOTS
sns.set()

# MAKE BOXPLOTS FOR NUMERICAL DATA
for column in fish_num:
    sns.boxplot(x=fish_num[column])
    plt.show()

# SET BIN ARGUMENT
n_obs = len(fish_num)
n_bins = int(round(np.sqrt(n_obs), 0))
print(n_bins)

# CREATE HISTOGRAMS FOR EACH COLUMN
for column in fish_num:
    plt.hist(fish_num[column], density=True, bins=n_bins)
    plt.title(f"{column}")
    plt.show()

# CREATE A CORRELATION MATRIX
c = fish_num.corr()
print(c)

# CREATE SCATTERPLOTS FOR EACH VARIABLE AS WELL
for column in fish_num:
    plt.scatter(fish_num[column], fish_num['Weight'])
    plt.title(f"Weight by {column}")
    plt.show()

# FIND SUMMARY STATISTICS
print(fish_num.describe())

# SPLIT DATA INTO TRAIN AND TEST SETS