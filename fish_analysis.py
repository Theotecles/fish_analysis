# IMPORT PACKAGES
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
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

# CONVERT STRINGS TO NUMBERS
species_mapping = {
    'Bream': 0, 
    'Roach': 1,
    'Whitefish': 2,
    'Parkki': 3,
    'Perch': 4,
    'Pike': 5,
    'Smelt': 6,
}

fishdf.Species = [species_mapping[item] for item in fishdf.Species]

# SPLIT DATA INTO TRAIN AND TEST SETS
y = fishdf['Weight']
X = fishdf.drop('Weight', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# FIT THE MODEL
est = sm.OLS(y_train, X_train)
est2 = est.fit()

# FIND MODEL RESULTS
print(est2.summary())

# USING A CUT OFF OF .05 FOR THE P VALUE
# SPLIT DATA INTO TRAIN AND TEST SETS
y = fishdf['Weight']
X = fishdf.drop(['Weight', 'Length2', 'Width'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# FIT THE MODEL
est = sm.OLS(y_train, X_train)
est2 = est.fit()

# FIND MODEL RESULTS
print(est2.summary())

# REFIT THE MODEL USING SKLEARN
fish_model = LinearRegression()
fish_model.fit(X_train, y_train)

# CREATE CALCULATE_RESIDUALS
def calculate_residuals(model, features, label):
    """
    Creates predictions on the features with the model and calculates residuals
    """
    predictions = model.predict(features)
    df_results = pd.DataFrame({'Actual': label, 'Predicted': predictions})
    df_results['Residuals'] = abs(df_results['Actual']) - abs(df_results['Predicted'])
    
    return df_results

# CREATE LINEAR_ASSUMPTION
def linear_assumption(model, features, label):
    """
    Linearity: Assumes that there is a linear relationship between the predictors and
               the response variable. If not, either a quadratic term or another
               algorithm should be used.
    """
    print('Assumption 1: Linear Relationship between the Target and the Feature', '\n')
        
    print('Checking with a scatter plot of actual vs. predicted.',
           'Predictions should follow the diagonal line.')
    
    # Calculating residuals for the plot
    df_results = calculate_residuals(model, features, label)
    
    # Plotting the actual vs predicted values
    sns.lmplot(x='Actual', y='Predicted', data=df_results, fit_reg=False, height=7)
        
    # Plotting the diagonal line
    line_coords = np.arange(df_results.min().min(), df_results.max().max())
    plt.plot(line_coords, line_coords,  # X and y points
             color='darkorange', linestyle='--')
    plt.title('Actual vs. Predicted')
    plt.show()

# TEST TO SEE IF THE MODEL IS LINEAR
linear_assumption(fish_model, X_train, y_train)

# CREATE NORMAL_ERRORS_ASSUMPTION
def normal_errors_assumption(model, features, label, p_value_thresh=0.05):
    """
    Normality: Assumes that the error terms are normally distributed. If they are not,
    nonlinear transformations of variables may solve this.
               
    This assumption being violated primarily causes issues with the confidence intervals
    """
    from statsmodels.stats.diagnostic import normal_ad
    print('Assumption 2: The error terms are normally distributed', '\n')
    
    # Calculating residuals for the Anderson-Darling test
    df_results = calculate_residuals(model, features, label)
    
    print('Using the Anderson-Darling test for normal distribution')

    # Performing the test on the residuals
    p_value = normal_ad(df_results['Residuals'])[1]
    print('p-value from the test - below 0.05 generally means non-normal:', p_value)
    
    # Reporting the normality of the residuals
    if p_value < p_value_thresh:
        print('Residuals are not normally distributed')
    else:
        print('Residuals are normally distributed')
    
    # Plotting the residuals distribution
    plt.subplots(figsize=(12, 6))
    plt.title('Distribution of Residuals')
    sns.distplot(df_results['Residuals'])
    plt.show()
    
    print()
    if p_value > p_value_thresh:
        print('Assumption satisfied')
    else:
        print('Assumption not satisfied')
        print()
        print('Confidence intervals will likely be affected')
        print('Try performing nonlinear transformations on variables')

# TEST FOR NORMALITY OF THE ERROR TERMS
normal_errors_assumption(fish_model, X_train, y_train)

# CREATE HOMESCEDASTICITY_ASSUMPTION
def homoscedasticity_assumption(model, features, label):
    """
    Homoscedasticity: Assumes that the errors exhibit constant variance
    """
    print('Assumption 5: Homoscedasticity of Error Terms', '\n')
    
    print('Residuals should have relative constant variance')
        
    # Calculating residuals for the plot
    df_results = calculate_residuals(model, features, label)

    # Plotting the residuals
    plt.subplots(figsize=(12, 6))
    ax = plt.subplot(111)  # To remove spines
    plt.scatter(x=df_results.index, y=df_results.Residuals, alpha=0.5)
    plt.plot(np.repeat(0, df_results.index.max()), color='darkorange', linestyle='--')
    ax.spines['right'].set_visible(False)  # Removing the right spine
    ax.spines['top'].set_visible(False)  # Removing the top spine
    plt.title('Residuals')
    plt.show()

homoscedasticity_assumption(fish_model, X_train, y_train)

# GET THE RMSE OF THE TRAINING AND TEST DATA
train_residuals = calculate_residuals(fish_model, X_train, y_train)
train_residuals_squared = (train_residuals['Residuals']) ** 2
train_mse = np.mean(train_residuals_squared)
train_rmse = np.sqrt(train_mse)
print(train_rmse)

test_residuals = calculate_residuals(fish_model, X_test, y_test)
test_residuals_squared = (test_residuals['Residuals']) ** 2
test_mse = np.mean(test_residuals_squared)
test_rmse = np.sqrt(test_mse)
print(test_rmse)
