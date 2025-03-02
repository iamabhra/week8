import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import stemgraphic
import seaborn as sns
from scipy.special import comb  # For combinatorial calculations (see line 209)
from scipy.stats import hypergeom # For dhyper() equivalent (see line 225)
from scipy.stats import binom # For dbinom() equivalent (see line 264)
from scipy.stats import poisson # For dpois() equivalent (see line 304)
from scipy.stats import nbinom # For dnbinom() equivalent (see line 324)
from scipy.stats import geom # For dgeom() equivalent (see line 338)
from scipy.stats import norm # For pnorm() equivalent (see line 349)
from scipy.stats import probplot # For qqnorm() equivalent (see line 367)
from scipy.stats import lognorm # For plnorm() equivalent (see line 429)
from scipy.stats import expon # For pexp() equivalent (see line 438)
from scipy.stats import gamma # For pgamma() equivalent (see line 453)
from scipy.stats import weibull_min # For pweibull() equivalent (see line 462)
import statsmodels.formula.api as sm
from scipy.stats import ttest_1samp
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

#Research Question 1
data = pd.read_csv('/Users/abhijitghosh/Documents/DataScience/IN_chemistry.csv')
columns = ["Secchi", "NO3_epi", "NH3_epi", "Total_Phos_epi"]
df = data[columns].dropna()

X = df[["NO3_epi", "NH3_epi", "Total_Phos_epi"]]
y = df["Secchi"]
# Summary of the model
X = sm.add_constant(X)

# Perform multiple linear regression
model = sm.OLS(y, X).fit()

f_statistic = model.fvalue
f_pvalue = model.f_pvalue
print(f"F-Statistic: {f_statistic}")
print(f"F-Test p-value: {f_pvalue}")
# Display regression results, including F-statistic and p-value
print(model.summary())

fitted_values = model.fittedvalues  # Predicted values (fitted by the model)
residuals = model.resid  # Residuals (actual - predicted)

# Plot residuals vs. fitted values
plt.figure(figsize=(8, 6))
plt.scatter(fitted_values, residuals, alpha=0.7, color='blue')
plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
plt.title('Residuals vs. Fitted Values', fontsize=14)
plt.xlabel('Fitted Values', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

#Assumption q-q plot
X = df[['NO3_epi', 'NH3_epi', 'Total_Phos_epi']]
y = df['Secchi']

# Train a linear regression model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and calculate residualsy
y_pred = model.predict(X_test)
residuals = y_test - y_pred


# Q-Q plot for residuals
plt.figure(figsize=(8, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q-Q Plot of Residuals", fontsize=14)
plt.xlabel("Theoretical Quantiles", fontsize=12)
plt.ylabel("Sample Quantiles", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

#pair plot


# Select relevant columns for the pair plot
columns_of_interest = ['Secchi', 'NO3_epi', 'NH3_epi', 'Total_Phos_epi']

# Create the pair plot
sns.pairplot(data[columns_of_interest], diag_kind='kde', kind='reg', plot_kws={'line_kws': {'color': 'red'}, 'scatter_kws': {'alpha': 0.6}})
plt.suptitle('Pair Plot for Research Question 1: Nutrients vs Water Clarity', y=1.02)
plt.show()

#Research Question 2
columns2 = ["Chlorophyll_a", "Total_Phos_epi", "TKN_epi", "Secchi"]
df2 = data[columns2].dropna()

X2 = df2[["Total_Phos_epi", "TKN_epi", "Secchi"]]
y2 = df2["Chlorophyll_a"]
# Summary of the model
X2 = sm.add_constant(X2)

# Perform multiple linear regression
model2 = sm.OLS(y2, X2).fit()

f_statistic2 = model2.fvalue
f_pvalue2 = model2.f_pvalue
print(f"F-Statistic: {f_statistic2}")
print(f"F-Test p-value: {f_pvalue2}")
# Display regression results, including F-statistic and p-value
print(model2.summary())

fitted_values2 = model2.fittedvalues  # Predicted values (fitted by the model)
residuals2 = model2.resid  # Residuals (actual - predicted)

# Plot residuals vs. fitted values
plt.figure(figsize=(8, 6))
plt.scatter(fitted_values2, residuals2, alpha=0.7, color='blue')
plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
plt.title('Residuals vs. Fitted Values', fontsize=14)
plt.xlabel('Fitted Values', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

#scatter plot

# Define independent variables and dependent variable
independent_vars = ['Total_Phos_epi', 'TKN_epi', 'Secchi']
dependent_var = 'Chlorophyll_a'

# Create scatter plots
plt.figure(figsize=(15, 5))
for i, var in enumerate(independent_vars, 1):
    plt.subplot(1, 3, i)  # Create subplots for each variable
    sns.scatterplot(x=df2[var], y=df2[dependent_var], color='blue', alpha=0.7)
    plt.title(f'Scatter Plot: {var} vs {dependent_var}')
    plt.xlabel(var)
    plt.ylabel(dependent_var)
    plt.grid(True)

# Adjust layout and display the plot
plt.tight_layout()
plt.show()


#Assumption q-q plot
X2 = df2[["Total_Phos_epi", "TKN_epi", "Secchi"]]
y2 = df2['Chlorophyll_a']

# Train a linear regression model
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)
model2 = LinearRegression()
model2.fit(X_train2, y_train2)

# Predict and calculate residualsy
y_pred2 = model2.predict(X_test2)
residuals2 = y_test2 - y_pred2


# Q-Q plot for residuals
plt.figure(figsize=(8, 6))
stats.probplot(residuals2, dist="norm", plot=plt)
plt.title("Q-Q Plot of Residuals", fontsize=14)
plt.xlabel("Theoretical Quantiles", fontsize=12)
plt.ylabel("Sample Quantiles", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


#Research Question 3
nitrate_levels = df['NO3_epi']  # Example nitrate data (mg/L)
ammonia_levels = df['NH3_epi']    # Example ammonia data (mg/L)
secchi_depths = df['Secchi']     # Example Secchi depth data (m)

# Thresholds for suitability
thresholds = {
    "NO3_epi": 10.0,  # Nitrate concentration (mg/L)
    "NH3_epi": 0.5,   # Ammonia concentration (mg/L)
    "Secchi": 1.5     # Minimum Secchi depth (m)
}

# Define directions of comparison for null hypothesis
comparison = {
    "NO3_epi": "≤",  # Nitrate should be less than or equal to the threshold
    "NH3_epi": "≤",  # Ammonia should be less than or equal to the threshold
    "Secchi": "≥"    # Secchi depth should be greater than or equal to the threshold
}

# Perform one-sample t-tests (one-tailed) for the metrics
results = []
for col, threshold in thresholds.items():
    sample = df[col].dropna()
    sample_mean = sample.mean()  # Sample mean as an estimate of population mean
    t_stat, p_value = ttest_1samp(sample, threshold)
    
    # Adjust for one-tailed p-value based on direction of comparison
    if comparison[col] == "≤":
        one_tailed_p = p_value / 2 if t_stat < 0 else 1 - (p_value / 2)
    elif comparison[col] == "≥":
        one_tailed_p = p_value / 2 if t_stat > 0 else 1 - (p_value / 2)
    
    # Conclusion
    conclusion = "Fails to Reject H₀" if one_tailed_p > 0.05 else "Rejects H₀"
    
    results.append({
        "Metric": col,
        "Sample Mean (Estimate of Population Mean)": sample_mean,
        "Threshold": threshold,
        "T-Statistic": t_stat,
        "P-Value (One-Tailed)": one_tailed_p,
        "Conclusion": conclusion
    })

# Create a results DataFrame
results_df = pd.DataFrame(results)
print("Water Quality Analysis Results")
print(results_df)

# Generate Q-Q plots
def qq_plot(df2, title):
    stats.probplot(df2, dist="norm", plot=plt)
    plt.title(f"Q-Q Plot for {title}")
    plt.show()

#Final Visualization

nitrate_threshold = 10
ammonia_threshold = 0.5
secchi_threshold = 1.5

# Evaluate suitability
data['Nitrate_Suitable'] = df['NO3_epi'] <= nitrate_threshold
data['Ammonia_Suitable'] = df['NH3_epi'] <= ammonia_threshold
data['Secchi_Suitable'] = df['Secchi'] >= secchi_threshold

# Count the number of lakes meeting each criterion
suitability_counts = {
    'Nitrate': data['Nitrate_Suitable'].sum(),
    'Ammonia': data['Ammonia_Suitable'].sum(),
    'Secchi': data['Secchi_Suitable'].sum()
}

# Convert to percentages
total_lakes = len(data)

suitability_percentages = {key: (value / total_lakes) * 100 for key, value in suitability_counts.items()}

# Plot the bar chart
plt.figure(figsize=(8, 6))
plt.bar(suitability_percentages.keys(), suitability_percentages.values(), color=['blue', 'green', 'orange'])
plt.ylim(0, 100)
plt.title("Percentage of Lakes Meeting Suitability Criteria", fontsize=14)
plt.ylabel("Percentage of Lakes (%)", fontsize=12)
plt.xlabel("Parameters", fontsize=12)
plt.axhline(50, color='red', linestyle='--', label="50% Threshold")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

