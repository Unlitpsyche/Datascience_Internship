import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from csv import reader
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# --- Problem 1: Read the data and remove all null or empty values. ---
print("--- Problem 1: Reading and Cleaning Data ---")

# Load the dataset
# The data is in a CSV where each row is a transaction with items separated by commas.
# We need to read it row by row and handle potential empty strings/NaN values.
transactions_list = []
with open('D:\studies\Programs\LabAssignment_Day15\LabAssignment_Day15\Dataset_Day15.csv', 'r') as file:
    csv_reader = reader(file)
    for row in csv_reader:
        # Filter out any empty strings or 'nan' values from the row
        # Ensure that items are stripped of leading/trailing whitespace
        cleaned_row = [item.strip() for item in row if item.strip() and item.strip().lower() != 'nan']
        if cleaned_row: # Only add non-empty transactions
            transactions_list.append(cleaned_row)

print(f"Original number of transactions: {len(transactions_list)}")
# Note: The problem statement implies removing null/empty *values within* transactions,
# not necessarily removing empty transactions themselves if they result from cleaning.
# The cleaning step above filters out empty items within a transaction.
# If a row becomes entirely empty after cleaning, it is not added to transactions_list.
print("Null or empty values within transactions have been removed.")
print(f"Number of transactions after cleaning: {len(transactions_list)}")
print("\n")

# --- Problem 2: Transform the data using TransactionEncoder to perform Market Basket Analysis. ---
print("--- Problem 2: Data Transformation using TransactionEncoder ---")

# Initialize TransactionEncoder
encoder = TransactionEncoder()

# Fit and transform the transactions into a one-hot encoded boolean DataFrame
onehot_encoded_data = encoder.fit(transactions_list).transform(transactions_list)

# Convert the boolean array into a Pandas DataFrame
itemsets_df = pd.DataFrame(onehot_encoded_data, columns=encoder.columns_)

print("Data successfully transformed using TransactionEncoder.")
print("First 5 rows of the transformed DataFrame:")
print(itemsets_df.head())
print(f"Shape of the transformed DataFrame: {itemsets_df.shape}")
print("\n")

# --- Problem 3: Use min_support = 0.02 to find frequent itemsets. ---
print("--- Problem 3: Finding Frequent Itemsets (min_support = 0.02) ---")

# Apply the Apriori algorithm to find frequent itemsets
# min_support is set to 0.02 as per the problem description.
frequent_itemsets = apriori(itemsets_df, min_support=0.02, use_colnames=True)

print(f"Number of frequent itemsets found: {len(frequent_itemsets)}")
print("Frequent itemsets (first 10 rows, sorted by support):")
print(frequent_itemsets.sort_values(by='support', ascending=False).head(10))
print("\n")

# --- Problem 4: Use the frequent itemsets to create association rules (take min_threshold as 15%) and evaluate the rules for the following metrics ---
print("--- Problem 4: Creating Association Rules (min_threshold = 15% confidence) ---")

# Create association rules using the frequent itemsets
# metric is set to 'confidence' and min_threshold to 0.15 (15%)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.15)

print(f"Number of association rules generated: {len(rules)}")
print("First 5 association rules:")
print(rules.head())
print("\n")

# --- Problem 4 (a): Find top 5 antecedent -> consequent rules based on ‘leverage’, ‘lift’ each. Explain your findings. ---
print("--- Problem 4 (a): Top 5 Rules by Leverage ---")
top_5_leverage_rules = rules.sort_values(by='leverage', ascending=False).head(5)
print("Top 5 antecedent -> consequent rules based on 'leverage':")
print(top_5_leverage_rules)
print("\nExplanation of Leverage:")
print("Leverage quantifies the difference between the observed frequency of A and C appearing together and the frequency expected if A and C were independent. A positive leverage indicates that A and C appear together more often than expected by chance, while a negative leverage indicates they appear together less often. A value of 0 means they are independent.")
print("For example, the top rule with a high leverage suggests that the co-occurrence of the antecedent and consequent is significantly higher than what would be expected if they were independent. This makes them good candidates for joint promotions or product placement.")
print("-" * 50)

print("\nTop 5 Rules by Lift")
top_5_lift_rules = rules.sort_values(by='lift', ascending=False).head(5)
print("Top 5 antecedent -> consequent rules based on 'lift':")
print(top_5_lift_rules)
print("\nExplanation of Lift:")
print("Lift indicates how much more likely the consequent is to be purchased when the antecedent is purchased, compared to when the consequent is purchased independently. A lift value greater than 1 suggests a positive correlation (they are purchased together more often). A value equal to 1 indicates independence, and a value less than 1 indicates a negative correlation (they are purchased together less often).")
print("For example, a rule with a lift of 2.5 means that customers who buy the antecedent are 2.5 times more likely to also buy the consequent than the average customer.")
print("-" * 50)

# --- Problem 4 (b) Find top 2 & bottom 2 antecedent -> consequent rules based on ‘zhang’s metric’. Explain your findings. ---
print("\nProblem 4 (b) Top 2 & Bottom 2 Rules by Zhang's Metric")

# Calculate Zhang's metric
# Zhang's metric formula: (Support(A->B) - (Support(A) * Support(B))) / max(Support(A->B) * (1 - Support(A)), Support(A) * (Support(B) - Support(A->B)))
# Source: AssociationRules_Day15.ipynb
def calculate_zhangs_metric(rule_row):
    support_ab = rule_row['support']
    support_a = rule_row['antecedent support']
    support_b = rule_row['consequent support']

    numerator = support_ab - (support_a * support_b)
    denominator_part1 = support_ab * (1 - support_a)
    denominator_part2 = support_a * (support_b - support_ab)

    # Handle cases where denominator might be zero to avoid division by zero errors
    if denominator_part1 == 0 and denominator_part2 == 0:
        return 0.0 # Or handle as NaN/special case if preferred
    denominator = max(denominator_part1, denominator_part2)

    return numerator / denominator if denominator != 0 else 0.0

# Apply Zhang's metric calculation
# Using .apply() with axis=1 for row-wise application
rules['zhangs_metric'] = rules.apply(calculate_zhangs_metric, axis=1)

print("Zhang's Metric calculated and added to the rules DataFrame.")

top_2_zhangs_rules = rules.sort_values(by='zhangs_metric', ascending=False).head(2)
print("\nTop 2 antecedent -> consequent rules based on 'zhangs_metric':")
print(top_2_zhangs_rules)

bottom_2_zhangs_rules = rules.sort_values(by='zhangs_metric', ascending=True).head(2)
print("\nBottom 2 antecedent -> consequent rules based on 'zhangs_metric':")
print(bottom_2_zhangs_rules)

print("\nExplanation of Zhang's Metric:")
print("Zhang's metric ranges from -1 to 1. A value close to 1 indicates a strong positive association (A and B are highly associated). A value close to -1 indicates a strong negative association (A and B are rarely bought together). A value near 0 suggests independence or a weak association.")
print("For the top rules (positive Zhang's metric), it means the items in the antecedent and consequent are strongly positively correlated. This suggests that if a customer buys the antecedent, they are very likely to buy the consequent, making them good candidates for bundling or cross-promotion.")
print("For the bottom rules (negative Zhang's metric), it means there is a negative or very weak association. These items are rarely bought together, even if their individual popularity might suggest otherwise. This information can be useful for store layout (placing them far apart) or for identifying products that should not be bundled.")
print("-" * 50)

