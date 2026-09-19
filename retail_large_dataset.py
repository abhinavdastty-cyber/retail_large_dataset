import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('C:\\Users\\Presto\\OneDrive\\Desktop\\abhinab\\retail_large_dataset.csv')
num_cols = [
    "age",
    "product_price",
    "quantity",
    "discount_percentage",
    "final_price",
    "delivery_days",
]

summary = df[num_cols].describe().T[["mean", "50%", "std", "min", "max"]]
summary = summary.rename(columns={"50%": "median"})
summary["skewness"] = df[num_cols].skew()
summary["kurtosis"] = df[num_cols].kurtosis()
print(summary)
print("\n")

cat_cols = [
    "product_category",
    "customer_segment",
    "payment_method",
    "return_status",
]
for col in cat_cols:
    print("--- Counts for:", col)
    print(df[col].value_counts())
    print("\n--- Percentages(%)for:", col)
    print((df[col].value_counts(normalize=True) * 100).round(2))
    print("\n")

print("---KEY FINDINGS---")
top_segment = df['customer_segment'].value_counts().index[0]
print("Majority Customer Segment:", top_segment)
top_category = df['product_category'].value_counts().index[0]
print("Highest Category:", top_category)
return_counts = df['return_status'].value_counts(normalize=True) * 100
return_rate = return_counts.get("Returned", 0)
print("Return Rate:", round(return_rate, 2), "%")
print("\n")

print("---IQR OUTLIERS---")
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_fence = Q1 - 1.5 * IQR
    upper_fence = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_fence) | (df[col] > upper_fence)]
    print(col, ":", len(outliers))
print("\n")

print("---Z-SCORE OUTLIERS---")
for col in num_cols:
    z_scores = (df[col] - df[col].mean()) / df[col].std()
    print(col, ":", (z_scores.abs() > 3).sum())

df[num_cols].plot(kind='box', subplots=True, layout=(2, 3), figsize=(10, 6))
plt.tight_layout()
plt.savefig("boxplots.png")
plt.close()

print("\n")
print("-----OBSERVATIONS------")
max_price = df["product_price"].max()
print("Max product price:", max_price)
fp_q1 = df["final_price"].quantile(0.25)
fp_q3 = df["final_price"].quantile(0.75)
fp_iqr = fp_q3 - fp_q1
fp_upper_fence = fp_q3 + (1.5 * fp_iqr)
fp_lower_fence = fp_q1 - (1.5 * fp_iqr)
high_fp_outliers = df[df["final_price"] > fp_upper_fence]
low_fp_outliers = df[df["final_price"] < fp_lower_fence]
print("\n")
print("High price categories:")
print(high_fp_outliers["product_category"].value_counts())
print("\n")
print("Low price categories:")
print(low_fp_outliers["product_category"].value_counts())
print("\n")
print("High average final price categories:")
print(high_fp_outliers["quantity"].mean())
print("\n")
print("Low average final price categories:")
print(low_fp_outliers["quantity"].mean())

del_q1 = df["delivery_days"].quantile(0.25)
del_q3 = df["delivery_days"].quantile(0.75)
del_iqr = del_q3 - del_q1
del_upper_fence = del_q3 + (1.5 * del_iqr)
long_delivery_outliers = df[df["delivery_days"] > del_upper_fence]
print("\n")
print("Long delivery outliers:", len(long_delivery_outliers))
print("\n")

print("--- 1. PEARSON CORRELATION ---")
corr_metrix = df[num_cols].corr()
print(corr_metrix.round(2))
print("\n")

plt.figure(figsize=(8, 6))
sns.heatmap(corr_metrix, annot=True, cmap="Blues", fmt=".2f")
plt.title("Correlation Heatmap")
plt.savefig("heatmap.png")
plt.close()

print("\n")
print("--- 2. CATEGORY-BASED ANALYSIS ---")
print("Average final price by customer segment")
segment_spending = df.groupby("customer_segment")["final_price"].mean().round(2)
print(segment_spending)
print("\n")

returned_items = df[df["return_status"] == "Returned"]
category_returns = (
    returned_items["product_category"].value_counts()
    / df["product_category"].value_counts()
) * 100
print("Returned rate(%) by product category:")
print(category_returns.round(2))
print("\n")

print("Average delivery days returned status:")
delivary_returned = df.groupby("return_status")["delivery_days"].mean().round(2)
print(delivary_returned)
print("\n")

print("--- CORRELATION MATRIX OBSERVATIONS ---")
price_corr = df["product_price"].corr(df["final_price"]).round(2)
qty_corr = df["quantity"].corr(df["final_price"]).round(2)
disc_corr = df["discount_percentage"].corr(df["final_price"]).round(2)
print("Correlation with final price:")
print("product_price", price_corr)
print("quantity", qty_corr)
print("discount_percentage", disc_corr)
delivery_corr = df["delivery_days"].corr(df["final_price"]).round(2)
print("Delivery days vs final price correlation:", delivery_corr)
print("\n")

indep_cols = ["age", "product_price", "quantity", "discount_percentage", "delivery_days"]
indep_corr = df[indep_cols].corr().abs()
high_corr = (indep_corr > 0.80) & (indep_corr < 1)
print("Any Multicollinearity detected(Correlation >0.80)?")
print(high_corr.any().any())
print("\n")

print("--- MULTIVARIATE ANALYSIS ---")
print("\n")
print("--- Category vs Discount vs Final Price ---")
cat_disc_price = df.groupby(["product_category", "discount_percentage"])["final_price"].mean().round(2)
print(cat_disc_price)
print("\n")

print("--- Segment vs Age vs Final Price ---")
seg_age_price = df.groupby(["customer_segment", "age"])["final_price"].mean().round(2)
print(seg_age_price)
print("\n")

if "shipping_type" in df.columns:
    print("--- Shipping Type vs Return Status vs Delivery Days ---")
    shiping_returns = df.groupby(["shipping_type", "return_status"])["delivery_days"].mean().round(2)
    print(shiping_returns)
    print("\n")

print("--- SKEWNESS & KURTOSIS AUTOMATED ANALYSIS ---")
skewness = df[num_cols].skew().round(2)
kurtosis = df[num_cols].kurtosis().round(2)
pos_skew = skewness[skewness > 0.5]
print("Positively skewed columns:")
print(pos_skew)
print("\n")

high_kurtosis = kurtosis[kurtosis > 1]
print("High kurtosis columns:")
print(high_kurtosis)
print("\n")

dist_summary = pd.DataFrame({"Skewness": skewness, "Kurtosis": kurtosis})
print("--- FULL DISTRIBUTION TABLE ---")
print(dist_summary)
print("\n")

print("--- KEY STATISTICAL FINDINGS ---")
print("1. Revenue skewness:", df["final_price"].skew().round(2))
print("Confirms right-skewed revenue distribution")
print("\n")

top_revenue_orders = df[df["final_price"] > fp_upper_fence]
print("2. Total high value transation:", len(top_revenue_orders))
print("Revenue driven by limited high-value transactions")
print("\n")

avg_spending_by_discount = df.groupby("discount_percentage")["final_price"].mean().round(2)
print("3. Average spending by discount percentage")
print(avg_spending_by_discount)
print("Moderate discounts maximize spending")
print("\n")

print("4. Maximum order value:", df["final_price"].max())
print("Outliers represent genuine premium purchases")
print("\n")

avg_delivery_by_return = df.groupby("return_status")["delivery_days"].mean().round(2)
print("5. Average delivery days by return status")
print(avg_delivery_by_return)
print("Return probability increases slightly with longer delivery times")