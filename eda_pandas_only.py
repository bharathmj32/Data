"""All Pandas EDA examples, runnable without PySpark.

Run: python eda_pandas_only.py
Requires: pip install pandas numpy matplotlib seaborn
Run from the folder containing sales_eda_practice.csv.
"""

# 0. Setup
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", 30)
sns.set_theme(style="whitegrid")

# Change ONLY this location + the column mapping if adapting a different dataset.
DATA_FILE = Path("sales_eda_practice.csv")
if not DATA_FILE.exists():
    DATA_FILE = Path("/mnt/data/eda_interview_practice/sales_eda_practice.csv")
assert DATA_FILE.exists(), f"CSV not found: {DATA_FILE}"

# To reuse with differently named CSV fields, e.g. {"sales": "m_revenue"}.
RENAME_MAP = {}
ID_COLS = ["order_line_id", "order_id", "customer_id", "product_id"]
NUM_COLS = ["quantity", "unit_price", "discount_pct", "m_revenue", "delivery_days", "customer_age"]
CATEGORY_COLS = ["product_name", "category", "region", "channel", "day_name"]
DATE_COLS = ["order_date", "last_updated"]
BOOLEAN_COLS = ["is_returned"]

# Pandas DataFrame. String IDs prevent unwanted conversion of IDs to numbers.
pd_raw = pd.read_csv(DATA_FILE, dtype={col: "string" for col in ID_COLS}).rename(columns=RENAME_MAP)



# 1. Pandas exploration
# 1A — Pandas: preview, dimensions, columns, schema, summary
print("First 5:")
print(pd_raw.head(5).to_string(index=False))
print("Last 5:")
print(pd_raw.tail(5).to_string(index=False))
print("Row count:", len(pd_raw), "Column count:", pd_raw.shape[1])
print("Column names:", pd_raw.columns.tolist())
print("Dtypes:")
print(pd_raw.dtypes)
print("Numeric summary:")
print(pd_raw.describe(include="number"))  # numeric only where inferred
print("All-column summary:")
print(pd_raw.describe(include="all").to_string())
print("Distinct per column:\n", pd_raw.nunique(dropna=False))

# 1. Classification
# 1C — Automated column classification by business-name rules and data types.
# Reuse these column-name overrides on other datasets (date text still classified as date).
def classify_pandas(df):
    groups = {"identifier": [], "datetime": [], "boolean": [],
              "numerical": [], "categorical": []}
    for c in df.columns:
        if c in ID_COLS or c.lower().endswith("_id"):
            group = "identifier"
        elif c in DATE_COLS or pd.api.types.is_datetime64_any_dtype(df[c]):
            group = "datetime"
        elif c in BOOLEAN_COLS or pd.api.types.is_bool_dtype(df[c]):
            group = "boolean"
        elif pd.api.types.is_numeric_dtype(df[c]) or c in NUM_COLS:
            group = "numerical"
        else:
            group = "categorical"
        groups[group].append(c)
    return groups


print("Pandas classification:", classify_pandas(pd_raw))

# 1. Relationships
# 1D — Relationships and business interpretation: inspect join-cardinality and revenue drivers.
# One customer -> many orders; one order -> many order lines; product -> category.
print("Orders:", pd_raw["order_id"].nunique(),
      "Order lines:", pd_raw["order_line_id"].nunique(),
      "Customers:", pd_raw["customer_id"].nunique())
print("Order lines per order (sample):")
print(pd_raw.groupby("order_id")["order_line_id"].nunique().sort_values(ascending=False).head())
print("Product -> category variants (raw string inconsistencies show up):")
print(pd_raw.groupby("product_id")["category"].nunique().sort_values(ascending=False).head())


# 2. Quality profile
# 2A — Pandas quality report: per-column null counts and percentages
null_report_pd = pd.DataFrame({
    "null_count": pd_raw.isna().sum(),
    "null_pct": (pd_raw.isna().mean() * 100).round(2)
}).sort_values("null_pct", ascending=False)
print(null_report_pd.to_string())

# Duplicate full rows vs duplicate PKs are different problems.
print("Exact duplicate rows:", int(pd_raw.duplicated().sum()))
print("Repeated non-null order_line_id entries:", int(pd_raw.loc[
    pd_raw["order_line_id"].notna(), "order_line_id"].duplicated().sum()))
print("Legitimate repeat order_id count:", int(pd_raw["order_id"].duplicated().sum()))

# Range, zero, negative, illogical and bad-date checks without changing raw input.
raw_qty = pd.to_numeric(pd_raw["quantity"], errors="coerce")
raw_price = pd.to_numeric(pd_raw["unit_price"], errors="coerce")
raw_discount = pd.to_numeric(pd_raw["discount_pct"], errors="coerce")
raw_revenue = pd.to_numeric(pd_raw["m_revenue"], errors="coerce")
raw_age = pd.to_numeric(pd_raw["customer_age"], errors="coerce")
raw_delivery = pd.to_numeric(pd_raw["delivery_days"], errors="coerce")
raw_dates = pd.to_datetime(pd_raw["order_date"], errors="coerce", format="%Y-%m-%d")
checks_pd = {
    "quantity <= 0": int(raw_qty.le(0).sum()),
    "unit_price <= 0": int(raw_price.le(0).sum()),
    "discount not in [0,100)": int((raw_discount.lt(0) | raw_discount.ge(100)).sum()),
    "m_revenue <= 0": int(raw_revenue.le(0).sum()),
    "age not in 18..100": int((raw_age.lt(18) | raw_age.gt(100)).sum()),
    "delivery_days not in 0..30": int((raw_delivery.lt(0) | raw_delivery.gt(30)).sum()),
    "missing/unparseable order_date": int(raw_dates.isna().sum()),
}
print("Pandas validation failures:", checks_pd)
print("Category spelling/case variants:", sorted(pd_raw["category"].dropna().unique())[:20])
print("Normalised category values:", sorted(pd_raw["category"].dropna().str.strip().str.title().unique()))
expected = raw_qty * raw_price * (1 - raw_discount / 100)
print("Revenue mismatches (> ₹1):", int(((raw_revenue - expected).abs() > 1).sum()))

# 2. Clean in a NEW frame
# 2C — Pandas cleaning. Keep a NEW frame: pd_clean (raw unchanged).
pd_clean = pd_raw.drop_duplicates().copy()
pd_clean[ID_COLS] = pd_clean[ID_COLS].apply(lambda s: s.str.strip().replace("", pd.NA))
for c in NUM_COLS:
    pd_clean[c] = pd.to_numeric(pd_clean[c], errors="coerce")
pd_clean["order_date"] = pd.to_datetime(pd_clean["order_date"], format="%Y-%m-%d", errors="coerce")
pd_clean["last_updated"] = pd.to_datetime(pd_clean["last_updated"], format="%Y-%m-%d %H:%M:%S", errors="coerce")

# Latest record wins when the same primary key occurs more than once.
pd_clean = (pd_clean.sort_values(["last_updated", "order_line_id"], na_position="first")
            .drop_duplicates(subset=["order_line_id"], keep="last").copy())
pd_clean = pd_clean.dropna(subset=ID_COLS + ["order_date", "quantity", "unit_price", "discount_pct"])
pd_clean = pd_clean.loc[
    (pd_clean["quantity"] > 0) & (pd_clean["unit_price"] > 0) &
    pd_clean["discount_pct"].between(0, 100, inclusive="left")
].copy()
for c in CATEGORY_COLS:
    pd_clean[c] = pd_clean[c].astype("string").str.strip().str.title().fillna("Unknown")
    pd_clean[c] = pd_clean[c].replace("", "Unknown")
# Use the trustworthy order date rather than the often-misspelled supplied weekday.
pd_clean["day_name"] = pd_clean["order_date"].dt.day_name()
pd_clean.loc[~pd_clean["customer_age"].between(18, 100), "customer_age"] = np.nan
pd_clean.loc[~pd_clean["delivery_days"].between(0, 30), "delivery_days"] = np.nan
pd_clean["is_returned"] = (pd_clean["is_returned"].astype("string").str.strip().str.lower()
    .map({"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False})
    .astype("boolean"))
# Policy: keep a record of original reported revenue, recalculate net sales.
pd_clean["m_revenue_original"] = pd_clean["m_revenue"]
pd_clean["m_revenue"] = (pd_clean["quantity"] * pd_clean["unit_price"] *
    (1 - pd_clean["discount_pct"] / 100)).round(2)
print("Pandas cleaned rows:", len(pd_clean), "unique line IDs:", pd_clean["order_line_id"].nunique())
assert pd_clean["order_line_id"].is_unique
assert pd_clean["m_revenue"].gt(0).all()
print(pd_clean.head(3).to_string(index=False))

# 3. Skew and statistics
# 3A — Pandas: skew distribution and top-key share (region/customer/category)
for key in ["region", "category", "customer_id"]:
    counts = pd_clean[key].value_counts(dropna=False)
    print(f"\n{key} top 10 counts:\n", counts.head(10).to_string())
    print(f"Most common {key} share: {counts.iloc[0] / counts.sum() * 100:.2f}%")
    print(f"Max-to-average-key-count skew ratio: {counts.max() / counts.mean():.2f}x")

# Pandas describe and explicit statistics; choose meaningful numeric columns.
print(pd_clean[NUM_COLS].describe().to_string())
print("Revenue mean:", pd_clean["m_revenue"].mean())
print("Revenue median:", pd_clean["m_revenue"].median())
print("Quantity mode(s):", pd_clean["quantity"].mode().tolist())
print("Revenue min/max:", pd_clean["m_revenue"].min(), pd_clean["m_revenue"].max())
print("Revenue sample std:", pd_clean["m_revenue"].std(ddof=1))

# 3. Outliers
# 3C — Outliers by 3-sigma (do NOT automatically drop genuine bulk orders)
mu = pd_clean["m_revenue"].mean()
sigma = pd_clean["m_revenue"].std(ddof=1)
lower, upper = mu - 3 * sigma, mu + 3 * sigma
pd_outliers = pd_clean.loc[
    (pd_clean["m_revenue"] < lower) | (pd_clean["m_revenue"] > upper)
].copy() if pd.notna(sigma) and sigma > 0 else pd_clean.iloc[0:0].copy()
print(f"Pandas mean={mu:.2f}, std={sigma:.2f}, outlier interval=[{lower:.2f}, {upper:.2f}]")
print(pd_outliers[["order_line_id", "product_name", "quantity", "m_revenue"]]
      .sort_values("m_revenue", ascending=False).head(10).to_string(index=False))



# 3. Correlation and pivot
correlation_cols = ["quantity", "unit_price", "discount_pct", "m_revenue", "delivery_days", "customer_age"]
print("Pandas Pearson correlation matrix:\n", pd_clean[correlation_cols].corr().round(3))
pivot_pd = pd.pivot_table(pd_clean, index="region", columns="category",
    values="m_revenue", aggfunc="sum", fill_value=0)
print("Pandas pivot:\n", pivot_pd.round(2).to_string())


# 4. KPIs
# 4A — Pandas KPIs: SUM, AVG, P90 / P95, AOV
order_revenue_pd = pd_clean.groupby("order_id", as_index=False)["m_revenue"].sum()
metrics_pd = {
    "total_revenue": pd_clean["m_revenue"].sum(),
    "mean_line_revenue": pd_clean["m_revenue"].mean(),
    "p90_line_revenue": pd_clean["m_revenue"].quantile(.90),
    "p95_line_revenue": pd_clean["m_revenue"].quantile(.95),
    "order_count": pd_clean["order_id"].nunique(),
    "avg_per_order_AOV": pd_clean["m_revenue"].sum() / pd_clean["order_id"].nunique(),
    "p90_order_revenue": order_revenue_pd["m_revenue"].quantile(.90),
    "p95_order_revenue": order_revenue_pd["m_revenue"].quantile(.95),
}
print(pd.Series(metrics_pd).round(2).to_string())

# Category share (%) = each category's revenue / all revenue * 100.
category_pd = pd_clean.groupby("category", as_index=False).agg(
    revenue=("m_revenue", "sum"), order_lines=("order_line_id", "count"))
category_pd["share_pct"] = 100 * category_pd["revenue"] / category_pd["revenue"].sum()
category_pd = category_pd.sort_values("revenue", ascending=False)
print("Category share:\n", category_pd.to_string(index=False))

# Top N contribution and top-5 customers / products (by revenue).
customer_pd = pd_clean.groupby("customer_id", as_index=False)["m_revenue"].sum() \
    .rename(columns={"m_revenue": "revenue"}).sort_values("revenue", ascending=False)
product_pd = pd_clean.groupby(["product_id", "product_name"], as_index=False)["m_revenue"].sum() \
    .rename(columns={"m_revenue": "revenue"}).sort_values("revenue", ascending=False)
print("Top 5 customers:\n", customer_pd.head(5).to_string(index=False))
print("Top 5 products:\n", product_pd.head(5).to_string(index=False))
print("Top 5 customers contribution %:", round(
    100 * customer_pd.head(5)["revenue"].sum() / metrics_pd["total_revenue"], 2))
print("Top 5 products contribution %:", round(
    100 * product_pd.head(5)["revenue"].sum() / metrics_pd["total_revenue"], 2))

# 4. Monthly growth and 7-day average
# 4B — Pandas monthly growth %, daily sales, true 7-calendar-day moving average
monthly_pd = pd_clean.assign(month=pd_clean["order_date"].dt.to_period("M").astype(str)) \
    .groupby("month", as_index=False)["m_revenue"].sum().rename(columns={"m_revenue": "revenue"})
monthly_pd["prev_revenue"] = monthly_pd["revenue"].shift(1)
monthly_pd["growth_pct"] = np.where(monthly_pd["prev_revenue"] > 0,
    (monthly_pd["revenue"] - monthly_pd["prev_revenue"]) /
    monthly_pd["prev_revenue"] * 100, np.nan)
print("Month-over-month growth:\n", monthly_pd.round(2).to_string(index=False))

# Resample fills every day, including zero-sales days; min_periods=1 for first 6 days.
daily_pd = (pd_clean.set_index("order_date")["m_revenue"].resample("D").sum()
            .rename("daily_revenue").to_frame())
daily_pd["moving_avg_7d"] = daily_pd["daily_revenue"].rolling(7, min_periods=1).mean()
daily_pd = daily_pd.reset_index()
print("Daily revenue / 7-day moving avg:\n", daily_pd.tail(10).round(2).to_string(index=False))

# 5. Matplotlib five charts
# 5A — MATPLOTLIB: 5 charts, each with a question and how to interpret it
# BAR: Which category generates the most revenue?
plt.figure(figsize=(8,4))
plt.bar(category_pd["category"], category_pd["revenue"])
plt.title("Bar — Revenue by category"); plt.xlabel("Category"); plt.ylabel("Revenue (₹)")
plt.xticks(rotation=25); plt.tight_layout(); plt.show()

# LINE: Is revenue rising/falling? Does it show spikes?
plt.figure(figsize=(10,4))
plt.plot(daily_pd["order_date"], daily_pd["daily_revenue"], alpha=.5, label="Daily")
plt.plot(daily_pd["order_date"], daily_pd["moving_avg_7d"], linewidth=2,
         label="7-day moving average")
plt.title("Line — Revenue trend"); plt.xlabel("Order date"); plt.ylabel("Revenue (₹)")
plt.legend(); plt.tight_layout(); plt.show()

# HISTOGRAM: Is line revenue right-skewed? Are a few sales very large?
plt.figure(figsize=(8,4))
plt.hist(pd_clean["m_revenue"], bins=30, edgecolor="white")
plt.title("Histogram — Line revenue distribution")
plt.xlabel("Revenue per order line (₹)"); plt.ylabel("Frequency")
plt.tight_layout(); plt.show()

# SCATTER: Does quantity correlate with revenue? What are the outliers?
plt.figure(figsize=(8,4))
plt.scatter(pd_clean["quantity"], pd_clean["m_revenue"], alpha=.5)
plt.title("Scatter — Quantity vs revenue"); plt.xlabel("Quantity"); plt.ylabel("Revenue (₹)")
plt.tight_layout(); plt.show()

# PIE: What % of total revenue comes from each category?
plt.figure(figsize=(7,6))
plt.pie(category_pd["revenue"], labels=category_pd["category"],
        autopct="%1.1f%%", startangle=90)
plt.title("Pie — Category revenue contribution"); plt.tight_layout(); plt.show()

# 5. Seaborn chart alternatives
# 5B — SEABORN: 4 native chart types. Pie uses Matplotlib (Seaborn has no pie API).
plt.figure(figsize=(8,4))
sns.barplot(data=category_pd, x="category", y="revenue", hue="category", legend=False)
plt.title("Seaborn bar — Category revenue"); plt.xticks(rotation=25)
plt.tight_layout(); plt.show()

plt.figure(figsize=(10,4))
sns.lineplot(data=daily_pd, x="order_date", y="moving_avg_7d")
plt.title("Seaborn line — Revenue 7-day moving average")
plt.tight_layout(); plt.show()

plt.figure(figsize=(8,4))
sns.histplot(data=pd_clean, x="m_revenue", bins=30, kde=True, color=sns.color_palette()[0])
plt.title("Seaborn histogram — Revenue distribution")
plt.tight_layout(); plt.show()

plt.figure(figsize=(8,4))
sns.scatterplot(data=pd_clean, x="quantity", y="m_revenue", hue="category", alpha=.7)
plt.title("Seaborn scatter — Quantity, revenue and category")
plt.tight_layout(); plt.show()

# Seaborn pie chart equivalent: plt.pie(...) as in cell 5A above.

# 5. Business insights
# 5D — Generate interview-ready data-driven observations, not guessed findings.
lead_category = category_pd.iloc[0]
lead_customer = customer_pd.iloc[0]
lead_product = product_pd.iloc[0]
latest_month = monthly_pd.iloc[-1]
region_share = pd_clean.groupby("region")["m_revenue"].sum().sort_values(ascending=False)
returned = pd_clean["is_returned"].dropna().astype(int)
print("INSIGHTS — finding / meaning / action")
print(f"1. Category: {lead_category['category']} leads with ₹{lead_category['revenue']:,.0f} "
      f"({lead_category['share_pct']:.1f}% share). Action: investigate its margins and inventory.")
print(f"2. Customer concentration: top 5 contribute "
      f"{100*customer_pd.head(5)['revenue'].sum()/metrics_pd['total_revenue']:.1f}% of revenue. "
      "Action: evaluate loyalty and concentration risk.")
print(f"3. Best customer: {lead_customer['customer_id']} (₹{lead_customer['revenue']:,.0f}). "
      "Action: review repeat purchases.")
print(f"4. Best product: {lead_product['product_name']} (₹{lead_product['revenue']:,.0f}). "
      "Action: check stock availability.")
print(f"5. Largest regional sales share: {region_share.index[0]} "
      f"({100*region_share.iloc[0]/region_share.sum():.1f}%). "
      "Action: assess regional dependence; do not assume it is demand without market sizing.")
print(f"6. Latest period: {latest_month['month']} growth "
      f"{latest_month['growth_pct']:.1f}%. Action: check promotions/seasonality before attributing cause.")
print(f"7. 3-sigma high/low revenue lines: {len(pd_outliers)}. "
      "Action: inspect legitimate bulk purchases before considering removal.")
print(f"8. Return rate among known labels: {100*returned.mean():.1f}% "
      f"(known labels: {len(returned)}). Action: segment by category and channel.")
print("Note: These are synthetic data. Correlation is not causation; prioritize an actionable test.")
