# Using public estimates + clear assumptions to build historical series (2018-2024) for three streams:
# - b2c_paying_subscribers (counts)
# - b2b_institutions (counts)
# - recruiter_leads (counts)
#
# Then run linear and exponential (log-linear) forecasts for 2025-2027 and compute combined
# monthly revenue using R = 299*x1 + 4167*x2 + 1000*x3 - 70000
#
# NOTE: These historical numbers are constructed from public reports and conservative assumptions.
# Replace them with your own data later if you want exact inputs.
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Historical years
years = np.array([2018,2019,2020,2021,2022,2023,2024])

# --- ASSUMPTIONS (derived from public reports; see chat for citations) ---
# B2C paying subscribers (counts) - conservative series (in number of users)
# Sources used: IBEF (paid users ~9.6M in 2021), Times of India projections (~37M by 2025),
# IMARC market sizing for context. These are assumptions for demonstration.
b2c = np.array([500000, 1200000, 2500000, 9600000, 12000000, 18000000, 20000000])

# B2B institutions (counts) - number of institutional clients (conservative)
b2b = np.array([5, 6, 8, 12, 20, 30, 50])

# Recruiter leads (counts) - monthly qualified leads sold to recruiters (conservative)
recruiter = np.array([10, 15, 25, 40, 60, 100, 200])

# Function to fit linear and log-linear models and forecast
def fit_and_forecast(years, values, future_years):
    X = years.reshape(-1,1)
    lin = LinearRegression().fit(X, values)
    lin_pred = lin.predict(future_years.reshape(-1,1))
    # If values are all positive, fit log-linear for exponential trend
    if np.all(values > 0):
        log_model = LinearRegression().fit(X, np.log(values))
        exp_pred = np.exp(log_model.predict(future_years.reshape(-1,1)))
    else:
        exp_pred = lin_pred  # fallback
    return lin_pred, exp_pred, lin

future_years = np.array([2025,2026,2027])

# Forecasts for each stream
b2c_lin_pred, b2c_exp_pred, b2c_lin_model = fit_and_forecast(years, b2c, future_years)
b2b_lin_pred, b2b_exp_pred, b2b_lin_model = fit_and_forecast(years, b2b, future_years)
rec_lin_pred, rec_exp_pred, rec_lin_model = fit_and_forecast(years, recruiter, future_years)

# Round and ensure non-negative
def clean(pred):
    return np.maximum(0, np.round(pred).astype(int))

b2c_lin = clean(b2c_lin_pred)
b2c_exp = clean(b2c_exp_pred)
b2b_lin = clean(b2b_lin_pred)
b2b_exp = clean(b2b_exp_pred)
rec_lin = clean(rec_lin_pred)
rec_exp = clean(rec_exp_pred)

unit_b2c = 299      # ₹ per B2C subscriber per month
unit_b2b = 4167     # ₹ per B2B institution per month (50,000/12)
unit_rec = 1000     # ₹ per recruiter lead
fixed_overhead = 70000  # ₹ monthly overhead

# Compute combined monthly revenue R = 299*x1 + 4167*x2 + 1000*x3 - 70000
def compute_R(x1, x2, x3):
    return unit_b2c*x1 + unit_b2b*x2 + unit_rec*x3 - fixed_overhead

R_lin = compute_R(b2c_lin, b2b_lin, rec_lin)
R_exp = compute_R(b2c_exp, b2b_exp, rec_exp)

# Build dataframes for display
hist_df = pd.DataFrame({
    "year": years,
    "b2c_paying_subscribers": b2c,
    "b2b_institutions": b2b,
    "recruiter_leads": recruiter
})

pred_df = pd.DataFrame({
    "year": future_years,
    "b2c_pred_linear": b2c_lin,
    "b2b_pred_linear": b2b_lin,
    "recruiter_pred_linear": rec_lin,
    "combined_monthly_revenue_linear": R_lin,
    "b2c_pred_exponential": b2c_exp,
    "b2b_pred_exponential": b2b_exp,
    "recruiter_pred_exponential": rec_exp,
    "combined_monthly_revenue_exponential": R_exp
})

# Display dataframes to user
import caas_jupyter_tools as cj
cj.display_dataframe_to_user("Assumed historical streams (2018-2024) — built from public reports & documented assumptions", hist_df)
cj.display_dataframe_to_user("Forecasts (2025-2027) and combined monthly revenue (linear vs exponential)", pred_df)

# Plotting combined subscribers (B2C) and revenue
plt.figure(figsize=(9,5))
plt.plot(years, b2c, 'o-', label="B2C historical (assumed)")
plt.plot(future_years, b2c_lin, 'x--', label="B2C forecast (linear)")
plt.plot(future_years, b2c_exp, 's--', label="B2C forecast (exponential)")
plt.xlabel("Year")
plt.ylabel("B2C paying subscribers (count)")
plt.title("B2C paying subscribers: historical assumptions -> forecasts")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(9,5))
plt.plot(future_years, R_lin, 'o-', label="Combined monthly revenue (linear forecast)")
plt.plot(future_years, R_exp, 's--', label="Combined monthly revenue (exponential forecast)")
plt.xlabel("Year")
plt.ylabel("Combined monthly revenue (₹)")
plt.title("Combined monthly revenue forecasts (₹ per month)")
plt.legend()
plt.tight_layout()
plt.show()

# Save CSV
pred_df.to_csv("/mnt/data/combined_forecast_revenue_2025_2027.csv", index=False)
print("Saved CSV: /mnt/data/combined_forecast_revenue_2025_2027.csv")

# Print compact table for quick copy-paste
print("\nForecast table (copyable):")
print(pred_df.to_string(index=False))
