# BMW Used Car Pricing Analysis (`carpricingpython`)

An end-to-end data science workflow in Python analyzing used BMW resale values, depreciation curves, and feature premiums using the [BMW Pricing Challenge](https://www.kaggle.com/datasets/danielkyrka/bmw-pricing-challenge) dataset from Kaggle.

> **Project Status:**
> Phase 1 Complete (Data Audit, Cleaning, Wrangling & Exploratory Data Analysis).
> Phase 2 (Regression Modeling & Price Prediction) coming next.

---

## 📊 Project Overview

The dataset contains **4,843 European used BMW listings** sold between January and September 2018 across 18 attributes—including model designation, mileage, engine power (kW), registration date, fuel type, paint color, body style, and eight anonymized equipment flags (`feature_1` through `feature_8`).

The objective of this phase is to audit data quality, engineer domain-specific automotive features, and uncover the primary drivers of secondary-market vehicle valuation prior to building predictive models.

---

## 🧹 Data Cleaning & Anomaly Remediation

While the raw dataset has zero `NaN` values and zero exact duplicate rows, statistical auditing revealed 66 corrupt or non-retail entries (1.36% of rows) that were cleaned prior to analysis:

| Anomaly | Raw Data Finding | Remediation |
| :--- | :--- | :--- |
| **Constant Column** | `maker_key` is `"BMW"` across 100% of records | Dropped `maker_key` |
| **Negative Mileage** | 1 record (`640 Gran Coupé`, 2017) listed at `-64 km` | Removed invalid record |
| **Extreme High Mileage** | 1 record (`118`, 2005) listed at `1,000,376 km` | Filtered out `mileage > 500,000 km` |
| **Zero Engine Power** | 1 record (`X1`, 2014) recorded as `0 kW` | Imputed with `X1` median (`105 kW`) |
| **Salvage / Placeholder Prices** | 62 records priced `< €1,000` (including late-model cars at `€100` or `€800`) | Removed `price < €1,000` |
| **Extreme Price Outliers** | 2 records (`2013 X3` at `€178,500`; `2014 X4` at `€142,800`) | Removed `price > €100,000` |

* **Final Clean Dataset:** **4,777 vehicles** (98.6% retention).

---

## 🛠️ Feature Engineering

To capture non-linear depreciation and simplify high-cardinality categorical variables, the pipeline engineers the following features:

* `vehicle_age_years`: Exact age of the vehicle at the date of sale (`sold_at - registration_date`).
* `km_per_year`: Annualized usage intensity (`mileage / vehicle_age_years`).
* `engine_hp`: Engine output converted from kilowatts (`kW`) to horsepower (`HP`).
* `total_features`: Aggregate count of optional equipment flags (`feature_1` through `feature_8`).
* `series`: Consolidation of 75 granular `model_key` values into high-level BMW model families (`1 Series` through `7 Series`, `X Series`, `Z Series`, and `M / i Performance`).

---

## 🔍 Key Exploratory Findings

1. **Right-Skewed Resale Distribution:** Clean sale prices center around a **median of €14,300** (IQR: €10,900 – €18,800) and a **mean of €15,955** ($\sigma = €8,572$), with a long upper tail driven by low-mileage SUVs and performance models.
2. **Primary Value Drivers (Pearson $r$ with `price`):**
   * **Engine Power (`engine_power`):** $r = +0.68$ (strongest single positive predictor)
   * **Equipment Count (`total_features`):** $r = +0.52$
   * **Vehicle Age (`vehicle_age_years`):** $r = -0.47$
   * **Odometer (`mileage`):** $r = -0.43$
3. **Compounding Depreciation:** Vehicles under 3 years old command €25,000–€70,000+, whereas vehicles crossing 200,000 km and 8+ years converge toward a €5,000–€8,000 floor.
4. **Model & Body Hierarchy:** Volume fleet models (`320`, `520`, `318`, `X3`, `116`) account for 57% of listings, while **6 Series (€33,200)**, **M / i Performance (€30,900)**, and **7 Series (€25,350)** command the highest median valuations.
5. **Equipment Option Lift:** Binary features carry vastly different market premiums—`feature_4` (+€7,200 median lift) and `feature_8` (+€5,900 lift) act as high-value trim packages, whereas `feature_7` (present on 94% of cars) reflects standard base equipment.
6. **Powertrain Mix:** Inventory is **95.8% diesel**, reflecting typical 2018 European corporate lease returns.

---

## 📈 EDA Dashboard

![BMW Used Car Pricing EDA Dashboard](bmw_eda_dashboard.png)

*(Upload `bmw_eda_dashboard.png` to the root of this repository to display the visual dashboard above.)*

---

## 📁 Repository Structure

```text
carpricingpython/
├── data/
│   └── bmw_pricing_challenge.csv   # Raw Kaggle dataset
├── src/
│   └── eda_pipeline.py             # Data cleaning, wrangling, and visualization script
├── bmw_eda_dashboard.png           # 4-panel EDA summary visual
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
