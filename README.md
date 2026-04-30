# 🍬 Nassau Candy — Supply Chain Optimization

> **A five-stage machine learning pipeline that simulates factory-product reassignment scenarios, predicts lead times, and delivers ranked recommendations to reduce shipping distances and improve margins for Nassau Candy Distributor.**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Key Results](#key-results)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Stage Breakdown](#stage-breakdown)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Technologies Used](#technologies-used)
- [Deliverables](#deliverables)

---

## Overview

Nassau Candy Distributor operates **5 factories** across the United States and distributes **14 candy products** to customers across 4 geographic regions (Atlantic, Gulf, Interior, Pacific). Factory-product assignments were historically governed by static legacy rules — with no capability to simulate alternatives or quantify impact before execution.

This project delivers:

- A **predictive model** for shipping lead time (R² ≈ 0.64–0.67)
- A **simulation engine** generating 56 factory reassignment scenarios
- A **ranked recommendation system** covering 99.5% of product revenue
- An **interactive Streamlit dashboard** for real-time scenario exploration

## Pipeline Architecture

Raw Data (10,194 orders)
│

▼

┌─────────────────────────────────────────────────────────────┐

│  Stage 1 — Data Preparation & Feature Engineering           │

│  Factory mapping · Haversine distances · Encode & Scale     │

│  Output: nassau_enriched.csv (9,783 rows × 31 columns)      │

└─────────────────────┬───────────────────────────────────────┘

│

▼

┌─────────────────────────────────────────────────────────────┐

│  Stage 2 — Predictive Modeling                               │

│  Linear Regression · Random Forest · Gradient Boosting       │

│  Per-band training (3 scheduling tiers) · R² ≈ 0.64–0.67     │

│  Output: stage2_band_models.pkl                              │

└─────────────────────┬───────────────────────────────────────┘

│

▼

┌─────────────────────────────────────────────────────────────┐

│  Stage 3 — Route & Product Clustering                        │

│  KMeans (k=5) · Silhouette score selection                   │

│  Identifies: slow routes · high-value inefficient products   │

│  Output: stage3_route_clusters.csv · stage3_product_clusters │

└─────────────────────┬───────────────────────────────────────┘

│

▼

┌─────────────────────────────────────────────────────────────┐

│  Stage 4 — Scenario Simulation Engine                        │

│  56 scenarios (14 products × 4 alternate factories)          │

│  4 KPIs: LT reduction · Profit impact · Confidence · Coverage│

│  Output: stage4_simulations.csv                              │

└─────────────────────┬───────────────────────────────────────┘

│

▼

┌─────────────────────────────────────────────────────────────┐

│  Stage 5 — Optimization & Recommendations                    │

│  Weighted composite scoring · Ranked reassignment table      │

│  Factory workload analysis · Revenue coverage report         │

│  Output: stage5_recommendations.csv                          │

└─────────────────────────────────────────────────────────────┘

---

## Key Results

| Metric | Value |
|---|---|
| **Dataset** | 9,783 orders · 14 products · 5 factories · 4 regions |
| **Best model R²** | 0.675 (Random Forest — Medium band) |
| **Model RMSE** | ~1.0 day (within-band std = 1.7 days) |
| **Avg lead time reduction** | 20.7 days (1.51%) — viable low-risk scenarios |
| **Avg profit impact** | $517 per product per year |
| **Avg confidence score** | 0.63 / 1.0 |
| **Recommendation coverage** | 99.5% of product revenue (8/14 products) |

### Top Recommendations

| # | Product | From | To | LT Reduction | Score |
|---|---|---|---|---|---|
| 1 | Scrumdiddlyumptious | Lot's O' Nuts | The Other Factory | 0.35% | 0.534 |
| 2 | Nutty Crunch | Lot's O' Nuts | The Other Factory | 0.67% | 0.531 |
| 3 | SweeTARTS | Sugar Shack | The Other Factory | 5.57% | 0.531 |

### Key Findings

- **Sugar Shack → Pacific** is the worst-performing route (avg 1,517 days — 196 days above portfolio mean)
- **Hair Toffee** ($14.88 avg GP) and **Lickable Wallpaper** ($10.00 avg GP) are the highest-priority reassignment candidates — high margin, slowest cluster
- **Standard Class** shipping is paradoxically faster than First Class — factory location matters more than ship mode
- **The Other Factory** (Tennessee) is the best alternate destination due to its central geographic position relative to Gulf and Interior regions

---

## Project Structure
nassau-candy-optimization/

│

├── nassau_pipeline.py              # Full pipeline — Stages 1–5 in one file

│

├── stages/                         # Individual stage scripts

│   ├── stage1_data_preparation.py

│   ├── stage1_eda_visualization.py

│   ├── stage2_predictive_modeling.py

│   ├── stage3_clustering.py

│   ├── stage4_simulation.py

│   └── stage5_recommendations.py

│

├── app.py                          # Streamlit dashboard

├── requirements.txt                # Python dependencies

│

├── data/                           # ← Place your CSV here

│   └── Nassau_Candy_Distributor.csv

│

└── outputs/                        # Auto-generated by pipeline

├── nassau_enriched.csv

├── nassau_eda_dashboard.png

├── stage2_band_models.pkl

├── stage2_model_results.csv

├── stage2_model_comparison.png

├── stage3_route_clusters.csv

├── stage3_product_clusters.csv

├── stage3_clustering.png

├── stage4_simulations.csv

├── stage4_simulation.png

├── stage5_recommendations.csv

├── stage5_top_recommendations.csv

├── stage5_recommendations.png

└── stage5_report.pkl

---

## Pipeline Architecture
