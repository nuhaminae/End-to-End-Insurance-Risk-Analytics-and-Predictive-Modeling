# End-to-End Insurance Risk Analytics and Predictive Modelling

This repository provides a comprehensive, reproducible workflow for the analysis of insurance risk data and the development of predictive models to assess and manage insurance risk. The project is structured to guide users from data ingestion through to exploratory data analysis (EDA), feature engineering, and visualisation, with a strong emphasis on data quality, transparency, and best practices.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Modelling Approach](#modelling-approach)
- [Results & Visualisations](#results--visualisations)
- [DVC & Data Management](#dvc--data-management)
- [Continuous Integration](#continuous-integration)
- [Contribution](#contribution)
- [Completion Status](#completion-status)

---

## Overview

The insurance industry relies extensively on data-driven techniques for risk assessment, pricing, and claims management. This repository demonstrates a full pipeline for insurance analytics, using Jupyter Notebooks and Python for all major analysis and modelling tasks. The project aims to serve as a practical resource for data scientists, actuaries, students, and industry professionals.

---

## Features

- **End-to-end workflow** covering:
  - Data loading, cleaning, and pre-processing
  - Exploratory data analysis (EDA) and visualisation
  - Feature engineering and extraction
  - Transparent, modular code and notebook examples
- **Reproducible research** using Jupyter Notebooks and DVC (Data Version Control)
- **Automated testing** via CI workflow (GitHub Actions)
- **Well-documented and adaptable** for application to similar datasets or business problems

---

## Project Structure

```
.
├── data/                     # Raw and processed datasets (excluded from repo if confidential)
├── notebooks/                # Jupyter Notebooks for each stage of the workflow
│   └──01_DataLoading.ipynb
|   └──02_EDA.ipynb
|    ...
├── TEST/                     # Unit test notebooks for validation and reproducibility
├── scripts/                  # Python scripts for reusable functions
├── plot images/              # Output graphs and visualisations
├── .dvc/                     # DVC configuration and meta-files
├── .github/workflows/        # Continuous integration workflows
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation

```

> **Note:** Some directories or files may be excluded from the repository for privacy or size considerations.

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/nuhaminae/End-to-End-Insurance-Risk-Analytics-and-Predictive-Modeling.git
   cd End-to-End-Insurance-Risk-Analytics-and-Predictive-Modeling
   ```

2. **Set up a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

---

## Usage

- Open the notebooks in the `notebooks/` directory and follow them in order to reproduce the workflow.
- Replace the sample data with your own dataset if necessary, and adapt parameters as required for your analysis.
- Visual outputs (plots and summary statistics) are saved in the `plot images/` directory.

---

## Data

The project expects as input a structured insurance dataset, typically including:
- Policyholder demographics (e.g. age, gender, location)
- Policy details (e.g. type, value, inception date)
- Claims history and outcomes
- Vehicle details (where relevant)
- Additional external or derived variables

> **Disclaimer:** No confidential or proprietary data is included in this repository. Example datasets and synthetic data may be used for demonstration only.

---

## Modelling Approach

1. **Pre-processing:**
   - Capitalisation of column names for consistency
   - Filling empty indices with 'NAN' and dropping empty or irrelevant columns
   - Proper conversion of date fields and sorting by key time columns

2. **EDA (Exploratory Data Analysis):**
   - Univariate and multivariate statistical summaries
   - Data visualisation (distribution plots, time series, bar charts)
   - Correlation analysis and detection of trends or anomalies

3. **Feature Engineering:**
   - Creation and transformation of features relevant to insurance risk
   - Aggregations (e.g. monthly premiums and claims per region)
   - Calculation of changes and trends over time

4. **Visualisation:**
   - Plots and charts are generated and saved for further interpretation and reporting

---

## Results & Visualisations

- Key EDA findings, statistical summaries, and visualisations are found in the relevant notebooks and in the `plot images/` directory.
- Examples include distributions of cover types by province, average premiums by region, and time series of claims and premium changes.

---

## DVC & Data Management

- **DVC (Data Version Control):** Used for tracking data files and ensuring reproducibility.
  - The `.dvc/config` file points to a specific local storage directory for versioned data management.
  - `.dvcignore` and `.dvc/.gitignore` are used to prevent temporary or confidential files from being tracked.
  - Large or sensitive data is managed outside of Git, with DVC handling versioning and retrieval.

---

## Continuous Integration

- **GitHub Actions Workflow** (`.github/workflows/ci.yml`):
  - Automated testing on every push or pull request
  - Sets up Python environment, installs dependencies, and (optionally) runs tests or validation steps

---

## Contribution

Contributions are welcomed! To participate:
1. Fork the repository and create a new branch.
2. Implement your feature or fix.
3. Ensure code is well-documented and tested.
4. Submit a pull request with a clear description of your changes.

Please review open issues and pull requests before starting major work.

---

## Completion Status

The project is completed. Please see the [commit history](https://github.com/nuhaminae/End-to-End-Insurance-Risk-Analytics-and-Predictive-Modeling/commits?author=nuhaminae) for further details.
