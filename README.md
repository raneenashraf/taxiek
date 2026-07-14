# 🚖 Taxiek Trip Pricing App

<p align="center">
  <img src="taxiek.jpg" alt="Taxiek Logo" width="350">
</p>

<p align="center">

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

</p>

An interactive **bilingual (English & Arabic)** data science web application that analyzes taxi trip data, visualizes key insights, and predicts taxi fares in **Egyptian Pounds (EGP)** using Machine Learning.

Built with **Streamlit**, **Scikit-Learn**, **Pandas**, **Matplotlib**, and **Seaborn**, the application combines interactive analytics with an intelligent fare prediction system.

---

# 🌟 Features

## 🌐 Bilingual User Interface

- Switch instantly between **English** and **العربية**
- Localized interface
- Arabic and English charts
- Dynamic labels and tooltips
- Bilingual prediction results

---

## 📊 Interactive Data Visualization

Explore the dataset through an interactive dashboard.

### Available Visualizations

- 📈 Histogram
- 📦 Box Plot
- 📉 Scatter Plot
- 📊 Bar Plot
- 🔥 Correlation Heatmap

Additional analytics include:

- Dataset preview
- Summary statistics
- Distribution analysis
- Outlier detection
- Correlation analysis

---

## 💰 Smart Price Prediction

Predict taxi fares instantly using Machine Learning.

### User Inputs

- 🚕 Trip Distance
- 👥 Passenger Count
- 🕒 Time of Day
- 📅 Day of Week
- 🚦 Traffic Conditions
- 🌦️ Weather Conditions
- 💵 Base Fare
- 📍 Per Kilometer Rate
- ⏱️ Per Minute Rate
- ⌛ Trip Duration

### Prediction Output

The application predicts:

- Total Trip Price (EGP)
- Fare Breakdown
- Base Fare
- Distance Cost
- Time Cost

Interactive pie charts visualize the contribution of each pricing component.

---

# 🧠 Machine Learning Pipeline

The application follows a complete preprocessing and modeling workflow.

## Data Cleaning

- Missing value handling
- Automatic USD → EGP conversion
- Numerical imputation (Mean)
- Categorical imputation (Mode)

---

## Feature Engineering

### Numerical Features

- MinMaxScaler
- PolynomialFeatures (Degree 2)

### Categorical Features

- OneHotEncoder
- Handle Unknown Categories

Implemented using **ColumnTransformer**.

---

## Model Training

The project evaluates multiple regression algorithms including:

- Polynomial Linear Regression
- Random Forest Regressor

Hyperparameter tuning is performed using:

- GridSearchCV

Evaluation metrics include:

- R² Score
- Mean Squared Error (MSE)

---

# 📊 Application Workflow

```text
Taxi Trip Dataset
        │
        ▼
Data Cleaning & Preprocessing
        │
        ▼
Feature Engineering
(ColumnTransformer)
        │
        ▼
Machine Learning Model
(Random Forest / Polynomial Regression)
        │
        ▼
Price Prediction
        │
        ▼
Interactive Dashboard
(Streamlit)
```

---

# 📁 Project Structure

```text
Taxi_app/
│
├── taxiek.py
├── taxi.py
├── taxi.ipynb
├── taxi_trip_pricing.csv
├── cleaned_taxi.csv
├── taxiek.jpg
├── requirements.txt
└── README.md
```

### File Description

| File | Description |
|------|-------------|
| `taxiek.py` | Main bilingual Streamlit application |
| `taxi.py` | Machine learning training & evaluation script |
| `taxi.ipynb` | Exploratory Data Analysis and experimentation |
| `taxi_trip_pricing.csv` | Original dataset |
| `cleaned_taxi.csv` | Cleaned dataset |
| `requirements.txt` | Project dependencies |

---

# 🛠 Technology Stack

| Layer | Technologies |
|---------|-------------|
| Frontend | Streamlit |
| Machine Learning | Scikit-Learn |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Model Selection | GridSearchCV |
| Feature Engineering | ColumnTransformer, PolynomialFeatures |
| Encoding | OneHotEncoder |
| Scaling | MinMaxScaler |

---

# 🚀 Installation

## Clone the Repository

```bash
git clone https://github.com/your-username/Taxi_app.git
```

```bash
cd Taxi_app
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Launch the Application

```bash
streamlit run taxiek.py
```

The application will open automatically at:

```
http://localhost:8501
```

---

## Train Models (Optional)

To retrain the machine learning models:

```bash
python taxi.py
```

The script prints:

- Model Performance
- Best Parameters
- R² Score
- Mean Squared Error

---

# 📦 Dependencies

- Streamlit
- Scikit-Learn
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

# ✨ Highlights

- 🌍 Bilingual Interface
- 📊 Interactive Dashboard
- 🤖 Machine Learning Prediction
- 📈 Exploratory Data Analysis
- 📉 Correlation Heatmaps
- 🚖 Smart Fare Estimation
- 💰 Detailed Cost Breakdown
- 🥧 Interactive Pie Charts

---



**AI Engineer | Data Scientist | Machine Learning Engineer**

Faculty of Computers and Data Science  
Alexandria University

---

# 📄 License

This project is intended for educational and research purposes.

© 2026 Raneen Ashraf. All Rights Reserved.

---

⭐ **If you found this project useful, consider giving it a star on GitHub!**
