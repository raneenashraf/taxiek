import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_squared_error
from PIL import Image
st.set_page_config(page_title="🚖 Taxiek Pricing", page_icon= "🚖",layout="wide")

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv(r"C:\Users\ranee\Dropbox\PC\Desktop\Taxi_app\taxi_trip_pricing.csv")
    
    # Convert USD to EGP if needed
    usd_to_egp = 48.0
    if data['Trip_Price'].max() < 200:
        data['Trip_Price'] = data['Trip_Price'] * usd_to_egp
    
    # Handle missing values
    data["Passenger_Count"] = data["Passenger_Count"].astype("Int64")
    for column in data.columns:
        if data[column].dtype in ['int64', 'float64']:
            data[column] = data[column].fillna(data[column].mean())
        else:
            data[column] = data[column].fillna(data[column].mode()[0])
    
    return data

data = load_data()

# Preprocessing setup
categorical_cols = ['Time_of_Day', 'Day_of_Week', 'Traffic_Conditions', 'Weather']
numerical_cols = [col for col in data.columns if col not in categorical_cols + ['Trip_Price']]

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Train model
@st.cache_resource
def train_model():
    X = data.drop(columns='Trip_Price')
    y = data['Trip_Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            random_state=42
        ))
    ])
    
    rf_pipeline.fit(X_train, y_train)
    return rf_pipeline

model = train_model()
# ========================
# ✅ APP with STREAMLIT
# ========================
st.sidebar.markdown("## 🌐 Select Language / اختر اللغة")
language = st.sidebar.radio("", ["English", "عربي"], horizontal=True)

# ========================
# 🗂️ Multilingual Text
# ========================
TEXT = {
    "English": {
        "title": "🚖 Taxiek Trip Pricing App",
        "modes": ["📊 Data Visualization", "💵 Price Prediction"],
        "show_data": "Show Raw Data",
        "plot_type": "Choose Plot Type",
        "histogram": "Histogram",
        "box": "Box Plot",
        "scatter": "Scatter Plot",
        "bar": "Bar Plot",
        "heatmap": "Correlation Heatmap",
        "summary": "Summary Statistics",
        "predict": "Predict Trip Price",
        "distance": "Trip Distance (km)",
        "time": "Time of Day",
        "day": "Day of Week",
        "passengers": "Passenger Count",
        "traffic": "Traffic Conditions",
        "weather": "Weather",
        "fare": "Base Fare (EGP)",
        "km_rate": "Per Km Rate (EGP)",
        "min_rate": "Per Minute Rate (EGP)",
        "duration": "Trip Duration (Minutes)",
        "predicted_price": "🚖 Predicted Trip Price:",
        "breakdown": "Price Breakdown",
        "components": ["Base Fare", "Distance Cost", "Time Cost"],
        "footer": "Developed by Raneen Ashraf | © 2025 Taxiek Pricing App"
    },
    "عربي": {
        "title": "🚖 تطبيق تسعير رحلات تاكسيك",
        "modes": ["📊 تحليل البيانات", "💵 توقع سعر الرحلة"],
        "show_data": "عرض البيانات الخام",
        "plot_type": "اختر نوع الرسم",
        "histogram": "مخطط التكرار",
        "box": "مربع الإحصائيات",
        "scatter": "مخطط مبعثر",
        "bar": "مخطط الأعمدة",
        "heatmap": "خريطة الارتباط",
        "summary": "ملخص إحصائي",
        "predict": "توقع سعر الرحلة",
        "distance": "مسافة الرحلة (كم)",
        "time": "وقت اليوم",
        "day": "يوم الأسبوع",
        "passengers": "عدد الركاب",
        "traffic": "حالة المرور",
        "weather": "الطقس",
        "fare": "السعر الأساسي (جنيه)",
        "km_rate": "سعر لكل كم (جنيه)",
        "min_rate": "سعر لكل دقيقة (جنيه)",
        "duration": "مدة الرحلة (دقائق)",
        "predicted_price": "🚖 السعر المتوقع: ",
        "breakdown": "تفصيل السعر",
        "components": ["السعر الأساسي", "تكلفة المسافة", "تكلفة الوقت"],
        "footer": "تم التطوير بواسطة رنين أشرف | © 2025 تطبيق تاكسيك"
    }
}

T = TEXT[language]

# ========================
# 📦 Load and Preprocess Data
# ========================
@st.cache_data
def load_data():
    df = pd.read_csv("C:/Users/ranee/Dropbox/PC/Desktop/Taxi_app/taxi_trip_pricing.csv")
    if df['Trip_Price'].max() < 200:
        df['Trip_Price'] *= 48.0  # Convert USD to EGP
    df['Passenger_Count'] = df['Passenger_Count'].astype("Int64")
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    return df

data = load_data()
categorical_cols = ['Time_of_Day', 'Day_of_Week', 'Traffic_Conditions', 'Weather']
numerical_cols = [c for c in data.columns if c not in categorical_cols + ['Trip_Price']]

# ========================
# 🤖 Train Model
# ========================
@st.cache_resource
def train_model():
    X = data.drop(columns='Trip_Price')
    y = data['Trip_Price']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline = Pipeline([
        ('preprocessor', ColumnTransformer([
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

model = train_model()

# ========================
# 🚀 App UI
# ========================
st.title(T["title"])
st.image("C:/Users/ranee/Dropbox/PC/Desktop/Taxi_app/taxiek.jpg", use_container_width=True)

# Tabs for modes
tab1, tab2 = st.tabs(T["modes"])

# ========================
# 📊 Data Visualization
# ========================
with tab1:
    if st.checkbox(T["show_data"]):
        st.write(data)

    plot = st.selectbox(T["plot_type"], [T["histogram"], T["box"], T["scatter"], T["bar"], T["heatmap"]])

    if plot == T["histogram"]:
        col = st.selectbox("Column", numerical_cols + ['Trip_Price'])
        bins = st.slider("Bins", 5, 100, 20)
        fig, ax = plt.subplots()
        sns.histplot(data[col], bins=bins, kde=True, ax=ax)
        st.pyplot(fig)

    elif plot == T["box"]:
        col = st.selectbox("Column", numerical_cols + ['Trip_Price'])
        fig, ax = plt.subplots()
        sns.boxplot(x=data[col], ax=ax)
        st.pyplot(fig)

    elif plot == T["scatter"]:
        x = st.selectbox("X", numerical_cols)
        y = st.selectbox("Y", numerical_cols + ['Trip_Price'])
        hue = st.selectbox("Hue", [None] + categorical_cols)
        fig, ax = plt.subplots()
        sns.scatterplot(data=data, x=x, y=y, hue=hue, ax=ax)
        st.pyplot(fig)

    elif plot == T["bar"]:
        x = st.selectbox("X", categorical_cols)
        y = st.selectbox("Y", ['Trip_Price'] + numerical_cols)
        fig, ax = plt.subplots()
        sns.barplot(data=data, x=x, y=y, ax=ax)
        st.pyplot(fig)

    elif plot == T["heatmap"]:
        fig, ax = plt.subplots()
        sns.heatmap(data[numerical_cols + ['Trip_Price']].corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    st.subheader(T["summary"])
    st.write(data.describe())

# ========================
# 💵 Prediction
# ========================
with tab2:
    col1, col2 = st.columns(2)
    with col1:
        dist = st.number_input(T["distance"], min_value=0.1, value=10.0)
        tod = st.selectbox(T["time"], ['Morning', 'Afternoon', 'Evening', 'Night'])
        dow = st.selectbox(T["day"], ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        pc = st.number_input(T["passengers"], min_value=1, max_value=8, value=2)
    with col2:
        traffic = st.selectbox(T["traffic"], ['Light', 'Moderate', 'Heavy', 'Standstill'])
        weather = st.selectbox(T["weather"], ['Clear', 'Rainy', 'Foggy', 'Stormy'])
        base = st.number_input(T["fare"], value=15.0)
        per_km = st.number_input(T["km_rate"], value=2.0)
        per_min = st.number_input(T["min_rate"], value=0.5)
        duration = st.number_input(T["duration"], value=25)

    if st.button(T["predict"]):
        new_data = pd.DataFrame([{
            'Trip_Distance_km': dist,
            'Time_of_Day': tod,
            'Day_of_Week': dow,
            'Passenger_Count': pc,
            'Traffic_Conditions': traffic,
            'Weather': weather,
            'Base_Fare': base,
            'Per_Km_Rate': per_km,
            'Per_Minute_Rate': per_min,
            'Trip_Duration_Minutes': duration
        }])

        price = model.predict(new_data)[0]
        st.success(f"{T['predicted_price']} {price:.2f} EGP")

        st.subheader("breakdown")
        breakdown = pd.DataFrame({
            "Component": ["Base Fare", "Distance Cost", "Time Cost"],
            "Amount (EGP)": [base, dist * per_km, duration * per_min]
        })
        st.table(breakdown)

        fig, ax = plt.subplots()
        ax.pie(breakdown["Amount (EGP)"], labels=breakdown["Component"], autopct='%1.1f%%')
        ax.set_title("breakdown")
        st.pyplot(fig)
       

        # Explanation for each component
        explanation_dict = {
            "English": {
                "Base Fare": "Fixed starting charge for the trip",
                "Distance Cost": "Charge based on distance (km)",
                "Time Cost": "Charge based on time (minutes)"
            },
            "عربي": {
                "السعر الأساسي": "رسوم ثابتة في بداية الرحلة",
                "تكلفة المسافة": "رسوم بالكيلومتر",
                "تكلفة الوقت": "رسوم حسب الدقائق"
            }
        }

        # Add explanation column
        breakdown["Meaning"] = [
            explanation_dict[language].get(comp, "") for comp in T["components"]
        ]

        st.table(breakdown)

        fig, ax = plt.subplots()
        labels = T["components"]
        if language == "عربي":
            labels = [label[::-1] for label in labels]  # optional: reverse for display
        ax.pie(breakdown["Amount (EGP)"], labels=labels, autopct='%1.1f%%')
        ax.set_title(T["breakdown"])

# Footer
st.markdown("---")
st.markdown("""
<p style='text-align: center; color: #666;'>
<br>Developed by Raneen Ashraf | © 7..2025 Taxiek Pricing App<br>
© 2025   حقوق محفوظة | تم التطوير بواسطة رنين أشرف @
</p>
""", unsafe_allow_html=True)
