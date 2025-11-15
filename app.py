import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="BigMart Sales Prediction", layout="wide")

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model():
    with open('california_housing_dataset.pkl', "rb") as f:
        return pickle.load(f)

model = load_model()

# ----------------------------
# UI Starts
# ----------------------------
st.title("🛒 BigMart Sales Prediction App")
st.write("Upload dataset OR manually input values to predict Item Outlet Sales.")

# ----------------------------
# File Upload Option
# ----------------------------
st.header("📁 Upload BigMart Test Dataset (CSV)")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    if st.button("Predict from CSV"):
        predictions = model.predict(df)
        df["Predicted_Sales"] = predictions
        st.success("Prediction Completed!")
        st.dataframe(df.head())

        # Download option
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Predicted CSV", csv, "predicted_bigmart.csv")

# ----------------------------
# Manual Prediction Section
# ----------------------------
st.header("📝 Manual Input Prediction")

item_weight = st.number_input("Item Weight", 1.0, 40.0, 12.0)
item_visibility = st.number_input("Item Visibility", 0.0, 0.5, 0.05)
item_mrp = st.number_input("Item MRP", 20.0, 400.0, 120.0)
outlet_years = st.number_input("Outlet Establishment (Years)", 1, 40, 10)

if st.button("Predict Manually"):
    input_df = pd.DataFrame({
        "Item_Weight": [item_weight],
        "Item_Visibility": [item_visibility],
        "Item_MRP": [item_mrp],
        "Outlet_Age": [outlet_years]
    })

    result = model.predict(input_df)[0]
    st.success(f"Predicted Sales: ₹ {result:,.2f}")
