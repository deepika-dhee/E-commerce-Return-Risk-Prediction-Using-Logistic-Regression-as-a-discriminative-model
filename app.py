import streamlit as st
import numpy as np
import pandas as pd
import pickle

# PAGE CONFIGURATION

st.set_page_config(
    page_title="E-commerce Return Prediction",
    page_icon="🛍️",
    layout="wide"
)
# CUSTOM CSS
st.markdown("""
<style>

.main-title {
    text-align: center;
    font-size: 38px;
    font-weight: bold;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}

.section-title {
    font-size: 24px;
    font-weight: bold;
    margin-top: 10px;
}

.metric-box {
    background: white;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 3px 12px rgba(0,0,0,0.08);
}

.metric-value {
    font-size: 28px;
    font-weight: bold;
}

.metric-label {
    font-size: 14px;
}

.result-title {
    font-size: 30px;
    font-weight: bold;
}

.high-risk {
    font-size: 28px;
    font-weight: bold;
}

.medium-risk {
    font-size: 28px;
    font-weight: bold;
}

.low-risk {
    font-size: 28px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# LOAD MODEL FILES

@st.cache_resource
def load_model():

    weights = np.load("weights.npy")

    with open("bias.pkl", "rb") as f:
        bias = pickle.load(f)

    with open("encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return weights, bias, encoder, scaler


weights, bias, encoder, scaler = load_model()

# SIGMOID FUNCTION

def sigmoid(z):

    z = np.clip(z, -500, 500)

    return 1 / (1 + np.exp(-z))

# HEADER

st.markdown(
    '<div class="main-title">🛍️ E-commerce Return Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Powered E-commerce Return Prediction System</div>',
    unsafe_allow_html=True
)

# MODEL PERFORMANCE

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        """
        <div class="metric-box">
            <div class="metric-value">86.25%</div>
            <div class="metric-label">ROC-AUC</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="metric-box">
            <div class="metric-value">77.95%</div>
            <div class="metric-label">Accuracy</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class="metric-box">
            <div class="metric-value">46.50%</div>
            <div class="metric-label">Return Rate</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        """
        <div class="metric-box">
            <div class="metric-value">47.79%</div>
            <div class="metric-label">Return Value Impact</div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown("---")

# INPUT SECTION

st.markdown(
    '<div class="section-title">📋 Customer & Order Information</div>',
    unsafe_allow_html=True
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["👤 Customer", "📦 Product", "🚚 Delivery", "📊 Activity"]
)

# CUSTOMER TAB

with tab1:

    col1, col2 = st.columns(2)

    with col1:

        customer_age = st.number_input(
            "Customer Age",
            min_value=18,
            max_value=80,
            value=30
        )

        customer_tenure_months = st.number_input(
            "Customer Tenure (Months)",
            min_value=0,
            max_value=120,
            value=24
        )

        past_purchase_count = st.number_input(
            "Past Purchase Count",
            min_value=0,
            max_value=200,
            value=10
        )

        past_return_rate = st.number_input(
            "Past Return Rate",
            min_value=0.0,
            max_value=1.0,
            value=0.20,
            step=0.01
        )

        used_coupon = st.selectbox(
            "Used Coupon",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

    with col2:

        quantity = st.number_input(
            "Quantity",
            min_value=1,
            max_value=20,
            value=1
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "COD",
                "Credit Card",
                "Upi",
                "Debit Card",
                "Digital Wallet"
            ]
        )

# PRODUCT TAB

with tab2:

    col1, col2 = st.columns(2)

    with col1:

        product_price = st.number_input(
            "Product Price",
            min_value=0.0,
            max_value=10000.0,
            value=100.0,
            step=10.0
        )

        product_rating = st.number_input(
            "Product Rating",
            min_value=1.0,
            max_value=5.0,
            value=4.0,
            step=0.1
        )

        product_category = st.selectbox(
            "Product Category",
            [
                "Beauty",
                "Clothing",
                "Electronics",
                "Footwear",
                "Home & Kitchen",
                "Sports"
            ]
        )

    with col2:

        size_issue = st.selectbox(
            "Size Issue",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        product_condition = st.selectbox(
            "Product Condition",
            [
                "damaged",
                "good",
                "minor_issue"
            ]
        )

        discount_percent = st.number_input(
            "Discount Percentage",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=1.0
        )

# DELIVERY TAB

with tab3:

    col1, col2 = st.columns(2)

    with col1:

        shipping_method = st.selectbox(
            "Shipping Method",
            [
                "Express",
                "Next Day",
                "Standard"
            ]
        )

        expected_delivery_days = st.number_input(
            "Expected Delivery Days",
            min_value=1,
            max_value=30,
            value=5
        )

    with col2:

        actual_delivery_days = st.number_input(
            "Actual Delivery Days",
            min_value=1,
            max_value=60,
            value=5
        )

        delivery_delay_days = st.number_input(
            "Delivery Delay Days",
            min_value=0,
            max_value=30,
            value=0
        )
# ACTIVITY TAB

with tab4:

    col1, col2 = st.columns(2)

    with col1:

        session_length_minutes = st.number_input(
            "Session Length (Minutes)",
            min_value=1,
            max_value=500,
            value=20
        )

        num_product_views = st.number_input(
            "Number of Product Views",
            min_value=0,
            max_value=500,
            value=10
        )

        num_previous_visits = st.number_input(
            "Number of Previous Visits",
            min_value=0,
            max_value=200,
            value=5
        )

    with col2:

        cart_additions = st.number_input(
            "Cart Additions",
            min_value=0,
            max_value=100,
            value=2
        )

        wishlist_additions = st.number_input(
            "Wishlist Additions",
            min_value=0,
            max_value=100,
            value=1
        )


st.markdown("---")

# PREDICTION BUTTON

predict_button = st.button(
    "🚀 Predict Return Risk",
    use_container_width=True
)

# PREDICTION

if predict_button:

    try:

        # NUMERICAL FEATURES

        numerical_cols = [
            "customer_age",
            "customer_tenure_months",
            "past_purchase_count",
            "past_return_rate",
            "product_price",
            "product_rating",
            "size_issue",
            "discount_percent",
            "used_coupon",
            "quantity",
            "expected_delivery_days",
            "actual_delivery_days",
            "delivery_delay_days",
            "session_length_minutes",
            "num_product_views",
            "num_previous_visits",
            "cart_additions",
            "wishlist_additions"
        ]

        numerical_data = {
            "customer_age": customer_age,
            "customer_tenure_months": customer_tenure_months,
            "past_purchase_count": past_purchase_count,
            "past_return_rate": past_return_rate,
            "product_price": product_price,
            "product_rating": product_rating,
            "size_issue": size_issue,
            "discount_percent": discount_percent,
            "used_coupon": used_coupon,
            "quantity": quantity,
            "expected_delivery_days": expected_delivery_days,
            "actual_delivery_days": actual_delivery_days,
            "delivery_delay_days": delivery_delay_days,
            "session_length_minutes": session_length_minutes,
            "num_product_views": num_product_views,
            "num_previous_visits": num_previous_visits,
            "cart_additions": cart_additions,
            "wishlist_additions": wishlist_additions
        }

        numerical_df = pd.DataFrame(
            [numerical_data],
            columns=numerical_cols
        )


        # CATEGORICAL FEATURES

        categorical_cols = [
            "product_category",
            "product_condition",
            "payment_method",
            "shipping_method"
        ]

        categorical_data = {
            "product_category": product_category,
            "product_condition": product_condition,
            "payment_method": payment_method,
            "shipping_method": shipping_method
        }

        categorical_df = pd.DataFrame(
            [categorical_data],
            columns=categorical_cols
        )

        # SCALE NUMERICAL FEATURES

        numerical_scaled = scaler.transform(
            numerical_df
        )

        numerical_scaled_df = pd.DataFrame(
            numerical_scaled,
            columns=numerical_cols
        )

        # ONE-HOT ENCODE CATEGORICAL FEATURES

        encoded = encoder.transform(
            categorical_df
        )

        encoded_df = pd.DataFrame(
            encoded,
            columns=encoder.get_feature_names_out(
                categorical_cols
            )
        )

        # COMBINE FEATURES

        X_input = pd.concat(
            [
                numerical_scaled_df,
                encoded_df
            ],
            axis=1
        )

        X_input = X_input.to_numpy()

        # CHECK FEATURE COUNT

        if X_input.shape[1] != len(weights):

            raise ValueError(
                f"Model expects {len(weights)} features, "
                f"but received {X_input.shape[1]} features."
            )

        # LOGISTIC REGRESSION PREDICTION

        z = np.dot(
            X_input,
            weights
        ) + bias

        probability = sigmoid(z)[0]

        # RISK LEVEL

        if probability >= 0.60:

            risk_level = "HIGH RISK"
            risk_class = "high-risk"

        elif probability >= 0.30:

            risk_level = "MEDIUM RISK"
            risk_class = "medium-risk"

        else:

            risk_level = "LOW RISK"
            risk_class = "low-risk"


        # RESULT

        st.markdown("---")

        st.markdown(
            '<div class="result-title">🔮 Return Risk Prediction</div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Return Probability",
                f"{probability * 100:.2f}%"
            )

            st.markdown(
                f'<div class="{risk_class}">{risk_level}</div>',
                unsafe_allow_html=True
            )

        # ORDER SUMMARY

        with col2:

            st.subheader("📦 Order Summary")

            st.write(
                f"**Product Category:** {product_category}"
            )

            st.write(
                f"**Product Price:** ₹{product_price:.2f}"
            )

            st.write(
                f"**Discount:** {discount_percent:.1f}%"
            )

            st.write(
                f"**Shipping:** {shipping_method}"
            )

            st.write(
                f"**Delivery Delay:** {delivery_delay_days} days"
            )

        # BUSINESS IMPACT

        st.markdown("---")

        st.subheader("💰 Estimated Business Impact")

        estimated_return_value = (
            product_price * probability
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Product Value",
                f"₹{product_price:.2f}"
            )

        with col2:

            st.metric(
                "Estimated Return Value",
                f"₹{estimated_return_value:.2f}"
            )

        with col3:

            st.metric(
                "Return Probability",
                f"{probability * 100:.2f}%"
            )

        # RECOMMENDATION

        st.markdown("---")

        st.subheader("💡 Recommendation")

        if probability >= 0.60:

            st.warning(
                "High return risk detected. "
                "Consider checking product condition, "
                "delivery performance, and customer history."
            )

        elif probability >= 0.30:

            st.info(
                "Moderate return risk detected. "
                "Monitor the order and provide accurate "
                "product and delivery information."
            )

        else:

            st.success(
                "Low return risk detected. "
                "The order is less likely to be returned."
            )

    except Exception as e:

        st.error(
            f"Prediction error: {e}"
        )

# FOOTER
st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;">
        E-commerce Return Prediction | 
        Logistic Regression using Gradient Descent
    </div>
    """,
    unsafe_allow_html=True
)