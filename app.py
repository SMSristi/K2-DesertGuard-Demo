import streamlit as st
import random

# Sidebar: Region & Inputs
st.sidebar.title("K2-DesertGuard Demo")
region = st.sidebar.selectbox(
    "Select UAE Region", 
    ["Abu Dhabi", "Dubai", "Sharjah", "Al Ain"]
)
soil_moisture = st.sidebar.slider("Soil Moisture Level (%)", 0, 100, 50)
water_usage = st.sidebar.slider("Water Usage Index", 0, 100, 30)
recent_rainfall = st.sidebar.slider("Recent Rainfall (mm)", 0, 100, 10)

# Main: Map & Prediction
st.title(f"Desertification Risk Monitor – {region}")

st.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/UAE_Regions_map.png/1200px-UAE_Regions_map.png?20140916094820",
    caption="UAE Regional Map (Satellite Source: Simulated)"
)

# Simulated 'chain-of-thought' reasoning for demo
def k2_think_reasoning(soil, water, rain):
    trace = [
        f"Analyzed satellite imagery for {region}.",
        f"Checked soil moisture: {soil}%.",
        f"Assessed water usage index: {water}.",
        f"Measured recent rainfall: {rain} mm.",
    ]
    # Simulated logic
    if soil < 30 and water > 70 and rain < 20:
        trace.append("Detected high water demand, low moisture, minimal rainfall.")
        risk = "High"
        reasons = (
            "- Immediate reduction in water usage needed.\n"
            "- Switch to drought-resistant crops.\n"
            "- Community awareness and soil restoration programs."
        )
    elif soil < 50 or water > 50:
        trace.append("Moderate stress signals: watch trends closely.")
        risk = "Medium"
        reasons = (
            "- Maintain irrigation controls.\n"
            "- Monthly soil health monitoring.\n"
            "- Encourage water-saving practices."
        )
    else:
        trace.append("Conditions stable; risk factors low.")
        risk = "Low"
        reasons = (
            "- Maintain current sustainable practices.\n"
            "- Regular soil and water monitoring.\n"
            "- Support land conservation initiatives."
        )
    return risk, trace, reasons

risk_level, reasoning_trace, recommendations = k2_think_reasoning(soil_moisture, water_usage, recent_rainfall)

# Display Prediction
icon = { "High":"🟥", "Medium":"🟨", "Low":"🟩" }
st.subheader(f"{icon[risk_level]} Desertification Risk Prediction: {risk_level}")

years = [2025, 2026, 2027, 2028, 2029]
risk_scores = {
    "Low": [2, 3, 2, 2, 3],
    "Medium": [5, 6, 7, 6, 7],
    "High": [8, 9, 9, 10, 10]
}[risk_level]
st.line_chart({"Risk Level": risk_scores}, use_container_width=True)

# Reasoning Trace Panel with Expand/Collapse
with st.expander("See AI Reasoning Trace"):
    for step in reasoning_trace:
        st.write(step)

# Scenario Testing Panel
st.markdown("#### What-If Scenario")
st.info(f"Try adjusting soil moisture, water use, or rainfall in the sidebar to see impact.")

# Recommendations Section
st.markdown("### AI-Generated Recommendations")
for rec in recommendations.split("\n"):
    st.write(rec)

st.caption("Data labels: Satellite Imagery, Govt Water Data, Local Weather Sensors (simulated for demo)")
