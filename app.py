import streamlit as st
import pandas as pd
import random


# Sidebar: Region & Inputs
st.sidebar.title("K2-DesertGuard Demo")
st.markdown("*Powered by K2 Think – AI-driven environmental analytics platform.*")

region = st.sidebar.selectbox(
    "Select UAE Region", 
    ["Abu Dhabi", "Dubai", "Sharjah", "Al Ain"]
)
soil_moisture = st.sidebar.slider("Soil Moisture Level (%)", 0, 100, 50)
water_usage = st.sidebar.slider("Water Usage Index", 0, 100, 30)
recent_rainfall = st.sidebar.slider("Recent Rainfall (mm)", 0, 100, 10)


# Main: Map & Prediction
st.title(f"Desertification Risk Monitor – {region}")

region_coords = {
    "Abu Dhabi": [24.466667, 54.366667],
    "Dubai": [25.276987, 55.296249],
    "Sharjah": [25.34626, 55.42093],
    "Al Ain": [24.2075, 55.7447]
}

lat, lon = region_coords[region]

# Interactive Map
map_data = pd.DataFrame({'lat': [lat], 'lon': [lon]})
st.map(map_data, zoom=8)

st.caption("UAE Regional Map (location marker is simulated for demo)")

# Google Maps Link
google_maps_url = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
st.markdown(f"[View {region} on Google Maps]({google_maps_url})")



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


# FIXED: Line chart with proper DataFrame format
years = [2025, 2026, 2027, 2028, 2029]
risk_scores = {
    "Low": [2, 3, 2, 2, 3],
    "Medium": [5, 6, 7, 6, 7],
    "High": [8, 9, 9, 10, 10]
}[risk_level]

# Create DataFrame with proper structure for line chart
chart_data = pd.DataFrame({
    "Year": years,
    "Risk Score": risk_scores
})
st.line_chart(chart_data, x="Year", y="Risk Score", use_container_width=True)
st.caption(f"Risk trend visualization for {region}")


# Reasoning Trace Panel with Expand/Collapse
with st.expander("See AI Reasoning Trace"):
    for step in reasoning_trace:
        st.write(f"• {step}")


# Scenario Testing Panel
st.markdown("#### What-If Scenario")
st.info(f"Try adjusting soil moisture, water use, or rainfall in the sidebar to see impact.")


# Recommendations Section
st.markdown("### AI-Generated Recommendations")
for rec in recommendations.split("\n"):
    if rec.strip():  # Only show non-empty lines
        st.write(rec)


st.caption("Data labels: Satellite Imagery, Govt Water Data, Local Weather Sensors (simulated for demo)")
