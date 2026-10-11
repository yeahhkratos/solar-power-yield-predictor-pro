import streamlit as st
import pandas as pd
import numpy as np

from weather_service import fetch_weather_forecast
from solar_model import (
    compute_physics_power,
    train_and_predict_xgboost,
    calculate_solar_summary
)
from ui_components import (
    inject_custom_styles,
    render_metric_cards,
    render_power_irradiance_chart,
    render_temp_efficiency_chart,
    render_data_download_section
)

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Solar Power Yield Predictor AI",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom modern glassmorphic CSS
inject_custom_styles()

# Preset Locations Dictionary
PRESET_LOCATIONS = {
    "Raghogarh-Vijaipur, MP 🇮🇳": (24.4442, 77.3150),
    "New Delhi, India 🇮🇳": (28.6139, 77.2090),
    "Tokyo, Japan 🇯🇵": (35.6762, 139.6503),
    "Berlin, Germany 🇩🇪": (52.5200, 13.4050),
    "Los Angeles, USA 🇺🇸": (34.0522, -118.2437),
    "Sydney, Australia 🇦🇺": (-33.8688, 151.2093),
    "Custom Location 📍": (None, None)
}

def main():
    # Header & Hero Section
    st.markdown("""
        <div style="text-align: center; padding: 1rem 0 2rem 0;">
            <h1 style="font-size: 2.8rem; font-weight: 800; background: linear-gradient(135deg, #FFD54F 0%, #FF8F00 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.3rem;">
                ☀️ Solar Panel Power Yield Predictor
            </h1>
            <p style="font-size: 1.1rem; color: #A0AEC0; max-width: 800px; margin: 0 auto;">
                Production-grade solar irradiance forecasting & hourly output estimation combining physical thermal dynamics with synthetic XGBoost machine learning.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar User Input Controls
    st.sidebar.markdown("## ⚙️ Configuration")
    
    # 1. Panel Specifications
    st.sidebar.markdown("### 📐 Solar Array Specs")
    area_m2 = st.sidebar.number_input(
        "Area of Solar Panel Array (m²)",
        min_value=1.0,
        max_value=10000.0,
        value=10.0,
        step=1.0,
        help="Total active surface area of your solar photovoltaic installation in square meters."
    )

    efficiency_pct = st.sidebar.slider(
        "Panel Efficiency (%)",
        min_value=10.0,
        max_value=25.0,
        value=18.0,
        step=0.5,
        help="Manufacturer rated solar panel efficiency under standard test conditions (STC)."
    )

    performance_ratio = st.sidebar.slider(
        "Performance Ratio (PR) / Loss Factor",
        min_value=0.70,
        max_value=0.95,
        value=0.75,
        step=0.01,
        help="System efficiency factor accounting for inverter losses, cabling, dirt/soiling, and shading (default 0.75)."
    )

    currency_choice = st.sidebar.selectbox(
        "Currency Unit",
        options=["₹ INR", "$ USD"],
        index=0
    )
    currency_symbol = "₹" if "INR" in currency_choice else "$"
    default_tariff = 7.50 if currency_symbol == "₹" else 0.15
    tariff_step = 0.50 if currency_symbol == "₹" else 0.01

    tariff_rate = st.sidebar.number_input(
        f"Electricity Tariff ({currency_symbol}/kWh)",
        min_value=0.01,
        max_value=500.0,
        value=default_tariff,
        step=tariff_step,
        help="Local electricity cost per kilowatt-hour for financial yield estimation."
    )

    # 2. Location & Forecast
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🌐 Location & Forecast Window")

    selected_preset = st.sidebar.selectbox(
        "Location Presets",
        options=list(PRESET_LOCATIONS.keys()),
        index=0
    )

    preset_lat, preset_lon = PRESET_LOCATIONS[selected_preset]
    default_lat = preset_lat if preset_lat is not None else 24.4442
    default_lon = preset_lon if preset_lon is not None else 77.3150

    lat = st.sidebar.number_input(
        "Latitude (°)",
        min_value=-90.0,
        max_value=90.0,
        value=default_lat,
        format="%.4f"
    )

    lon = st.sidebar.number_input(
        "Longitude (°)",
        min_value=-180.0,
        max_value=180.0,
        value=default_lon,
        format="%.4f"
    )

    forecast_days = st.sidebar.radio(
        "Forecast Duration",
        options=[3, 5, 7],
        index=0,
        format_func=lambda x: f"{x} Days",
        horizontal=True
    )

    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Model Note**: Weather forecasts are fetched real-time from the Open-Meteo Global Forecasting API.")

    # Main Execution Flow
    with st.spinner("🛰️ Fetching Open-Meteo weather forecast data & running predictions..."):
        try:
            df_weather = fetch_weather_forecast(lat=lat, lon=lon, forecast_days=forecast_days)
        except Exception as e:
            st.error("Failed to load weather data. Please adjust inputs or retry.")
            return

    # Calculate Physics Model Power & Derating
    df_physics = compute_physics_power(
        df=df_weather,
        area_m2=area_m2,
        efficiency_pct=efficiency_pct,
        performance_ratio=performance_ratio
    )

    # Train & Predict with XGBoost ML Model
    df_full, r2_score = train_and_predict_xgboost(
        df_forecast=df_physics,
        area_m2=area_m2,
        efficiency_pct=efficiency_pct,
        performance_ratio=performance_ratio
    )

    # Compute Summary Statistics
    summary = calculate_solar_summary(df_full, tariff_rate=tariff_rate, currency_symbol=currency_symbol)

    # Render Metric Cards Top Section
    render_metric_cards(summary)

    # Interactive Dashboard Tabs
    tab_charts, tab_comparison, tab_table, tab_specs = st.tabs([
        "📊 Forecast & Generation Charts",
        "🤖 XGBoost vs Physics Model",
        "📋 Hourly Data & CSV Export",
        "⚙️ System Parameters Summary"
    ])

    with tab_charts:
        st.markdown("### 📈 Solar Power Output vs Solar Irradiance")
        fig_power = render_power_irradiance_chart(df_full)
        st.plotly_chart(fig_power, use_container_width=True)

        st.markdown("### 🌡️ Thermal Derating Timeline (Temperature vs Efficiency Loss)")
        fig_temp = render_temp_efficiency_chart(df_full)
        st.plotly_chart(fig_temp, use_container_width=True)

    with tab_comparison:
        st.markdown("### 🤖 XGBoost ML Model vs Physics Baseline")
        st.markdown(f"""
            > **Model Confidence Index**: XGBoost model validation $R^2 = {r2_score:.3f}$
            > 
            > The physics model computes linear theoretical power based strictly on $P = A \\times G \\times \\eta \\times PR$ (adjusted for temperature). 
            > The **XGBoost model** refines this by learning complex multi-variable interactions like diffuse light scattering during heavy cloud cover, atmospheric relative humidity attenuation, and non-linear inverter efficiency curves.
        """)

        # Delta comparison chart
        df_full["power_diff_kw"] = df_full["xgb_power_kw"] - df_full["physics_power_kw"]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Total Expected Energy Today (XGBoost)",
                value=f"{summary['energy_today_kwh']} kWh",
                delta=f"{summary['energy_today_kwh'] - summary['physics_today_kwh']:.2f} kWh vs Physics"
            )
        with col2:
            st.metric(
                label="Peak Power Generation",
                value=f"{summary['peak_kw']} kW",
                delta=f"{r2_score*100:.1f}% Model Accuracy"
            )

        st.dataframe(
            df_full[["time", "shortwave_radiation", "temperature_2m", "cloud_cover", "physics_power_kw", "xgb_power_kw", "power_diff_kw"]]
            .rename(columns={
                "time": "Timestamp",
                "shortwave_radiation": "Irradiance (W/m²)",
                "temperature_2m": "Temp (°C)",
                "cloud_cover": "Cloud Cover (%)",
                "physics_power_kw": "Physics kW",
                "xgb_power_kw": "XGBoost kW",
                "power_diff_kw": "Refinement Delta (kW)"
            }),
            use_container_width=True,
            hide_index=True
        )

    with tab_table:
        st.markdown("### 📋 Complete Hourly Forecast & Output Table")
        render_data_download_section(df_full)

    with tab_specs:
        st.markdown("### ⚙️ Solar Installation Parameters")
        spec_col1, spec_col2 = st.columns(2)

        with spec_col1:
            st.markdown(f"""
                - **Array Surface Area**: `{area_m2:.1f} m²`
                - **Rated STC Efficiency**: `{efficiency_pct:.1f}%`
                - **System Performance Ratio**: `{performance_ratio:.2f}`
                - **Location Coordinates**: Lat `{lat:.4f}°`, Lon `{lon:.4f}°`
            """)

        with spec_col2:
            st.markdown(f"""
                - **Thermal Derating Coefficient**: `-0.40% / °C` (above 25°C)
                - **Forecast Horizon**: `{forecast_days} Days` ({len(df_full)} hourly samples)
                - **Electricity Tariff Rate**: `{currency_symbol}{tariff_rate:.2f} / kWh`
                - **Data Source**: Open-Meteo High-Resolution Global Forecast
            """)

if __name__ == "__main__":
    main()
