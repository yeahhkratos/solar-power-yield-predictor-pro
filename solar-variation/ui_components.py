import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def inject_custom_styles():
    """Injects modern custom CSS styling for glassmorphic cards, metrics, and typography."""
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Glassmorphism Metric Cards */
        .metric-card-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }

        .metric-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 16px;
            padding: 1.25rem 1.5rem;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.18);
            transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
            position: relative;
            overflow: hidden;
        }

        .metric-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 40px 0 rgba(255, 180, 0, 0.18);
            border-color: rgba(255, 184, 0, 0.4);
        }

        .metric-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 4px;
            background: linear-gradient(90deg, #FFB800 0%, #FF8A00 100%);
        }

        .metric-card.green-accent::before {
            background: linear-gradient(90deg, #00E676 0%, #00B0FF 100%);
        }

        .metric-card.blue-accent::before {
            background: linear-gradient(90deg, #29B6F6 0%, #7E57C2 100%);
        }

        .metric-card.purple-accent::before {
            background: linear-gradient(90deg, #AB47BC 0%, #FF4081 100%);
        }

        .metric-title {
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #A0AEC0;
            margin-bottom: 0.4rem;
        }

        .metric-value {
            font-size: 2.1rem;
            font-weight: 800;
            color: #FFFFFF;
            line-height: 1.1;
            display: flex;
            align-items: baseline;
            gap: 0.3rem;
        }

        .metric-unit {
            font-size: 1.05rem;
            font-weight: 600;
            color: #FFB800;
        }

        .metric-subtitle {
            font-size: 0.78rem;
            color: #718096;
            margin-top: 0.5rem;
        }

        /* Streamlit Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: rgba(0, 0, 0, 0.2);
            padding: 6px;
            border-radius: 12px;
        }

        .stTabs [data-baseweb="tab"] {
            height: 44px;
            border-radius: 8px;
            font-weight: 600;
            color: #A0AEC0;
        }

        .stTabs [aria-selected="true"] {
            background-color: rgba(255, 184, 0, 0.15) !important;
            color: #FFB800 !important;
            border: 1px solid rgba(255, 184, 0, 0.3) !important;
        }
        </style>
    """, unsafe_allow_html=True)


def render_metric_cards(summary: dict):
    """Renders sleek glassmorphic metric cards for Key Performance Indicators."""
    html_content = f"""
    <div class="metric-card-container">
        <div class="metric-card">
            <div class="metric-title">⚡ Today's Expected Energy</div>
            <div class="metric-value">{summary['energy_today_kwh']} <span class="metric-unit">kWh</span></div>
            <div class="metric-subtitle">Physics baseline: {summary['physics_today_kwh']} kWh</div>
        </div>
        <div class="metric-card blue-accent">
            <div class="metric-title">🔥 Peak Power Output</div>
            <div class="metric-value">{summary['peak_kw']} <span class="metric-unit" style="color: #29B6F6;">kW</span></div>
            <div class="metric-subtitle">Max instantaneous generation</div>
        </div>
        <div class="metric-card green-accent">
            <div class="metric-title">💰 Est. Daily Savings</div>
            <div class="metric-value">{summary.get('currency_symbol', '₹')}{summary['cost_savings']}</div>
            <div class="metric-subtitle">Based on local tariff rate</div>
        </div>
        <div class="metric-card purple-accent">
            <div class="metric-title">📅 Total Forecast Yield</div>
            <div class="metric-value">{summary['total_forecast_kwh']} <span class="metric-unit" style="color: #AB47BC;">kWh</span></div>
            <div class="metric-subtitle">Cumulated multi-day generation</div>
        </div>
    </div>
    """
    st.markdown(html_content, unsafe_allow_html=True)


def render_power_irradiance_chart(df: pd.DataFrame) -> go.Figure:
    """
    Renders dual-axis Plotly chart: Hourly Power Output Curve (kW) overlaid with Shortwave Radiation (W/m2).
    """
    fig = make_subplots(
        specs=[[{"secondary_y": True}]],
        subplot_titles=("Hourly Solar Power Output & Irradiance Forecast",)
    )

    # Primary Y-axis: XGBoost Power Output
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["xgb_power_kw"],
            name="XGBoost Estimated Power (kW)",
            mode="lines",
            line=dict(color="#FFB800", width=3),
            fill="tozeroy",
            fillcolor="rgba(255, 184, 0, 0.15)",
            hovertemplate="%{x|%b %d, %H:%M}<br><b>XGBoost Power:</b> %{y:.2f} kW<extra></extra>"
        ),
        secondary_y=False
    )

    # Primary Y-axis: Physics Power Output (Dashed reference line)
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["physics_power_kw"],
            name="Physics Baseline Power (kW)",
            mode="lines",
            line=dict(color="#FF5722", width=2, dash="dash"),
            hovertemplate="%{x|%b %d, %H:%M}<br><b>Physics Power:</b> %{y:.2f} kW<extra></extra>"
        ),
        secondary_y=False
    )

    # Secondary Y-axis: Shortwave Radiation
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["shortwave_radiation"],
            name="Shortwave Radiation (W/m²)",
            mode="lines",
            line=dict(color="#00E676", width=2, dash="dot"),
            hovertemplate="%{x|%b %d, %H:%M}<br><b>Irradiance:</b> %{y:.1f} W/m²<extra></extra>"
        ),
        secondary_y=True
    )

    # Layout styling
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        font=dict(family="Inter, sans-serif", color="#E2E8F0"),
        height=480,
        margin=dict(l=20, r=20, t=50, b=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0.3)"
        ),
        hovermode="x unified"
    )

    fig.update_xaxes(title_text="Time", showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(
        title_text="<b>Power Output (kW)</b>",
        secondary_y=False,
        showgrid=True,
        gridcolor="rgba(255,255,255,0.08)",
        color="#FFB800"
    )
    fig.update_yaxes(
        title_text="<b>Solar Irradiance (W/m²)</b>",
        secondary_y=True,
        showgrid=False,
        color="#00E676"
    )

    return fig


def render_temp_efficiency_chart(df: pd.DataFrame) -> go.Figure:
    """
    Renders timeline chart of Temperature (°C) vs Solar Efficiency Loss (%).
    Highlights thermal stress zones when temperature > 25°C.
    """
    fig = make_subplots(
        specs=[[{"secondary_y": True}]],
        subplot_titles=("Thermal Impact: Ambient Temperature vs Solar Efficiency Derating",)
    )

    # Ambient Temperature (°C)
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["temperature_2m"],
            name="Ambient Temp (°C)",
            mode="lines+markers",
            line=dict(color="#FF7043", width=2.5),
            marker=dict(size=4),
            hovertemplate="%{x|%b %d, %H:%M}<br><b>Temperature:</b> %{y:.1f} °C<extra></extra>"
        ),
        secondary_y=False
    )

    # Standard STC reference line at 25°C
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=[25.0] * len(df),
            name="STC Baseline (25°C)",
            mode="lines",
            line=dict(color="rgba(255,255,255,0.4)", width=1.5, dash="dash"),
            hoverinfo="none"
        ),
        secondary_y=False
    )

    # Efficiency Loss (%)
    fig.add_trace(
        go.Bar(
            x=df["time"],
            y=df["efficiency_loss_pct"],
            name="Thermal Efficiency Loss (%)",
            marker=dict(color="rgba(239, 83, 80, 0.7)"),
            hovertemplate="%{x|%b %d, %H:%M}<br><b>Efficiency Drop:</b> -%{y:.2f}%<extra></extra>"
        ),
        secondary_y=True
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        font=dict(family="Inter, sans-serif", color="#E2E8F0"),
        height=420,
        margin=dict(l=20, r=20, t=50, b=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0.3)"
        ),
        hovermode="x unified"
    )

    fig.update_xaxes(title_text="Time", showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(
        title_text="<b>Temperature (°C)</b>",
        secondary_y=False,
        showgrid=True,
        gridcolor="rgba(255,255,255,0.08)",
        color="#FF7043"
    )
    fig.update_yaxes(
        title_text="<b>Efficiency Loss (%)</b>",
        secondary_y=True,
        showgrid=False,
        color="#EF5350"
    )

    return fig


def render_data_download_section(df: pd.DataFrame):
    """Renders downloadable CSV table and preview."""
    export_df = df[[
        "time",
        "shortwave_radiation",
        "direct_normal_irradiance",
        "temperature_2m",
        "cloud_cover",
        "relative_humidity_2m",
        "physics_power_kw",
        "xgb_power_kw",
        "efficiency_loss_pct"
    ]].copy()

    export_df.columns = [
        "Timestamp",
        "Shortwave Irradiance (W/m2)",
        "Direct Normal Irradiance (W/m2)",
        "Temperature (°C)",
        "Cloud Cover (%)",
        "Relative Humidity (%)",
        "Physics Power Output (kW)",
        "XGBoost Predicted Power (kW)",
        "Thermal Efficiency Loss (%)"
    ]

    csv_bytes = export_df.to_csv(index=False).encode('utf-8')

    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption(f"Showing total of {len(export_df)} hourly forecast data points.")
    with col2:
        st.download_button(
            label="📥 Download CSV Summary",
            data=csv_bytes,
            file_name="solar_power_forecast_summary.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.dataframe(
        export_df,
        use_container_width=True,
        height=380,
        hide_index=True
    )
