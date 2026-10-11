import requests
import pandas as pd
import streamlit as st

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_weather_forecast(lat: float, lon: float, forecast_days: int = 3) -> pd.DataFrame:
    """
    Fetches hourly weather forecast data from the Open-Meteo API for a given location.

    Parameters:
        lat (float): Latitude
        lon (float): Longitude
        forecast_days (int): Forecast duration in days (3, 5, or 7)

    Returns:
        pd.DataFrame: Hourly dataframe containing weather variables.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "shortwave_radiation",
            "direct_normal_irradiance",
            "temperature_2m",
            "cloud_cover",
            "relative_humidity_2m"
        ],
        "forecast_days": forecast_days,
        "timezone": "auto"
    }

    headers = {
        "User-Agent": "SolarPanelYieldPredictor/1.0"
    }

    try:
        response = requests.get(OPEN_METEO_URL, params=params, headers=headers, timeout=12)
        response.raise_for_status()
        data = response.json()

        if "hourly" not in data:
            raise ValueError("Invalid payload received from Open-Meteo API: missing 'hourly' data field.")

        hourly_data = data["hourly"]
        df = pd.DataFrame(hourly_data)

        # Convert time to datetime index
        df["time"] = pd.to_datetime(df["time"])
        
        # Ensure numerical types and handle any missing values
        numeric_cols = [
            "shortwave_radiation",
            "direct_normal_irradiance",
            "temperature_2m",
            "cloud_cover",
            "relative_humidity_2m"
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            else:
                df[col] = 0.0

        # Feature engineering: hour of day, daytime flag
        df["hour"] = df["time"].dt.hour
        df["day"] = df["time"].dt.date
        df["is_daytime"] = (df["shortwave_radiation"] > 5.0).astype(int)

        return df

    except requests.exceptions.Timeout:
        st.error("⏰ Open-Meteo API request timed out (12s). Please try again or check your internet connection.")
        raise RuntimeError("API Timeout")
    except requests.exceptions.HTTPError as http_err:
        st.error(f"🌐 HTTP error occurred while contacting Open-Meteo API: {http_err}")
        raise RuntimeError(f"HTTP Error: {http_err}")
    except requests.exceptions.RequestException as req_err:
        st.error(f"🔌 Network failure connecting to Open-Meteo API: {req_err}")
        raise RuntimeError(f"Request Error: {req_err}")
    except Exception as e:
        st.error(f"⚠️ Unexpected error parsing weather data: {e}")
        raise e
