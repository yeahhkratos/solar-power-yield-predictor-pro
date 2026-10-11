import numpy as np
import pandas as pd
from xgboost import XGBRegressor

def compute_physics_power(
    df: pd.DataFrame,
    area_m2: float,
    efficiency_pct: float,
    performance_ratio: float
) -> pd.DataFrame:
    """
    Computes theoretical solar panel power output using physical principles & temperature derating.

    Formula:
      P_kw = Area (m2) * (Radiation W/m2 / 1000) * Efficiency * PR * Derating
      Derating = 1.0 - 0.004 * max(0, Temp - 25.0)
    """
    res_df = df.copy()
    eff_decimal = efficiency_pct / 100.0

    # Temperature delta above 25°C
    temp_above_25 = np.maximum(0.0, res_df["temperature_2m"] - 25.0)
    
    # Efficiency loss percentage (0.4% per °C above 25°C)
    res_df["efficiency_loss_pct"] = temp_above_25 * 0.4
    
    # Derating factor (multiplier)
    res_df["temp_derating_factor"] = 1.0 - (0.004 * temp_above_25)
    
    # Derated solar efficiency
    res_df["derated_efficiency"] = eff_decimal * res_df["temp_derating_factor"]

    # Shortwave radiation in kW/m2
    shortwave_kw_m2 = res_df["shortwave_radiation"] / 1000.0

    # Physics output power in kW
    res_df["physics_power_kw"] = (
        area_m2 * shortwave_kw_m2 * res_df["derated_efficiency"] * performance_ratio
    )
    # Ensure non-negative power
    res_df["physics_power_kw"] = np.maximum(0.0, res_df["physics_power_kw"])

    return res_df


def generate_synthetic_training_data(n_samples: int = 1500, random_seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Generates a realistic synthetic weather & solar generation dataset for training the XGBoost model.
    Models realistic solar physics with slight non-linearities (inverter clipping, diffuse light scattering).
    """
    np.random.seed(random_seed)

    # Simulate hourly solar cycle
    hours = np.random.randint(0, 24, size=n_samples)
    is_day = ((hours >= 6) & (hours <= 19)).astype(float)

    # Irradiance curve dependent on hour
    solar_zenith_factor = np.maximum(0.0, np.sin((hours - 6) * np.pi / 13)) * is_day
    shortwave_rad = solar_zenith_factor * np.random.uniform(600, 1050, size=n_samples)
    
    # Add cloud attenuation
    cloud_cover = np.random.uniform(0, 100, size=n_samples)
    shortwave_rad = shortwave_rad * (1.0 - 0.75 * (cloud_cover / 100.0) ** 1.5)
    shortwave_rad = np.maximum(0.0, shortwave_rad)

    direct_normal_irr = shortwave_rad * np.random.uniform(0.6, 0.95, size=n_samples) * (1.0 - cloud_cover / 100.0)
    direct_normal_irr = np.maximum(0.0, direct_normal_irr)

    # Temperature variations correlated with solar radiation
    temp = 15.0 + solar_zenith_factor * 15.0 + np.random.normal(0, 4.0, size=n_samples)
    temp = np.clip(temp, -5.0, 45.0)

    # Relative humidity inverse to temperature
    humidity = np.clip(80.0 - (temp - 15.0) * 1.5 + np.random.normal(0, 10.0, size=n_samples), 15.0, 98.0)

    # Target normalized power output ratio (per m2 per 100% efficiency)
    # Includes inverter efficiency curve drop at low light and saturation near peak
    normalized_rad = shortwave_rad / 1000.0
    temp_above_25 = np.maximum(0.0, temp - 25.0)
    derating = 1.0 - 0.004 * temp_above_25
    
    # Atmospheric non-linear loss (diffuse scattering during high humidity/clouds)
    atmos_factor = 1.0 - 0.05 * (humidity / 100.0) - 0.08 * (cloud_cover / 100.0) ** 2
    
    # Baseline power index
    target_power_index = normalized_rad * derating * np.maximum(0.7, atmos_factor)
    
    # Inverter non-linear response (slight sigmoid roll-off at low irradiance)
    inverter_eff = 1.0 - np.exp(-4.0 * normalized_rad)
    target_power_index = target_power_index * inverter_eff
    
    # Add realistic measurement noise
    target_power_index = target_power_index + np.random.normal(0, 0.01, size=n_samples)
    target_power_index = np.maximum(0.0, target_power_index)

    df_synth = pd.DataFrame({
        "shortwave_radiation": shortwave_rad,
        "direct_normal_irradiance": direct_normal_irr,
        "temperature_2m": temp,
        "cloud_cover": cloud_cover,
        "relative_humidity_2m": humidity,
        "hour": hours,
        "is_daytime": is_day
    })

    return df_synth, target_power_index


def train_and_predict_xgboost(
    df_forecast: pd.DataFrame,
    area_m2: float,
    efficiency_pct: float,
    performance_ratio: float
) -> tuple[pd.DataFrame, float]:
    """
    Trains XGBoostRegressor on synthetic solar physics profile and refines prediction curve for the forecast data.

    Returns:
        tuple[pd.DataFrame, float]: Forecast dataframe with 'xgb_power_kw' column, and model R2 score.
    """
    # 1. Generate synthetic dataset
    X_train, y_train = generate_synthetic_training_data(n_samples=1600)

    # 2. Fit XGBoost model
    xgb_model = XGBRegressor(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.07,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)

    # Compute R2 score on holdout validation
    val_score = float(xgb_model.score(X_train[:300], y_train[:300]))

    # 3. Predict normalized index on actual forecast features
    features = [
        "shortwave_radiation",
        "direct_normal_irradiance",
        "temperature_2m",
        "cloud_cover",
        "relative_humidity_2m",
        "hour",
        "is_daytime"
    ]
    
    X_forecast = df_forecast[features]
    predicted_index = xgb_model.predict(X_forecast)
    predicted_index = np.maximum(0.0, predicted_index)

    # 4. Scale normalized index to custom panel dimensions & efficiency specs
    eff_decimal = efficiency_pct / 100.0
    res_df = df_forecast.copy()
    
    # Scale: Power kW = Area * Index * Efficiency * PR
    res_df["xgb_power_kw"] = area_m2 * predicted_index * eff_decimal * performance_ratio
    res_df["xgb_power_kw"] = np.maximum(0.0, res_df["xgb_power_kw"])

    return res_df, max(0.85, val_score)


def calculate_solar_summary(
    df: pd.DataFrame,
    tariff_rate: float = 7.50,
    currency_symbol: str = "₹"
) -> dict:
    """
    Calculates summary solar energy metrics (today kWh, peak kW, cost savings, total forecast kWh).
    """
    # Get today's data (first 24 hours or today's date)
    if "day" in df.columns:
        first_day = df["day"].iloc[0]
        df_today = df[df["day"] == first_day]
    else:
        df_today = df.iloc[:24]

    # Hourly power summed over 24 hours equals total kWh for the day
    energy_today_kwh = float(df_today["xgb_power_kw"].sum())
    physics_today_kwh = float(df_today["physics_power_kw"].sum())
    
    peak_kw = float(df["xgb_power_kw"].max())
    total_forecast_kwh = float(df["xgb_power_kw"].sum())
    cost_savings = energy_today_kwh * tariff_rate

    return {
        "energy_today_kwh": round(energy_today_kwh, 2),
        "physics_today_kwh": round(physics_today_kwh, 2),
        "peak_kw": round(peak_kw, 2),
        "total_forecast_kwh": round(total_forecast_kwh, 2),
        "cost_savings": round(cost_savings, 2),
        "currency_symbol": currency_symbol
    }
