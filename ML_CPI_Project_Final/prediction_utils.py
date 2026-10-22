
import numpy as np
import pandas as pd

def get_weather_season(month):
    if month in [12, 1, 2]:
        return "Cool_Dry"
    elif month in [3, 4, 5]:
        return "Hot_Dry"
    elif month in [6, 7, 8]:
        return "Rainy"
    else:
        return "Rainy_Transition"


def prepare_prediction_input(
    geolocation,
    commodity_code,
    commodity_name,
    year,
    month,
    current_cpi,
    cpi_lag_1,
    cpi_lag_2,
    cpi_lag_3
):
    
    quarter = (month - 1) // 3 + 1
    
    cpi_change = current_cpi - cpi_lag_1
    
    cpi_pct_change = (
        (current_cpi - cpi_lag_1) / cpi_lag_1
    )
    
    previous_cpis = [
        cpi_lag_1,
        cpi_lag_2,
        cpi_lag_3
    ]
    
    rolling_mean_3 = np.mean(previous_cpis)
    
    rolling_std_3 = np.std(
        previous_cpis,
        ddof=1
    )
    
    month_sin = np.sin(
        2 * np.pi * month / 12
    )
    
    month_cos = np.cos(
        2 * np.pi * month / 12
    )
    
    weather_season = get_weather_season(month)
    
    prediction_input = pd.DataFrame([{
        "Geolocation": geolocation,
        "Commodity_Code": commodity_code,
        "Commodity_Name": commodity_name,
        "CPI": current_cpi,
        "Year": year,
        "Month": month,
        "Quarter": quarter,
        "CPI_Lag_1": cpi_lag_1,
        "CPI_Lag_2": cpi_lag_2,
        "CPI_Lag_3": cpi_lag_3,
        "CPI_Change": cpi_change,
        "CPI_Pct_Change": cpi_pct_change,
        "CPI_Rolling_Mean_3": rolling_mean_3,
        "CPI_Rolling_Std_3": rolling_std_3,
        "Month_Sin": month_sin,
        "Month_Cos": month_cos,
        "Weather_Season": weather_season
    }])
    
    return prediction_input
