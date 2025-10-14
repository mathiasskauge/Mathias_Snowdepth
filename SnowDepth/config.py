FEATURE_NAMES = [
    # Base backscatter in dB
    "Sigma_VH", "Sigma_VV",
    "Gamma_VH", "Gamma_VV",
    "Beta_VH",  "Beta_VV",
    "Gamma_VH_RTC", "Gamma_VV_RTC",
    # Linear sums/differences
    "Sigma_sum", "Gamma_sum", "Beta_sum", "Gamma_RTC_sum",
    "Sigma_diff", "Gamma_diff", "Beta_diff", "Gamma_RTC_diff",
    # Ratios (in dB)
    "Sigma_ratio", "Gamma_ratio", "Beta_ratio", "Gamma_RTC_ratio",
    # Angles
    "LIA", "IAFE",
    # Topography
    "Elevation", "Slope", "sin_Aspect", "cos_Aspect", "Veg_Heigth"
]