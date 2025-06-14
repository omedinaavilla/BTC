import yfinance as yf
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import numpy as np

def interpretar_adf(serie, nombre):
    resultado = adfuller(serie.dropna())
    estadistico = resultado[0]
    pvalor = resultado[1]
    print(f"Resultados ADF - {nombre}")
    print("===================================")
    print(f"Estadístico ADF: {estadistico:.4f}")
    print(f"Valor p: {pvalor:.4f}")
    if pvalor < 0.05:
       print("Rechazamos H0 -> La serie es estacionaria.\n")
    else:
        print("No se rechaza H0 -> La serie NO es estacionaria.\n")

# Descarga de datos
btc = yf.download("BTC-USD", start="2018-12-01", auto_adjust=False)
print("Columnas:", btc.columns)

# Análisis ADF
# Serie original
interpretar_adf(btc['Close'], "Cierre BTC")

# Log-transformada
log_serie = np.log(btc['Close'])
interpretar_adf(log_serie, "Log(Cierre BTC)")

# Diferenciada
diff_log = log_serie.diff()
interpretar_adf(diff_log, "Diferencia de Log(Cierre BTC)")

