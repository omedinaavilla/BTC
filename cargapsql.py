import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv


load_dotenv()

def cargar_a_postgres():
    # Descarga de datos
    btc = yf.download("BTC-USD", start="2018-12-01", auto_adjust=False)
    btc.reset_index(inplace=True)
    btc.columns = btc.columns.get_level_values(0)

    btc.rename(columns={
        "Date": "fecha",
        "Open": "apertura",
        "High": "maximo",
        "Low": "minimo",
        "Close": "cierre",
        "Adj Close": "cierre_ajustado",
        "Volume": "volumen"
    }, inplace=True)

    # Conexión a PostgreSQL usando variables de entorno
    usuario = os.getenv("DB_USER", "postgres")
    contrasena = os.getenv("DB_PASSWORD", "password")
    host = os.getenv("DB_HOST")
    puerto = os.getenv("DB_PORT", "5432")
    base_datos = os.getenv("DB_NAME", "btc")

    engine = create_engine(f"postgresql+psycopg2://{usuario}:{contrasena}@{host}:{puerto}/{base_datos}")

    btc[["fecha", "apertura", "maximo", "minimo", "cierre", "volumen", "cierre_ajustado"]].to_sql(
        'btc_data', engine, if_exists='replace', index=False
    )

    print("✅ Datos guardados en PostgreSQL correctamente")

if __name__ == '__main__':
    cargar_a_postgres()
