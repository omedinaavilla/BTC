import pandas as pd
from sqlalchemy import create_engine

def obtener_datos():
    usuario = "postgres"
    contraseña = "Omar1122"
    host = "db"
    puerto = "5432"
    base_datos = "btc"

    engine = create_engine(f"postgresql+psycopg2://{usuario}:{contraseña}@{host}:{puerto}/{base_datos}")
    df = pd.read_sql("SELECT * FROM btc_data", engine)
    return df
