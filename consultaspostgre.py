import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

# Cargar variables del archivo .env
load_dotenv()

def obtener_datos():
    usuario = os.getenv("PGUSER")
    contrasena = os.getenv("PGPASSWORD")
    host = os.getenv("PGHOST")
    puerto = os.getenv("PGPORT")
    base_datos = os.getenv("PGDATABASE")

    # Crear conexión a PostgreSQL
    engine = create_engine(f"postgresql+psycopg2://{usuario}:{contrasena}@{host}:{puerto}/{base_datos}")
    
    # Consultar datos
    df = pd.read_sql("SELECT * FROM btc_data", engine)
    return df
