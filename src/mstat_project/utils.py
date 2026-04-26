import os

import sqlalchemy


def get_db_engine() -> sqlalchemy.Engine:
    user = os.getenv(key="POSTGRES_USER", default="postgres")
    password = os.getenv(key="POSTGRES_PASSWORD")
    db_name = os.getenv(key="POSTGRES_DB_NAME", default="mstat-db")
    host = os.getenv(key="POSTGRES_HOST", default="127.0.0.1")
    port = os.getenv(key="POSTGRES_PORT", default="5434")

    if password is None:
        raise ValueError("No password set for connecting to Postgres DB")

    url = f"postgresql+psycopg2://{user}:{password}@{host}:{int(port)}/{db_name}"
    return sqlalchemy.create_engine(url)


if __name__ == "__main__":
    import polars as pl

    eng = get_db_engine()
    with eng.connect() as conn:
        df = pl.read_database(query="SELECT * FROM _stg.stg_image_manifest LIMIT 100", connection=conn)
        print(df.head())
