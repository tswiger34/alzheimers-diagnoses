from typing import Self

import polars as pl
import pydantic


class DataDictionaryField(pydantic.BaseModel):
    field_name: str
    vals: list[str | int] | None
    val_descriptions: dict[str | int, str] | None


class DataDictionary(pydantic.BaseModel):
    fields: list[DataDictionaryField]

    @classmethod
    def from_df(cls) -> Self:
        import os

        from dotenv import load_dotenv

        load_dotenv()

        tbl_names = {"MRIMETA"}
        data_dir: str = os.getenv(key="DATA_DIR", default="data")
        file_name = os.getenv(key="FILE_NAME_DATADIC", default="DATADIC.csv")

        df: pl.DataFrame = pl.read_csv(source=f"{data_dir}/files/{file_name}")
