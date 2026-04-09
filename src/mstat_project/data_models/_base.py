import os
from typing import Self

import patito as pt
import polars as pl
from dotenv import load_dotenv


class BaseFileDataModel(pt.Model):
    @staticmethod
    def get_file_path(env_var: str, path_prefix: str | None = None) -> str:
        """Build a file path from environment configuration.

        Args:
            env_var (str): Name of the environment variable that stores the file name.
            path_prefix (str | None, optional) Prefix the path if the relative path changes. Defaults to None

        Raises:
            ValueError: Raised when ``env_var`` is unset or empty.

        Returns:
            str: Path to the file under the configured ``DATA_DIR/files`` directory.
        """

        load_dotenv()

        data_dir: str = os.getenv(key="DATA_DIR", default="data")
        file_name: str | None = os.getenv(key=env_var)

        if not file_name:
            raise ValueError(
                "Environment variable %s could not be read, make sure you have set this variable", env_var
            )

        fp = f"{data_dir}/files/{file_name}"

        if path_prefix:
            fp = f"{path_prefix.removesuffix('/')}/{fp}"

        return fp

    @classmethod
    def from_file(cls, file_path: str) -> pt.DataFrame[Self]:
        """Creates a data frame by reading the file path and validating the data

        Returns:
            pt.DataFrame[Self]: The validated data frame
        """
        raw_df: pt.DataFrame[Self] = (
            pt.DataFrame(data=pl.read_csv(source=file_path)).set_model(model=cls).cast().derive()
        )
        raw_df.validate()
        return raw_df
