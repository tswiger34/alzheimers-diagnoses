from typing import Generator

import polars as pl


def clean_str_col_exprs(raw_df: pl.DataFrame) -> Generator[pl.Expr, None, None]:
    """Creates polars expresions for cleaning and normalizing string columns, except for IDs

    It does this by iterating over all of the string columns in the data frame that are not suffixed with '_id',
    converts it to uppercase, strips trailing and leading white space, then replaces empty strings with `None`
    so that they will register as nulls.

    Args:
        raw_df (pl.DataFrame): The raw data frame that needs to be cleaned

    Yields:
        Generator[Expr, None, None]: Polars expressions for cleaning and normalizing string columns
    """
    for col_name, dtype in raw_df.schema.items():
        if isinstance(dtype, pl.String) and not col_name.endswith("_id"):
            yield pl.col(name=col_name).str.to_uppercase().str.strip_chars().replace(old="", new=None)
        else:
            continue
