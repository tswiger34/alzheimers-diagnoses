import polars as pl

from mstat_project.data_viz.components.metric_card import MetricCard  # noqa
from mstat_project.utils import get_db_engine


def get_dataset(tbl_name: str, limit: int | None = None) -> pl.DataFrame:
    eng = get_db_engine()
    with eng.connect() as conn:
        query = "SELECT * FROM %s" % tbl_name
        if limit is not None:
            query = "%s LIMIT %i" % (query, limit)
        df = pl.read_database(query, connection=conn)
    return df
