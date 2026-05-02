import dagster as dg


@dg.asset(
    name="raw_images",
    partitions_def=dg.StaticPartitionsDefinition(partition_keys=[f"cohort_{i}" for i in range(1, 13)]),
    kinds={"Python"},
)
def raw_images(): ...
