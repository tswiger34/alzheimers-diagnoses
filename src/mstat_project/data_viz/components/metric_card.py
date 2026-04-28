from typing import Any, TypedDict


class MetricCard(TypedDict):
    title: str
    main_value: Any
    eyebrow: str
    description: str
    sub_metrics: list[Any]
