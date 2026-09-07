with
    follow_up_schedule as (
        select *
        from (
                values (1, 6), (2, 12), (3, 24), (4, 36), (5, 48), (6, 60), (7, 72), (8, 84), (9, 96), (10, 108), (11, 120), (12, 132), (13, 144), (14, 156), (15, 168), (16, 180)
            ) as t (visit_num, follow_up_months)
    ),
    follow_up_bins as (
        select
            visit_num,
            'v' || visit_num as visit_code,
            follow_up_months,
            case
                when lag(follow_up_months) over (
                    order by follow_up_months
                ) is null then 0
                else (
                    lag(follow_up_months) over (
                        order by follow_up_months
                    ) + follow_up_months
                ) / 2.0
            end::NUMERIC(10, 2) as follow_up_start,
            case
                when lead(follow_up_months) over (
                    order by follow_up_months
                ) is null then 180
                else (
                    follow_up_months + lead(follow_up_months) over (
                        order by follow_up_months
                    )
                ) / 2.0
            end::NUMERIC(10, 2) as follow_up_end
        from follow_up_schedule
    )
select
    0 as visit_num,
    'BL' as visit_code,
    0 as follow_up_months,
    0 as follow_up_start,
    0 as follow_up_end
union all
select
    visit_num,
    visit_code,
    follow_up_months,
    follow_up_start,
    follow_up_end
from follow_up_bins
order by visit_num