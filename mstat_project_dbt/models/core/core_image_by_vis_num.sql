-- SELECT
--     bkts.visit_num AS bkt_num,
--     cis.*,
--     CASE
--         WHEN bkts.visit_num IS NULL THEN TRUE
--         ELSE FALSE
--     END AS is_visit_censored,
--     cis.is_censored AS is_right_censored

-- FROM {{ ref('core_follow_up_buckets') }} AS bkts
-- FULL OUTER JOIN {{ ref('core_image_set') }} AS cis
--     ON cis.visit_num = bkts.visit_num

SELECT 
    bkts.visit_num,
    cis.ptid,
    cis.visit_num AS img_visit_num,
    cis.image_date
FROM {{ ref('core_follow_up_buckets') }} AS bkts
LEFT JOIN {{ ref('core_image_set') }} AS cis
    ON cis.visit_num = bkts.visit_num