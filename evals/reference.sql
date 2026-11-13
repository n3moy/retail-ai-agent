-- Independent reference for the submitted manager_a grants and fixed UTC periods.
-- The public theLook dataset is refreshed. Compute the oracle once per live run.
SELECT
  COUNT(DISTINCT IF(EXTRACT(YEAR FROM i.created_at) = 2024, i.order_id, NULL)) AS jan_orders,
  ROUND(SUM(IF(EXTRACT(YEAR FROM i.created_at) = 2024, i.sale_price, NULL)), 2) AS jan_revenue,
  COUNTIF(EXTRACT(YEAR FROM i.created_at) = 2099) AS future_items
FROM `bigquery-public-data.thelook_ecommerce.order_items` AS i
JOIN `bigquery-public-data.thelook_ecommerce.products` AS p ON p.id = i.product_id
WHERE p.brand = 'Calvin Klein' AND i.status = 'Complete'
  AND ((i.created_at >= '2024-01-01' AND i.created_at < '2024-02-01')
    OR (i.created_at >= '2099-01-01' AND i.created_at < '2099-02-01'))
