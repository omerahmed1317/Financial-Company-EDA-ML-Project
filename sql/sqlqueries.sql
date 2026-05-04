-- Show me  Top 10 companies by profit
SELECT name, profit
FROM financial
ORDER BY profit DESC
LIMIT 10;

-- Show Country with highest sales
SELECT country, SUM(sales) AS total_sales
FROM financial
GROUP BY country
ORDER BY total_sales DESC;

-- Show Average profit per country
SELECT country, AVG(profit) AS avg_profit
FROM financial
GROUP BY country;

-- Show Companies with negative profit
SELECT name, profit
FROM financial
WHERE profit < 0;

-- Top 5 companies by market value
SELECT name, market_value
FROM financial
ORDER BY market_value DESC
LIMIT 5;
-- ✅ 6. Profit margin > 20%
SELECT name, profit_margin
FROM financial
WHERE profit_margin > 0.2;
-- ✅ 7. Country with most companies
SELECT country, COUNT(*) AS company_count
FROM financial
GROUP BY country
ORDER BY company_count DESC;
-- ✅ 8. Companies with high sales but low profit
SELECT name, sales, profit
FROM financial
WHERE sales > 100 AND profit < 10;
-- ✅ 9. Rank companies by profit
SELECT name, profit,
RANK() OVER (ORDER BY profit DESC) AS rank_position
FROM financial;
-- ✅ 10. Top company per country
SELECT country, name, profit
FROM (
    SELECT country, name, profit,
    RANK() OVER (PARTITION BY country ORDER BY profit DESC) AS rnk
    FROM financial
) t
WHERE rnk = 1;
-- ✅ 11. Total market value of all companies
SELECT SUM(market_value) AS total_market_value
FROM financial;
-- ✅ 12. Industry-wise average profit (if column exists)
SELECT industry, AVG(profit)
FROM financial
GROUP BY industry;
-- ✅ 13. Companies with highest asset turnover
SELECT name, asset_turnover
FROM financial
ORDER BY asset_turnover DESC
LIMIT 10;
-- ✅ 14. Detect inefficient companies
SELECT name, assets, profit
FROM financial
WHERE assets > 1000 AND profit < 10;
--  ✅ 15. Percentage contribution by country
SELECT country,
SUM(sales) / (SELECT SUM(sales) FROM financial) * 100 AS contribution
FROM financial
GROUP BY country;