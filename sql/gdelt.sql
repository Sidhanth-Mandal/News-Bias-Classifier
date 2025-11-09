WITH filtered_articles AS (
  SELECT 
    DocumentIdentifier as url,
    -- Extract title from V2Themes field (first theme often contains title info)
    SPLIT(V2Themes, ';')[SAFE_OFFSET(0)] as title,
    -- Extract domain from URL as source
    REGEXP_EXTRACT(DocumentIdentifier, r'https?://(?:www\.)?([^/]+)') as source,
    -- Convert DATE (YYYYMMDDHHMMSS) to proper DATE
    DATE(
      CAST(SUBSTR(CAST(DATE AS STRING), 1, 4) AS INT64),
      CAST(SUBSTR(CAST(DATE AS STRING), 5, 2) AS INT64), 
      CAST(SUBSTR(CAST(DATE AS STRING), 7, 2) AS INT64)
    ) as publish_date,
    V2Themes as themes,
    -- Add row number for sampling
    ROW_NUMBER() OVER (
      PARTITION BY REGEXP_EXTRACT(DocumentIdentifier, r'https?://(?:www\.)?([^/]+)')
      ORDER BY RAND()
    ) as rn
  FROM 
    `gdelt-bq.gdeltv2.gkg`
  WHERE 
    -- Filter by date range (2022-01-01 to 2024-08-01)
    DATE >= 20220101000000
    AND DATE <= 20240801235959
    
    -- Filter by Indian news outlets
    AND (
      DocumentIdentifier LIKE '%ndtv.com%' OR
      DocumentIdentifier LIKE '%timesofindia.indiatimes.com%' OR
      DocumentIdentifier LIKE '%thehindu.com%' OR
      DocumentIdentifier LIKE '%indianexpress.com%' OR
      DocumentIdentifier LIKE '%hindustantimes.com%' OR
      DocumentIdentifier LIKE '%republicworld.com%' OR
      DocumentIdentifier LIKE '%indiatoday.in%' OR
      DocumentIdentifier LIKE '%scroll.in%' OR
      DocumentIdentifier LIKE '%deccanherald.com%'
    )
    
    -- Filter by political/economic themes
    AND (
      V2Themes LIKE '%POLITICS%' OR
      V2Themes LIKE '%ELECTION%' OR
      V2Themes LIKE '%GOVERNMENT%' OR
      V2Themes LIKE '%PROTEST%' OR
      V2Themes LIKE '%ECONOMY%'
    )
    
    -- Ensure we have valid URLs and themes
    AND DocumentIdentifier IS NOT NULL
    AND V2Themes IS NOT NULL
)

SELECT 
  url,
  title,
  source,
  publish_date,
  themes
FROM 
  filtered_articles
WHERE 
  -- Limit to 2000 articles per source (randomly sampled)
  rn <= 2000
  
  -- Additional cleanup - ensure we have valid dates
  AND publish_date IS NOT NULL
  
ORDER BY 
  source,
  publish_date DESC;