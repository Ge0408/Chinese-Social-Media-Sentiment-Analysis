-- ============================================================================
-- Social Media Sentiment Analysis — Analytical SQL Queries
-- Database: PostgreSQL
-- Table:    social_media_posts
-- Columns:  post_id, date, month, weekday, hour, topic, text, label,
--           sentiment_score, followers, likes, retweets, comments,
--           engagement_total, discovered_topic, discovered_topic_name
-- ============================================================================

-- ============================================================================
-- Query 1: Sentiment Distribution Over Time (Monthly Trends with Window Functions)
-- ----------------------------------------------------------------------------
-- Purpose: Show how average sentiment evolves month-over-month, along with
--          the count of posts and a 3-month moving average to smooth noise.
-- Window functions: AVG() OVER (ORDER BY month ROWS BETWEEN 2 PRECEDING AND
--                   CURRENT ROW) for the moving average.
-- ============================================================================

WITH monthly_stats AS (
    SELECT
        month,
        COUNT(*)                                          AS post_count,
        ROUND(AVG(sentiment_score)::numeric, 4)            AS avg_sentiment,
        ROUND(STDDEV(sentiment_score)::numeric, 4)         AS stddev_sentiment,
        SUM(CASE WHEN label = 'positive' THEN 1 ELSE 0 END) AS positive_count,
        SUM(CASE WHEN label = 'neutral'  THEN 1 ELSE 0 END) AS neutral_count,
        SUM(CASE WHEN label = 'negative' THEN 1 ELSE 0 END) AS negative_count
    FROM social_media_posts
    GROUP BY month
)
SELECT
    month,
    post_count,
    avg_sentiment,
    stddev_sentiment,
    positive_count,
    neutral_count,
    negative_count,
    -- 3-month moving average of avg_sentiment (window frame)
    ROUND(AVG(avg_sentiment) OVER (
        ORDER BY month
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    )::numeric, 4)                                        AS mov_avg_sentiment_3m,
    -- Difference from previous month
    ROUND(
        avg_sentiment - LAG(avg_sentiment) OVER (ORDER BY month),
        4
    )                                                     AS sentiment_mom_change
FROM monthly_stats
ORDER BY month;


-- ============================================================================
-- Query 2: Topic Engagement Ranking
-- ----------------------------------------------------------------------------
-- Purpose: Rank topics by average engagement, compare each topic's average
--          against the overall average (AVG window across all rows), and
--          show how much above/below the global norm each topic sits.
-- ============================================================================

WITH topic_engagement AS (
    SELECT
        discovered_topic_name,
        COUNT(*)                                          AS post_count,
        ROUND(AVG(engagement_total)::numeric, 2)           AS avg_engagement,
        ROUND(AVG(likes)::numeric, 2)                      AS avg_likes,
        ROUND(AVG(retweets)::numeric, 2)                    AS avg_retweets,
        ROUND(AVG(comments)::numeric, 2)                    AS avg_comments,
        ROUND(AVG(sentiment_score)::numeric, 4)             AS avg_sentiment
    FROM social_media_posts
    WHERE discovered_topic_name IS NOT NULL
    GROUP BY discovered_topic_name
),
global_stats AS (
    SELECT
        ROUND(AVG(engagement_total)::numeric, 2) AS global_avg_engagement
    FROM social_media_posts
)
SELECT
    te.discovered_topic_name,
    te.post_count,
    te.avg_engagement,
    te.avg_likes,
    te.avg_retweets,
    te.avg_comments,
    te.avg_sentiment,
    gs.global_avg_engagement,
    -- How far above/below the global average this topic sits
    ROUND(
        (te.avg_engagement - gs.global_avg_engagement) / NULLIF(gs.global_avg_engagement, 0) * 100,
        2
    )                                                     AS pct_above_global_avg,
    -- Rank topics by avg_engagement (ties get same rank)
    RANK() OVER (ORDER BY te.avg_engagement DESC)         AS engagement_rank
FROM topic_engagement te
CROSS JOIN global_stats gs
ORDER BY engagement_rank;


-- ============================================================================
-- Query 3: Peak Posting Hours Analysis (Hourly Distribution by Topic)
-- ----------------------------------------------------------------------------
-- Purpose: Identify which hours of the day produce the most posts per topic,
--          and which topic dominates each hour. Useful for scheduling content.
-- ============================================================================

WITH hourly_topic_counts AS (
    SELECT
        hour,
        discovered_topic_name,
        COUNT(*)                                          AS post_count
    FROM social_media_posts
    WHERE discovered_topic_name IS NOT NULL
    GROUP BY hour, discovered_topic_name
),
hourly_ranked AS (
    SELECT
        hour,
        discovered_topic_name,
        post_count,
        -- Which topic posts most in this hour?
        ROW_NUMBER() OVER (
            PARTITION BY hour
            ORDER BY post_count DESC
        )                                                 AS topic_rank_in_hour,
        -- What fraction of this hour's posts does this topic occupy?
        ROUND(
            100.0 * post_count / SUM(post_count) OVER (PARTITION BY hour),
            2
        )                                                 AS pct_of_hour
    FROM hourly_topic_counts
)
SELECT
    hour,
    discovered_topic_name,
    post_count,
    pct_of_hour,
    CASE
        WHEN topic_rank_in_hour = 1 THEN '★ Peak'
        WHEN topic_rank_in_hour = 2 THEN '☆ Runner-up'
        ELSE ''
    END                                                   AS hour_dominance
FROM hourly_ranked
WHERE topic_rank_in_hour <= 3  -- Top 3 topics per hour
ORDER BY hour, topic_rank_in_hour;


-- ============================================================================
-- Query 4: Sentiment Score Correlation with Engagement Metrics
-- ----------------------------------------------------------------------------
-- Purpose: Investigate whether more extreme sentiment drives higher
--          engagement by bucketing sentiment scores into deciles and
--          computing average engagement per bucket.
-- ============================================================================

WITH sentiment_buckets AS (
    SELECT
        -- Divide sentiment scores into 10 equal-width buckets
        CASE
            WHEN sentiment_score < -0.8 THEN '1  (Very Negative)'
            WHEN sentiment_score < -0.6 THEN '2  (Negative)'
            WHEN sentiment_score < -0.4 THEN '3  (Slightly Negative)'
            WHEN sentiment_score < -0.2 THEN '4  (Mildly Negative)'
            WHEN sentiment_score <  0.0 THEN '5  (Slightly Neg–Neutral)'
            WHEN sentiment_score <  0.2 THEN '6  (Slightly Pos–Neutral)'
            WHEN sentiment_score <  0.4 THEN '7  (Mildly Positive)'
            WHEN sentiment_score <  0.6 THEN '8  (Slightly Positive)'
            WHEN sentiment_score <  0.8 THEN '9  (Positive)'
            ELSE                          '10 (Very Positive)'
        END                                               AS sentiment_bucket,
        sentiment_score,
        engagement_total,
        likes,
        retweets,
        comments,
        followers
    FROM social_media_posts
)
SELECT
    sentiment_bucket,
    COUNT(*)                                              AS post_count,
    ROUND(AVG(engagement_total)::numeric, 2)               AS avg_engagement,
    ROUND(AVG(likes)::numeric, 2)                          AS avg_likes,
    ROUND(AVG(retweets)::numeric, 2)                       AS avg_retweets,
    ROUND(AVG(comments)::numeric, 2)                       AS avg_comments,
    -- Correlation proxy: engagement per follower
    ROUND(
        AVG(CASE WHEN followers > 0
            THEN engagement_total::numeric / followers
            ELSE NULL
        END)::numeric, 6
    )                                                      AS avg_engagement_rate,
    -- Median sentiment within bucket (using percentile_cont)
    ROUND(
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY sentiment_score)::numeric,
        4
    )                                                      AS median_sentiment
FROM sentiment_buckets
GROUP BY sentiment_bucket
ORDER BY sentiment_bucket;


-- ============================================================================
-- Query 5: High-Impact Negative Posts Detection
-- ----------------------------------------------------------------------------
-- Purpose: Find negative-sentiment posts whose engagement lands in the
--          top 10 % (90th percentile or above). These are "viral negativity"
--          posts that may need urgent attention or PR response.
-- ============================================================================

WITH engagement_percentiles AS (
    SELECT
        PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY engagement_total)
          AS p90_engagement
    FROM social_media_posts
)
SELECT
    p.post_id,
    p.date,
    p.hour,
    p.topic,
    p.text,
    p.sentiment_score,
    p.label,
    p.engagement_total,
    p.likes,
    p.retweets,
    p.comments,
    p.followers,
    ep.p90_engagement,
    -- How many times above the 90th percentile threshold?
    ROUND(
        p.engagement_total::numeric / NULLIF(ep.p90_engagement, 0),
        2
    )                                                     AS times_above_p90
FROM social_media_posts p
CROSS JOIN engagement_percentiles ep
WHERE p.label = 'negative'
  AND p.engagement_total >= ep.p90_engagement
ORDER BY p.engagement_total DESC;


-- ============================================================================
-- Query 6: Content Topic Diversity Per User (Simulated)
-- ----------------------------------------------------------------------------
-- Purpose: Simulate per-author topic diversity by using post_id as a
--          proxy for distinct content creators. Measures how many distinct
--          topics each simulated user posts about, and what fraction of
--          their posts their top topic occupies.
-- ============================================================================

-- Note: Since there is no real user/author column, we simulate one by
--       grouping posts into cohorts of 10 based on post_id ordering.
--       Replace the cohort logic with a real author_id column when available.

WITH author_cohort AS (
    SELECT
        -- Simulate an author: every 10 sequential posts = 1 author
        CEIL(post_id / 10.0)::int  AS simulated_author_id,
        discovered_topic_name,
        engagement_total
    FROM social_media_posts
    WHERE discovered_topic_name IS NOT NULL
),
author_stats AS (
    SELECT
        simulated_author_id,
        COUNT(DISTINCT discovered_topic_name)  AS unique_topics,
        COUNT(*)                               AS total_posts,
        -- What's the single topic this author posted about most?
        MODE() WITHIN GROUP (ORDER BY discovered_topic_name)
                                               AS dominant_topic,
        ROUND(AVG(engagement_total)::numeric, 2) AS avg_engagement
    FROM author_cohort
    GROUP BY simulated_author_id
),
author_topic_breakdown AS (
    SELECT
        simulated_author_id,
        discovered_topic_name,
        COUNT(*) AS topic_post_count,
        ROW_NUMBER() OVER (
            PARTITION BY simulated_author_id
            ORDER BY COUNT(*) DESC
        ) AS topic_rank
    FROM author_cohort
    GROUP BY simulated_author_id, discovered_topic_name
)
SELECT
    s.simulated_author_id,
    s.total_posts,
    s.unique_topics,
    s.dominant_topic,
    -- Fraction of the author's posts that are their dominant topic
    ROUND(
        100.0 * t.topic_post_count / NULLIF(s.total_posts, 0),
        2
    )                                                AS dominant_topic_pct,
    s.avg_engagement,
    CASE
        WHEN s.unique_topics >= 5 THEN 'High diversity'
        WHEN s.unique_topics >= 3 THEN 'Medium diversity'
        WHEN s.unique_topics >= 2 THEN 'Low diversity'
        ELSE 'Focused (single topic)'
    END                                              AS diversity_label
FROM author_stats s
JOIN author_topic_breakdown t
    ON  t.simulated_author_id = s.simulated_author_id
    AND t.topic_rank = 1
ORDER BY s.unique_topics DESC, s.total_posts DESC
LIMIT 50;


-- ============================================================================
-- Query 7: Weekday vs Weekend Posting Pattern
-- ----------------------------------------------------------------------------
-- Purpose: Compare posting volume, sentiment, and engagement between
--          weekdays (Mon–Fri) and weekends (Sat–Sun), broken down by
--          topic. Also includes a per-topic breakdown.
-- ============================================================================

WITH is_weekend AS (
    SELECT
        *,
        CASE WHEN weekday IN ('Saturday', 'Sunday')
            THEN 'Weekend'
            ELSE 'Weekday'
        END AS day_type
    FROM social_media_posts
)
SELECT
    day_type,
    discovered_topic_name,
    COUNT(*)                                          AS post_count,
    ROUND(AVG(sentiment_score)::numeric, 4)            AS avg_sentiment,
    ROUND(AVG(engagement_total)::numeric, 2)           AS avg_engagement,
    ROUND(AVG(likes)::numeric, 2)                      AS avg_likes,
    ROUND(AVG(retweets)::numeric, 2)                   AS avg_retweets,
    ROUND(AVG(comments)::numeric, 2)                   AS avg_comments,
    -- Share of the day-type's total posts
    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY day_type),
        2
    )                                                  AS pct_of_daytype
FROM is_weekend
WHERE discovered_topic_name IS NOT NULL
GROUP BY day_type, discovered_topic_name
ORDER BY day_type, post_count DESC;

-- Summary rollup: weekday vs weekend aggregates across all topics
SELECT
    CASE WHEN weekday IN ('Saturday', 'Sunday')
        THEN 'Weekend'
        ELSE 'Weekday'
    END                                               AS day_type,
    COUNT(*)                                          AS total_posts,
    ROUND(AVG(sentiment_score)::numeric, 4)            AS avg_sentiment,
    ROUND(AVG(engagement_total)::numeric, 2)           AS avg_engagement,
    ROUND(AVG(followers)::numeric, 2)                  AS avg_followers
FROM social_media_posts
GROUP BY day_type
ORDER BY day_type;


-- ============================================================================
-- Query 8: Topic Popularity Momentum (Growing / Declining Topics by Month)
-- ----------------------------------------------------------------------------
-- Purpose: Detect which topics are gaining or losing traction month over
--          month using LEAD/LAG window functions. Topics with rising
--          post counts are "growing"; those with falling counts are
--          "declining".
-- ============================================================================

WITH monthly_topic_counts AS (
    SELECT
        month,
        discovered_topic_name,
        COUNT(*)                                       AS post_count
    FROM social_media_posts
    WHERE discovered_topic_name IS NOT NULL
    GROUP BY month, discovered_topic_name
),
monthly_with_lag AS (
    SELECT
        month,
        discovered_topic_name,
        post_count,
        -- Post count from the previous month
        LAG(post_count) OVER (
            PARTITION BY discovered_topic_name
            ORDER BY month
        )                                              AS prev_month_count,
        -- Post count from the next month
        LEAD(post_count) OVER (
            PARTITION BY discovered_topic_name
            ORDER BY month
        )                                              AS next_month_count
    FROM monthly_topic_counts
),
topic_momentum AS (
    SELECT
        month,
        discovered_topic_name,
        post_count,
        prev_month_count,
        -- Month-over-month change (absolute)
        post_count - prev_month_count                  AS mom_change_abs,
        -- Month-over-month change (percentage)
        CASE
            WHEN prev_month_count > 0 THEN
                ROUND(
                    (post_count - prev_month_count)::numeric
                    / prev_month_count * 100,
                    2
                )
            ELSE NULL
        END                                            AS mom_change_pct,
        -- Momentum label
        CASE
            WHEN post_count > prev_month_count
                 AND prev_month_count IS NOT NULL
                THEN 'Growing'
            WHEN post_count < prev_month_count
                 AND prev_month_count IS NOT NULL
                THEN 'Declining'
            WHEN prev_month_count IS NULL THEN 'First appearance'
            ELSE 'Stable'
        END                                            AS momentum
    FROM monthly_with_lag
)
SELECT
    month,
    discovered_topic_name,
    post_count,
    prev_month_count,
    mom_change_abs,
    mom_change_pct,
    momentum,
    -- Rank topics within each month by growth rate
    RANK() OVER (
        PARTITION BY month
        ORDER BY mom_change_pct DESC NULLS LAST
    )                                                  AS growth_rank
FROM topic_momentum
ORDER BY month, growth_rank;

-- Summary: topic momentum across the entire time range
-- Shows which topics are net-growing, net-declining, or stable
WITH first_last AS (
    SELECT
        discovered_topic_name,
        MIN(month) AS first_month,
        MAX(month) AS last_month
    FROM social_media_posts
    WHERE discovered_topic_name IS NOT NULL
    GROUP BY discovered_topic_name
),
first_last_counts AS (
    SELECT
        fl.discovered_topic_name,
        fl.first_month,
        fl.last_month,
        COUNT(*) FILTER (WHERE p.month = fl.first_month)
            AS first_month_count,
        COUNT(*) FILTER (WHERE p.month = fl.last_month)
            AS last_month_count
    FROM first_last fl
    JOIN social_media_posts p
        ON  p.discovered_topic_name = fl.discovered_topic_name
        AND (p.month = fl.first_month OR p.month = fl.last_month)
    GROUP BY fl.discovered_topic_name, fl.first_month, fl.last_month
)
SELECT
    discovered_topic_name,
    first_month,
    last_month,
    first_month_count,
    last_month_count,
    last_month_count - first_month_count AS net_change,
    ROUND(
        (last_month_count - first_month_count)::numeric
        / NULLIF(first_month_count, 0) * 100,
        2
    )                                   AS pct_change,
    CASE
        WHEN last_month_count > first_month_count THEN 'Growing  ↑'
        WHEN last_month_count < first_month_count THEN 'Declining ↓'
        ELSE 'Stable   →'
    END                                 AS overall_momentum
FROM first_last_counts
ORDER BY pct_change DESC NULLS LAST;
