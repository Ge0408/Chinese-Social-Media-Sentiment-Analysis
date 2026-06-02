# 社交舆情情感分析

> NLP + 数据分析作品集项目 | Social Media Sentiment Analytics

分析流程覆盖：数据生成 → 文本预处理 → 情感分类建模 → 主题建模 → 时间趋势分析 → 交互仪表盘

---

## 项目结构

```
Chinese-Social-Media-Sentiment-Analysis/
├── data/
│   └── social_media_posts.csv           # 20,000 条社交媒体帖子
├── notebooks/
│   └── 01_Sentiment_Analysis.py         # 完整分析 Pipeline
├── dashboard/
│   └── app.py                           # Streamlit 交互仪表盘
├── sql/
│   └── sentiment_queries.sql            # 8 个 PostgreSQL 分析查询
├── models/
│   ├── tfidf.pkl                        # TF-IDF 向量化器
│   ├── nmf.pkl                          # NMF 主题模型
│   └── classifier.pkl                   # 情感分类模型
├── reports/
│   ├── 01_sentiment_eda.png
│   └── 02_model_performance.png
└── README.md
```

## 数据集

20,000 条模拟社交媒体帖子数据，覆盖 5 个话题领域

| 字段 | 说明 | 类型 |
|------|------|------|
| post_id | 帖子 ID | int |
| date | 发布日期 | date |
| topic | 话题领域 | Technology / Entertainment / Social / Finance / Food |
| text | 帖子正文 | string |
| label | 情感标签 | positive / neutral / negative |
| sentiment_score | 情感得分 (-1 ~ +1) | float |
| likes / retweets / comments | 互动指标 | int |
| engagement_total | 总互动量 | int |

## 分析内容

### 情感分布
整体情感倾向分析，各话题领域情感对比

### 时间趋势
日发帖量趋势、小时活跃度模式、话题月度热度变化

### 文本分类

| 模型 | Accuracy | 说明 |
|------|----------|------|
| Naive Bayes | 1.000 | 基准模型 |
| **Logistic Regression** | **1.000** | 最优模型 |
| Random Forest | 1.000 | 集成模型 |

### 主题建模
NMF 无监督主题发现，提取 4 个潜在话题

### 互动分析
不同情感类型帖子的互动量对比，高互动内容特征挖掘

## 技术栈

- **数据处理**: Pandas, NumPy
- **NLP**: Scikit-learn (TF-IDF, NMF)
- **可视化**: Matplotlib, Seaborn, Plotly
- **机器学习**: MultinomialNB, LogisticRegression, RandomForest
- **交互仪表盘**: Streamlit
- **数据库**: PostgreSQL

## 运行

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit plotly joblib wordcloud
python notebooks/01_Sentiment_Analysis.py
streamlit run dashboard/app.py
```

## License

MIT
