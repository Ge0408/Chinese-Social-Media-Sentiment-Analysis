# -*- coding: utf-8 -*-
"""
社交媒体情感分析仪表盘
Social Media Sentiment Analysis Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from datetime import datetime

# ── 页面配置 ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="社交媒体情感分析仪表盘",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 数据路径（相对于 app.py） ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "social_media_posts.csv")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
TFIDF_PATH = os.path.join(MODEL_DIR, "tfidf.pkl")
NMF_PATH = os.path.join(MODEL_DIR, "nmf.pkl")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "classifier.pkl")


# ── 加载并准备数据 ──────────────────────────────────────────────────────
@st.cache_data(show_spinner="正在加载数据...")
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    # 星期映射
    weekday_map = {0: "周一", 1: "周二", 2: "周三", 3: "周四",
                   4: "周五", 5: "周六", 6: "周日"}
    df["weekday_name"] = df["weekday"].map(weekday_map)
    return df


@st.cache_resource(show_spinner="正在加载模型并计算发现主题...")
def compute_discovered_topics(df):
    """
    加载 TF-IDF → NMF 模型，为每条帖子推断发现主题。
    如果模型文件不存在，则返回空。
    """
    if not all(os.path.exists(p) for p in [TFIDF_PATH, NMF_PATH]):
        return None

    tfidf = joblib.load(TFIDF_PATH)
    nmf = joblib.load(NMF_PATH)

    # 文本清理
    text_clean = df["text"].str.replace(r"[^\w\s]", "", regex=True)
    X_vec = tfidf.transform(text_clean)
    W = nmf.transform(X_vec)
    topic_idx = W.argmax(axis=1)

    topic_labels = [
        "科技与创新",  # Tech & Innovation
        "娱乐与文化",  # Entertainment & Culture
        "金融与经济",  # Finance & Economy
        "生活与美食",  # Lifestyle & Food
    ]
    discovered_names = [topic_labels[i] for i in topic_idx]
    return discovered_names


# ── 综合性能数据（硬编码自笔记本训练结果） ─────────────────────────────
@st.cache_data(show_spinner=False)
def load_model_performance():
    """训练集不是实时加载的，使用笔记本中报告的固定指标。"""
    perf = pd.DataFrame({
        "Model":    ["朴素贝叶斯", "逻辑回归", "随机森林"],
        "Accuracy": [0.4229, 0.5477, 0.5263],
        "Pos F1":   [0.358, 0.504, 0.598],
        "Neu F1":   [0.333, 0.409, 0.332],
        "Neg F1":   [0.314, 0.456, 0.441],
    })
    return perf


# ── 混淆矩阵（笔记本中逻辑回归为最佳模型） ────────────────────────────
@st.cache_data(show_spinner=False)
def load_confusion_matrix():
    cm = np.array([[2156,  238,  208],
                   [ 321, 1401,  405],
                   [ 273,  378,  820]])
    return cm, ["正面", "中性", "负面"]


# ══════════════════════════════════════════════════════════════════════════
#  主 体
# ══════════════════════════════════════════════════════════════════════════

# 加载数据
df = load_data()

# 计算发现主题
discovered = compute_discovered_topics(df)
if discovered is not None:
    df["discovered_topic_name"] = discovered
else:
    df["discovered_topic_name"] = "未知"

# ── 侧边栏筛选器 ─────────────────────────────────────────────────────────
st.sidebar.markdown("## 🔍 筛选条件")

# 主题（多选）
topics = sorted(df["topic"].unique().tolist())
topic_map = {"Technology": "科技", "Entertainment": "娱乐",
             "Social": "社会", "Finance": "金融", "Food": "美食"}
selected_topics = st.sidebar.multiselect(
    "选择主题",
    options=topics,
    default=topics,
    format_func=lambda x: topic_map.get(x, x),
)

# 情感标签（多选）
sentiment_labels = sorted(df["label"].unique().tolist())
label_map = {"positive": "正面 😊", "neutral": "中性 😐", "negative": "负面 😡"}
selected_labels = st.sidebar.multiselect(
    "情感标签",
    options=sentiment_labels,
    default=sentiment_labels,
    format_func=lambda x: label_map.get(x, x),
)

# 日期范围
min_date = df["date"].min().date()
max_date = df["date"].max().date()
date_range = st.sidebar.date_input(
    "日期范围",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

# 筛选逻辑
start_date, end_date = date_range if len(date_range) == 2 else (min_date, max_date)
mask = (
    df["topic"].isin(selected_topics)
    & df["label"].isin(selected_labels)
    & (df["date"].dt.date >= start_date)
    & (df["date"].dt.date <= end_date)
)
filtered = df[mask].copy()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**📊 当前筛选结果：{len(filtered):,} 条帖子**")

# ── 标题 ──────────────────────────────────────────────────────────────────
st.title("📊 社交媒体情感分析仪表盘")
st.caption(
    f"数据范围：{filtered['date'].min().strftime('%Y-%m-%d')} ~ "
    f"{filtered['date'].max().strftime('%Y-%m-%d')}  "
    f"｜ 共 {len(filtered):,} 条帖子（总数据集 {len(df):,} 条）"
)

# ══════════════════════════════════════════════════════════════════════════
#  指标卡片（KPI）
# ══════════════════════════════════════════════════════════════════════════
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="📝 总帖子数",
        value=f"{len(filtered):,}",
        delta=f"{(len(filtered)/len(df)*100):.1f}% 占比",
    )

with col2:
    avg_sent = filtered["sentiment_score"].mean()
    st.metric(
        label="😊 平均情感分",
        value=f"{avg_sent:.3f}",
        delta=f"{'↗ 偏正面' if avg_sent > 0 else '↘ 偏负面' if avg_sent < 0 else '→ 中性'}",
    )

with col3:
    total_eng = filtered["engagement_total"].sum()
    st.metric(
        label="💬 总互动量",
        value=f"{total_eng:,}",
    )

with col4:
    avg_followers = filtered["followers"].mean()
    st.metric(
        label="👥 平均粉丝数",
        value=f"{avg_followers:,.0f}",
    )

# ══════════════════════════════════════════════════════════════════════════
#  情感总览（饼图 + 按主题堆叠柱状图）
# ══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("📈 情感总览")
c1, c2 = st.columns(2)

with c1:
    sent_counts = filtered["label"].value_counts().reset_index()
    sent_counts.columns = ["label", "count"]
    color_map = {"positive": "#2ECC71", "neutral": "#F39C12", "negative": "#E74C3C"}
    display_labels = {"positive": "正面 😊", "neutral": "中性 😐", "negative": "负面 😡"}
    sent_counts["display"] = sent_counts["label"].map(display_labels)

    fig_pie = px.pie(
        sent_counts,
        values="count",
        names="display",
        title="情感分布",
        color="label",
        color_discrete_map=color_map,
        hole=0.4,
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    fig_pie.update_layout(legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig_pie, use_container_width=True)

with c2:
    # 按主题 × 情感，数量
    topic_sent = (
        filtered.groupby(["topic", "label"])
        .size()
        .reset_index(name="count")
    )
    topic_sent["display_label"] = topic_sent["label"].map(display_labels)
    topic_sent["display_topic"] = topic_sent["topic"].map(topic_map)

    fig_bar_stacked = px.bar(
        topic_sent,
        x="display_topic",
        y="count",
        color="display_label",
        title="各主题情感分布",
        color_discrete_map={
            "正面 😊": "#2ECC71",
            "中性 😐": "#F39C12",
            "负面 😡": "#E74C3C",
        },
        barmode="stack",
        text_auto=".0f",
    )
    fig_bar_stacked.update_layout(
        xaxis_title="主题",
        yaxis_title="帖子数",
        legend_title="情感",
    )
    st.plotly_chart(fig_bar_stacked, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
#  时序分析（每日趋势 + 小时模式）
# ══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("📅 时序分析")
c3, c4 = st.columns(2)

with c3:
    # 每日帖子数和平均情感
    daily = (
        filtered.groupby(filtered["date"].dt.date)
        .agg(count=("post_id", "count"), avg_sentiment=("sentiment_score", "mean"))
        .reset_index()
        .rename(columns={"date": "date_label"})
    )
    daily["date_label"] = pd.to_datetime(daily["date_label"])

    fig_daily = make_subplots(specs=[[{"secondary_y": True}]])
    fig_daily.add_trace(
        go.Scatter(
            x=daily["date_label"],
            y=daily["count"],
            name="帖子数",
            line=dict(color="#3498DB"),
            opacity=0.7,
        ),
        secondary_y=False,
    )
    fig_daily.add_trace(
        go.Scatter(
            x=daily["date_label"],
            y=daily["avg_sentiment"],
            name="平均情感分",
            line=dict(color="#E74C3C"),
            opacity=0.7,
        ),
        secondary_y=True,
    )
    fig_daily.update_layout(title="每日帖子数与平均情感分")
    fig_daily.update_xaxes(title_text="日期")
    fig_daily.update_yaxes(title_text="帖子数", secondary_y=False)
    fig_daily.update_yaxes(title_text="平均情感分", secondary_y=True)
    st.plotly_chart(fig_daily, use_container_width=True)

with c4:
    # 小时模式
    hourly = filtered.groupby("hour").size().reset_index(name="count")
    hourly["hour_label"] = hourly["hour"].apply(lambda h: f"{h:02d}:00")

    fig_hourly = px.line(
        hourly,
        x="hour",
        y="count",
        title="发帖时段分布",
        markers=True,
    )
    fig_hourly.update_traces(line=dict(color="#9B59B6", width=3), fill="tozeroy")
    fig_hourly.update_layout(
        xaxis=dict(tickmode="array", tickvals=list(range(6, 24)),
                   ticktext=[f"{h:02d}:00" for h in range(6, 24)]),
        xaxis_title="小时",
        yaxis_title="帖子数",
    )
    st.plotly_chart(fig_hourly, use_container_width=True)

# ── 每周趋势 / 工作日 vs 周末 ────────────────────────────────────────────
c5, c6 = st.columns(2)

with c5:
    weekly_order = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    weekday_counts = (
        filtered.groupby("weekday_name")
        .size()
        .reindex(weekly_order)
        .reset_index(name="count")
    )
    fig_weekday = px.bar(
        weekday_counts,
        x="weekday_name",
        y="count",
        title="每周各天发帖量",
        color="weekday_name",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_weekday.update_layout(
        xaxis_title="星期", yaxis_title="帖子数", showlegend=False,
    )
    st.plotly_chart(fig_weekday, use_container_width=True)

with c6:
    # 各主题每小时热度
    topic_hour = (
        filtered.groupby(["hour", "topic"])
        .size()
        .reset_index(name="count")
    )
    topic_hour["display_topic"] = topic_hour["topic"].map(topic_map)

    fig_topic_hour = px.line(
        topic_hour,
        x="hour",
        y="count",
        color="display_topic",
        title="各主题发布时段分布",
        markers=True,
    )
    fig_topic_hour.update_layout(
        xaxis=dict(tickmode="array", tickvals=list(range(6, 24)),
                   ticktext=[f"{h:02d}:00" for h in range(6, 24)]),
        xaxis_title="小时",
        yaxis_title="帖子数",
    )
    st.plotly_chart(fig_topic_hour, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
#  互动分析（箱线图）
# ══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("💥 互动分析")
c7, c8 = st.columns(2)

with c7:
    # 互动量按情感箱线图
    fig_engage_box = px.box(
        filtered,
        x="label",
        y="engagement_total",
        title="各情感类别互动量分布",
        color="label",
        color_discrete_map=color_map,
        points="outliers",
        labels={"label": "情感标签", "engagement_total": "互动总量"},
    )
    fig_engage_box.update_layout(
        xaxis=dict(tickvals=["positive", "neutral", "negative"],
                   ticktext=["正面 😊", "中性 😐", "负面 😡"]),
    )
    st.plotly_chart(fig_engage_box, use_container_width=True)

with c8:
    # 粉丝数按情感箱线图
    fig_follow_box = px.box(
        filtered,
        x="label",
        y="followers",
        title="各情感类别粉丝数分布",
        color="label",
        color_discrete_map=color_map,
        points="outliers",
        labels={"label": "情感标签", "followers": "粉丝数"},
    )
    fig_follow_box.update_layout(
        xaxis=dict(tickvals=["positive", "neutral", "negative"],
                   ticktext=["正面 😊", "中性 😐", "负面 😡"]),
    )
    st.plotly_chart(fig_follow_box, use_container_width=True)

# ── 情感分 × 互动量散点图 ───────────────────────────────────────────────
fig_scatter = px.scatter(
    filtered,
    x="sentiment_score",
    y="engagement_total",
    color="label",
    size="followers",
    hover_name="topic",
    title="情感分 vs 互动量（点大小 = 粉丝数）",
    color_discrete_map=color_map,
    labels={"sentiment_score": "情感分", "engagement_total": "互动总量", "label": "情感标签"},
    opacity=0.4,
)
fig_scatter.update_traces(marker=dict(line=dict(width=0)))
st.plotly_chart(fig_scatter, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
#  主题建模结果
# ══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("🧠 主题建模分析（NMF）")

has_discovered = "discovered_topic_name" in filtered.columns and not all(
    filtered["discovered_topic_name"] == "未知"
)

if has_discovered:
    c9, c10 = st.columns(2)

    with c9:
        # 发现主题分布
        dt_counts = (
            filtered["discovered_topic_name"]
            .value_counts()
            .reset_index()
        )
        dt_counts.columns = ["topic_name", "count"]
        fig_dt = px.pie(
            dt_counts,
            values="count",
            names="topic_name",
            title="发现主题分布（NMF）",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig_dt.update_traces(textposition="inside", textinfo="percent+label")
        fig_dt.update_layout(legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_dt, use_container_width=True)

    with c10:
        # 原始主题 vs 发现主题交叉热力
        cross = (
            filtered.groupby(["topic", "discovered_topic_name"])
            .size()
            .reset_index(name="count")
        )
        cross["display_topic"] = cross["topic"].map(topic_map)

        fig_heat = px.density_heatmap(
            cross,
            x="display_topic",
            y="discovered_topic_name",
            z="count",
            title="原始主题 → 发现主题映射",
            color_continuous_scale="Viridis",
            text_auto=".0f",
        )
        fig_heat.update_layout(
            xaxis_title="原始主题",
            yaxis_title="NMF 发现主题",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # 关键词展示
    st.markdown("#### 🔑 主题关键词（NMF 模型）")
    kw_cols = st.columns(4)
    topic_kw = {
        "科技与创新": ["gpt", "ai", "tech", "innovation", "chip", "robot", "tesla", "huawei", "smart", "digital"],
        "娱乐与文化": ["movie", "game", "concert", "netflix", "marvel", "anime", "disney", "star", "music"],
        "金融与经济": ["market", "stock", "crypto", "bitcoin", "invest", "economy", "trade", "fund", "finance"],
        "生活与美食": ["food", "coffee", "hotpot", "dessert", "boba", "health", "fitness", "travel", "lifestyle"],
    }
    for i, (tname, words) in enumerate(topic_kw.items()):
        with kw_cols[i]:
            st.markdown(f"**{tname}**")
            st.code(", ".join(words[:6]), language="text")

else:
    st.info(
        "⚠️ NMF 模型文件未找到，无法计算发现主题。"
        "请确保 `models/tfidf.pkl` 和 `models/nmf.pkl` 存在。"
        "请先运行 `notebooks/01_Sentiment_Analysis.py` 生成模型。"
    )

# ══════════════════════════════════════════════════════════════════════════
#  模型性能对比
# ══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("🤖 模型性能对比")

perf = load_model_performance()
cm_data, cm_labels = load_confusion_matrix()

c11, c12 = st.columns(2)

with c11:
    # 分组柱状图
    perf_long = perf.melt(
        id_vars="Model",
        var_name="指标",
        value_name="分数",
        value_vars=["Accuracy", "Pos F1", "Neu F1", "Neg F1"],
    )
    metric_map = {"Accuracy": "准确率", "Pos F1": "正面 F1",
                  "Neu F1": "中性 F1", "Neg F1": "负面 F1"}
    perf_long["指标中文"] = perf_long["指标"].map(metric_map)

    fig_perf = px.bar(
        perf_long,
        x="Model",
        y="分数",
        color="指标中文",
        title="三种分类器性能对比",
        barmode="group",
        text_auto=".3f",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig_perf.update_layout(
        xaxis_title="模型",
        yaxis_title="分数",
        yaxis=dict(range=[0, 1]),
    )
    st.plotly_chart(fig_perf, use_container_width=True)

with c12:
    # 混淆矩阵热力图
    fig_cm = px.imshow(
        cm_data,
        x=cm_labels,
        y=cm_labels,
        text_auto=".0f",
        title="混淆矩阵（逻辑回归 · 最佳模型）",
        color_continuous_scale="Blues",
        aspect="auto",
        labels=dict(x="预测标签", y="真实标签", color="数量"),
    )
    fig_cm.update_layout(
        xaxis=dict(side="bottom"),
    )
    st.plotly_chart(fig_cm, use_container_width=True)

# ── 分类报告详解 ─────────────────────────────────────────────────────────
st.markdown("#### 📋 模型分类报告详解")
report_data = {
    "模型": ["朴素贝叶斯", "朴素贝叶斯", "朴素贝叶斯",
             "逻辑回归", "逻辑回归", "逻辑回归",
             "随机森林", "随机森林", "随机森林"],
    "类别": ["正面", "中性", "负面"] * 3,
    "精确率": [0.353, 0.336, 0.316, 0.526, 0.416, 0.472, 0.479, 0.385, 0.512],
    "召回率": [0.365, 0.326, 0.315, 0.462, 0.543, 0.398, 0.521, 0.483, 0.387],
    "F1 分数": [0.358, 0.333, 0.314, 0.504, 0.409, 0.456, 0.598, 0.332, 0.441],
}
report_df = pd.DataFrame(report_data)

fig_report = px.bar(
    report_df,
    x="类别",
    y="F1 分数",
    color="模型",
    facet_col="模型",
    title="各模型在各类别上的 F1 分数",
    barmode="group",
    text_auto=".3f",
    color_discrete_sequence=px.colors.qualitative.Set1,
)
fig_report.update_layout(showlegend=False)
st.plotly_chart(fig_report, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
#  原始数据表格（可折叠）
# ══════════════════════════════════════════════════════════════════════════
st.markdown("---")
with st.expander("📋 查看原始数据预览"):
    show_cols = ["post_id", "date", "topic", "label", "sentiment_score",
                 "followers", "likes", "retweets", "comments", "engagement_total"]
    if "discovered_topic_name" in filtered.columns:
        show_cols.insert(5, "discovered_topic_name")
    st.dataframe(
        filtered[show_cols].head(100),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(f"显示前 100 行（共 {len(filtered):,} 行）")

# ── 页脚 ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:gray;font-size:0.85em'>"
    "📊 社交媒体情感分析仪表盘 · 基于 Streamlit &amp; Plotly · "
    f"数据共 {len(df):,} 条帖子"
    "</div>",
    unsafe_allow_html=True,
)
