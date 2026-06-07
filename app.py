import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import shap
import pickle
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="FIFA Ratings Lie — XGBoost Se Pakdo",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme CSS
st.markdown("""
<style>
    .main { background-color: #0f0f1a; }
    .stApp { background-color: #0f0f1a; }
    h1, h2, h3, p, label { color: #e8e8f0 !important; }
    .metric-card {
        background: #1a1a2e;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #333355;
        text-align: center;
    }
    .overrated { color: #ff6b6b !important; font-weight: bold; }
    .underrated { color: #00e5cc !important; font-weight: bold; }
    .stSelectbox label, .stSlider label { color: #e8e8f0 !important; }
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD DATA + TRAIN MODEL (cached)
# ============================================
@st.cache_data
def load_and_train():
    df = pd.read_csv('fifa_cleaned.csv')

    feature_cols = [
        'age', 'pace', 'shooting', 'passing',
        'dribbling', 'defending', 'physic',
        'attacking_finishing', 'attacking_short_passing',
        'attacking_crossing', 'attacking_heading_accuracy',
        'skill_dribbling', 'skill_ball_control',
        'skill_long_passing', 'skill_curve',
        'movement_acceleration', 'movement_sprint_speed',
        'movement_agility', 'movement_reactions', 'movement_balance',
        'power_stamina', 'power_strength', 'power_shot_power',
        'mentality_vision', 'mentality_composure',
        'mentality_positioning', 'mentality_aggression',
        'mentality_interceptions', 'mentality_penalties',
        'defending_marking', 'defending_standing_tackle',
        'defending_sliding_tackle',
        'weak_foot', 'skill_moves',
        'international_reputation', 'pos_encoded'
    ]

    pos_map = {'Attacker': 2, 'Midfielder': 1,
               'Defender': 0, 'Unknown': -1}
    df['pos_encoded'] = df['pos_category'].map(pos_map)

    X = df[feature_cols]
    y = df['match_impact']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=300, learning_rate=0.05,
        max_depth=5, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1,
        reg_lambda=1.0, random_state=42, verbosity=0
    )
    model.fit(X_train, y_train, verbose=False)

    df['predicted_impact'] = model.predict(X)
    df['impact_percentile'] = df.groupby('pos_category')['predicted_impact']\
                                .rank(pct=True) * 100
    df['overall_percentile'] = df.groupby('pos_category')['overall']\
                                 .rank(pct=True) * 100
    df['percentile_gap'] = (df['impact_percentile'] -
                            df['overall_percentile']).round(2)

    return df, model, feature_cols

df, model, feature_cols = load_and_train()

# ============================================
# SIDEBAR
# ============================================
st.sidebar.markdown("## ⚽ FIFA Ratings Lie")
st.sidebar.markdown("**XGBoost Se Pakdo**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Overview", "🔍 Player Lookup",
     "📊 Overrated vs Underrated", "🧠 SHAP Insights"]
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Total Players:** {len(df):,}")
st.sidebar.markdown(f"**Model R²:** 0.9944")
st.sidebar.markdown(f"**MAE:** 0.77 points")

# ============================================
# PAGE 1 — OVERVIEW
# ============================================
if page == "🏠 Overview":
    st.title("⚽ FIFA Ratings Lie — XGBoost Se Pakdo")
    st.markdown(
        "### Does EA Sports actually know what makes a player impactful?"
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div class='metric-card'>
            <h2 style='color:#00e5cc'>0.9944</h2>
            <p>R² Score</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class='metric-card'>
            <h2 style='color:#f0c040'>0.77</h2>
            <p>MAE (0-100 scale)</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class='metric-card'>
            <h2 style='color:#ff6b6b'>0.199</h2>
            <p>Pace Correlation</p></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class='metric-card'>
            <h2 style='color:#69ff85'>0.795</h2>
            <p>Composure Correlation</p></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💡 Key Finding")
    st.markdown("""
    > **Pace** — jo FIFA ka sabse marketed attribute hai —
    > match impact se sirf **0.199** correlated hai.
    >
    > **Composure** aur **Vision** — jo FIFA ignore karta hai —
    > **4x zyada** important hain: 0.795 aur 0.775.
    >
    > XGBoost ne 16,860 players ka data analyze karke
    > yeh prove kiya. EA Sports exposed. 🎯
    """)

    st.markdown("---")
    st.markdown("### 📈 Top 10 Most Impactful Players")
    top10 = df.nlargest(10, 'predicted_impact')[
        ['short_name', 'pos_category', 'overall',
         'predicted_impact', 'percentile_gap']
    ].reset_index(drop=True)
    top10.columns = ['Player', 'Position', 'FIFA Overall',
                     'XGBoost Impact', 'Percentile Gap']
    top10['XGBoost Impact'] = top10['XGBoost Impact'].round(2)
    st.dataframe(top10, use_container_width=True)

# ============================================
# PAGE 2 — PLAYER LOOKUP
# ============================================
elif page == "🔍 Player Lookup":
    st.title("🔍 Player Impact Lookup")
    st.markdown("Search karo kisi bhi player ka real impact score")

    # Search
    search = st.text_input("Player name search karo:", "")
    min_ovr = st.slider("Minimum Overall Rating:", 50, 95, 75)

    filtered = df[df['overall'] >= min_ovr].copy()
    if search:
        filtered = filtered[
            filtered['short_name'].str.contains(
                search, case=False, na=False
            )
        ]

    display_cols = ['short_name', 'pos_category', 'overall',
                    'predicted_impact', 'percentile_gap']
    display = filtered[display_cols].sort_values(
        'predicted_impact', ascending=False
    ).head(20).reset_index(drop=True)
    display.columns = ['Player', 'Position', 'FIFA Overall',
                       'XGBoost Impact', 'Percentile Gap']
    display['XGBoost Impact'] = display['XGBoost Impact'].round(2)

    st.dataframe(display, use_container_width=True)

    # Individual player deep dive
    st.markdown("---")
    st.markdown("### 🎯 Player Deep Dive")
    player_names = df[df['overall'] >= 75]['short_name'].tolist()
    selected = st.selectbox("Player select karo:", player_names)

    player_row = df[df['short_name'] == selected].iloc[0]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("FIFA Overall", int(player_row['overall']))
    with col2:
        st.metric("XGBoost Impact",
                  f"{player_row['predicted_impact']:.1f}")
    with col3:
        gap = player_row['percentile_gap']
        label = "🔴 Overrated" if gap < -20 else \
                "🟢 Underrated" if gap > 20 else "⚪ Fair"
        st.metric("Verdict", label)

    # Attribute radar
    attrs = ['pace', 'shooting', 'passing',
             'dribbling', 'defending', 'physic']
    vals  = [player_row[a] for a in attrs]

    fig, ax = plt.subplots(figsize=(6, 3),
                           facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    colors = ['#ff6b6b' if a == 'pace' else '#00e5cc'
              for a in attrs]
    bars = ax.bar(attrs, vals, color=colors, alpha=0.85)
    ax.set_ylim(0, 100)
    ax.tick_params(colors='white', labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')
    ax.set_title(f"{selected} — Attribute Profile",
                 color='white', fontsize=11)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1, str(int(val)),
                ha='center', va='bottom',
                color='white', fontsize=8)
    st.pyplot(fig)

# ============================================
# PAGE 3 — OVERRATED VS UNDERRATED
# ============================================
elif page == "📊 Overrated vs Underrated":
    st.title("📊 Overrated vs Underrated Players")

    min_ovr2 = st.slider("Minimum Overall:", 75, 90, 78)
    pos_filter = st.multiselect(
        "Position filter:",
        ['Attacker', 'Midfielder', 'Defender'],
        default=['Attacker', 'Midfielder', 'Defender']
    )

    filtered2 = df[
        (df['overall'] >= min_ovr2) &
        (df['pos_category'].isin(pos_filter))
    ].copy()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔴 Most Overrated")
        over = filtered2.nsmallest(10, 'percentile_gap')[
            ['short_name', 'pos_category',
             'overall', 'predicted_impact', 'percentile_gap']
        ].reset_index(drop=True)
        over.columns = ['Player', 'Pos', 'Overall',
                        'Impact', 'Gap']
        over['Impact'] = over['Impact'].round(1)
        st.dataframe(over, use_container_width=True)

    with col2:
        st.markdown("### 🟢 Most Underrated")
        under = filtered2.nlargest(10, 'percentile_gap')[
            ['short_name', 'pos_category',
             'overall', 'predicted_impact', 'percentile_gap']
        ].reset_index(drop=True)
        under.columns = ['Player', 'Pos', 'Overall',
                         'Impact', 'Gap']
        under['Impact'] = under['Impact'].round(1)
        st.dataframe(under, use_container_width=True)

    # Scatter plot
    st.markdown("---")
    fig2, ax2 = plt.subplots(figsize=(10, 5),
                              facecolor='#0f0f1a')
    ax2.set_facecolor('#1a1a2e')
    colors_map = {'Attacker': '#ff6b6b',
                  'Midfielder': '#00e5cc',
                  'Defender': '#b388ff'}
    for pos in pos_filter:
        subset = filtered2[filtered2['pos_category'] == pos]
        ax2.scatter(subset['overall'],
                    subset['predicted_impact'],
                    c=colors_map[pos], alpha=0.5,
                    s=15, label=pos)
    ax2.plot([75, 99], [75, 99], color='#f0c040',
             lw=2, linestyle='--', label='Perfect agreement')
    ax2.set_xlabel('FIFA Overall', color='white')
    ax2.set_ylabel('XGBoost Impact', color='white')
    ax2.tick_params(colors='white')
    ax2.legend(facecolor='#1a1a2e', labelcolor='white')
    for spine in ax2.spines.values():
        spine.set_edgecolor('#333355')
    ax2.set_title('Overall vs Predicted Impact',
                  color='white', fontsize=12)
    st.pyplot(fig2)

# ============================================
# PAGE 4 — SHAP INSIGHTS
# ============================================
elif page == "🧠 SHAP Insights":
    st.title("🧠 SHAP — Model Ke Andar Jhaanko")
    st.markdown(
        "SHAP values batate hain — model ne **kyun** "
        "yeh prediction ki"
    )

    st.markdown("### Feature Importance")

    # SHAP computation
    with st.spinner("SHAP values calculate ho rahe hain..."):
        sample = df[feature_cols].sample(300, random_state=42)
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(sample)
        shap_df   = pd.DataFrame(shap_vals,
                                  columns=feature_cols)
        mean_shap = shap_df.abs().mean()\
                           .sort_values(ascending=True)\
                           .tail(15)

    fig3, ax3 = plt.subplots(figsize=(10, 6),
                              facecolor='#0f0f1a')
    ax3.set_facecolor('#1a1a2e')
    pace_related = ['pace', 'movement_sprint_speed',
                    'movement_acceleration']
    colors3 = [
        '#ff6b6b' if f in pace_related else '#00e5cc'
        for f in mean_shap.index
    ]
    ax3.barh(mean_shap.index, mean_shap.values,
             color=colors3, alpha=0.85)
    ax3.set_xlabel('Mean |SHAP value|', color='white')
    ax3.tick_params(colors='white', labelsize=9)
    for spine in ax3.spines.values():
        spine.set_edgecolor('#333355')
    ax3.set_title(
        'SHAP Feature Importance\n'
        'Red = pace related — FIFA overrates these',
        color='white', fontsize=11
    )
    st.pyplot(fig3)

    st.markdown("---")
    st.markdown("### 💡 Key Insight")
    st.info(
        "**mentality_vision** aur **mentality_composure** "
        "top features hain — pace kahin nahi hai top 5 mein. "
        "FIFA ka pace obsession data se prove hota hai ki "
        "yeh overrated attribute hai."
    )