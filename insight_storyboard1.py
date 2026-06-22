import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import spearmanr, pearsonr, shapiro, normaltest
import re

# =============================================================================
# Настройка страницы
# =============================================================================
st.set_page_config(
    page_title="📊 Сториборд: эффективность преподавателей",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Кастомный CSS — градиентный фон + глассморфизм
# =============================================================================
st.markdown("""
<style>
/* Градиентный фон для всей страницы */
.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    background-attachment: fixed;
}

.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    max-width: 1400px;
}

/* Глассморфизм для Hero-секции */
.hero-section {
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: 24px;
    padding: 48px 40px;
    color: white;
    margin-bottom: 32px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.3);
    position: relative;
    overflow: hidden;
}

.hero-section::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 600px;
    height: 600px;
    background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
    border-radius: 50%;
}

.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    margin: 0 0 12px 0;
    letter-spacing: -0.02em;
    position: relative;
    z-index: 1;
    text-shadow: 0 2px 10px rgba(0,0,0,0.2);
}

.hero-subtitle {
    font-size: 1.15rem;
    opacity: 0.95;
    margin: 0;
    line-height: 1.6;
    max-width: 800px;
    position: relative;
    z-index: 1;
}

/* KPI-карточки с глассморфизмом */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 20px;
    margin: 32px 0;
}

.kpi-card {
    background: rgba(255, 255, 255, 0.2);
    backdrop-filter: blur(15px);
    -webkit-backdrop-filter: blur(15px);
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.3);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.kpi-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, rgba(255,255,255,0.8), rgba(255,255,255,0.4));
}

.kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
    background: rgba(255, 255, 255, 0.25);
}

.kpi-icon {
    font-size: 2rem;
    margin-bottom: 8px;
}

.kpi-value {
    font-size: 2.2rem;
    font-weight: 800;
    color: white;
    line-height: 1;
    margin: 8px 0;
    text-shadow: 0 2px 8px rgba(0,0,0,0.2);
}

.kpi-label {
    font-size: 0.9rem;
    color: rgba(255, 255, 255, 0.9);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Секции */
.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 48px 0 24px 0;
    padding: 16px 24px;
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(15px);
    -webkit-backdrop-filter: blur(15px);
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.3);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
}

.section-icon {
    font-size: 1.8rem;
    filter: drop-shadow(0 2px 4px rgba(0,0,0,0.2));
}

.section-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: white;
    margin: 0;
    text-shadow: 0 2px 8px rgba(0,0,0,0.2);
}

.section-count {
    background: rgba(255, 255, 255, 0.3);
    color: white;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-left: auto;
    backdrop-filter: blur(10px);
}

/* Карточки инсайтов с глассморфизмом */
.insight-card {
    background: rgba(255, 255, 255, 0.2);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: 20px;
    padding: 28px;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.3);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.insight-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 6px;
    height: 100%;
    background: linear-gradient(180deg, rgba(255,255,255,0.9), rgba(255,255,255,0.5));
}

.insight-card.positive::before {
    background: linear-gradient(180deg, #10b981, #059669);
}

.insight-card.negative::before {
    background: linear-gradient(180deg, #ef4444, #dc2626);
}

.insight-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 16px 48px rgba(0, 0, 0, 0.15);
    background: rgba(255, 255, 255, 0.25);
}

.insight-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    margin-bottom: 16px;
    flex-wrap: wrap;
    gap: 12px;
}

.insight-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: white;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 10px;
    text-shadow: 0 2px 8px rgba(0,0,0,0.2);
}

.insight-badges {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}

.badge {
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 4px;
    white-space: nowrap;
    backdrop-filter: blur(10px);
}

.badge-strength-weak {
    background: rgba(254, 243, 199, 0.9);
    color: #92400e;
}

.badge-strength-moderate {
    background: rgba(219, 234, 254, 0.9);
    color: #1e40af;
}

.badge-strength-strong {
    background: rgba(252, 231, 243, 0.9);
    color: #9f1239;
}

.badge-method {
    background: rgba(241, 245, 249, 0.9);
    color: #475569;
}

.badge-direction-pos {
    background: rgba(209, 250, 229, 0.9);
    color: #065f46;
}

.badge-direction-neg {
    background: rgba(254, 226, 226, 0.9);
    color: #991b1b;
}

.insight-description {
    color: rgba(255, 255, 255, 0.95);
    font-size: 0.98rem;
    line-height: 1.6;
    margin: 12px 0 20px 0;
    padding: 14px 18px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    border-left: 3px solid rgba(255, 255, 255, 0.5);
    backdrop-filter: blur(10px);
}

.insight-stats {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 16px;
    padding-top: 16px;
    border-top: 1px solid rgba(255, 255, 255, 0.3);
}

.stat-pill {
    background: rgba(255, 255, 255, 0.15);
    padding: 8px 14px;
    border-radius: 12px;
    font-size: 0.85rem;
    color: white;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    border: 1px solid rgba(255, 255, 255, 0.3);
    backdrop-filter: blur(10px);
}

.stat-pill b {
    color: white;
}

/* Выводы с глассморфизмом */
.conclusion-card {
    background: rgba(30, 41, 59, 0.4);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: 24px;
    padding: 36px;
    color: white;
    margin: 40px 0;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.conclusion-title {
    font-size: 1.8rem;
    font-weight: 800;
    margin: 0 0 20px 0;
    display: flex;
    align-items: center;
    gap: 12px;
    text-shadow: 0 2px 10px rgba(0,0,0,0.3);
}

.conclusion-item {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 12px;
    border-left: 4px solid rgba(102, 126, 234, 0.8);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.conclusion-item-title {
    font-weight: 700;
    font-size: 1.05rem;
    margin-bottom: 6px;
    color: #e0e7ff;
    text-shadow: 0 1px 4px rgba(0,0,0,0.2);
}

.conclusion-item-text {
    font-size: 0.95rem;
    line-height: 1.6;
    color: rgba(203, 213, 225, 0.95);
}

/* Сайдбар с контрастным фоном */
.css-1d391kg, [data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(30, 41, 59, 0.95) 0%, rgba(51, 65, 85, 0.95) 100%);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
}

[data-testid="stSidebar"] .stMarkdown {
    color: white !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stMultiSelect label {
    color: white !important;
    text-shadow: 0 1px 4px rgba(0,0,0,0.3);
}

[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stSlider > div > div,
[data-testid="stSidebar"] .stMultiSelect > div > div {
    background: rgba(255, 255, 255, 0.15) !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    color: white !important;
    backdrop-filter: blur(10px);
}

[data-testid="stSidebar"] .stSelectbox svg,
[data-testid="stSidebar"] .stMultiSelect svg {
    color: white !important;
}

/* Заголовки категорий */
.category-header {
    background: rgba(30, 41, 59, 0.6);
    backdrop-filter: blur(15px);
    -webkit-backdrop-filter: blur(15px);
    border-radius: 16px;
    padding: 20px 24px;
    margin-top: 32px;
    margin-bottom: 16px;
    border: 1px solid rgba(255, 255, 255, 0.2);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
}

.category-title {
    color: white !important;
    font-size: 1.3rem !important;
    font-weight: 700 !important;
    margin: 0 !important;
    text-shadow: 0 2px 8px rgba(0,0,0,0.3);
}

.category-count {
    color: rgba(255, 255, 255, 0.8) !important;
    font-size: 0.9rem !important;
    margin-top: 4px !important;
}

/* Анимация появления */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.animate-in {
    animation: fadeInUp 0.5s ease-out;
}

/* Пустое состояние */
.empty-state {
    text-align: center;
    padding: 60px 20px;
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: 20px;
    border: 2px dashed rgba(255, 255, 255, 0.4);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
}

.empty-state-icon {
    font-size: 4rem;
    margin-bottom: 16px;
}

.empty-state-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: white;
    margin-bottom: 8px;
    text-shadow: 0 2px 8px rgba(0,0,0,0.2);
}

.empty-state-text {
    color: rgba(255, 255, 255, 0.9);
    font-size: 0.95rem;
}

/* Футер */
.footer {
    text-align: center;
    padding: 32px 20px;
    color: rgba(255, 255, 255, 0.8);
    font-size: 0.85rem;
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(15px);
    -webkit-backdrop-filter: blur(15px);
    border-radius: 16px;
    margin-top: 60px;
    border: 1px solid rgba(255, 255, 255, 0.2);
}

/* Графики Plotly */
.js-plotly-plot .plotly {
    background: rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px;
    backdrop-filter: blur(10px);
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Вспомогательные функции (из analysis.py)
# =============================================================================
ALPHA_NORMALITY = 0.05

def _clean_name(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip().lower().replace("ё", "е")
    s = re.sub(r"[^a-zа-я\- ]+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def two_words_key(name: str) -> str:
    base = _clean_name(name)
    parts = base.split()
    if len(parts) < 2:
        return ""
    return f"{parts[0]} {parts[1]}"

def resolve_key_by_known_set(name: str, known_keys: set) -> str:
    base = _clean_name(name)
    parts = base.split()
    if len(parts) < 2:
        return ""
    a, b = parts[0], parts[1]
    cand1 = f"{a} {b}"
    cand2 = f"{b} {a}"
    if cand1 in known_keys:
        return cand1
    if cand2 in known_keys:
        return cand2
    return cand1

def apply_special_aliases(key: str) -> str:
    if key in {"катя автушко", "автушко катя"}:
        return "автушко екатерина"
    return key

def normality_test(series: pd.Series, alpha: float = ALPHA_NORMALITY):
    x = pd.to_numeric(series, errors="coerce").dropna().values
    n = len(x)
    if n < 8:
        return (False, np.nan, "n<8")
    try:
        if n <= 5000:
            stat, p = shapiro(x)
            return (p > alpha, float(p), "Shapiro-Wilk")
        else:
            stat, p = normaltest(x)
            return (p > alpha, float(p), "D'Agostino")
    except Exception:
        return (False, np.nan, "error")

def corr_auto(x: pd.Series, y: pd.Series, alpha: float = ALPHA_NORMALITY):
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    n = len(df)
    if n < 3:
        return {"method": None, "n": n, "r": np.nan, "p": np.nan, "note": "n<3"}
    if df["x"].nunique() < 2 or df["y"].nunique() < 2:
        return {"method": None, "n": n, "r": np.nan, "p": np.nan, "note": "constant"}
    x_norm, x_p, x_test = normality_test(df["x"], alpha=alpha)
    y_norm, y_p, y_test = normality_test(df["y"], alpha=alpha)
    if x_norm and y_norm:
        r, p = pearsonr(df["x"], df["y"])
        return {
            "method": "Pearson", "n": n, "r": float(r), "p": float(p),
            "x_norm_p": x_p, "x_norm_test": x_test,
            "y_norm_p": y_p, "y_norm_test": y_test, "note": "ok",
        }
    r, p = spearmanr(df["x"], df["y"])
    return {
        "method": "Spearman", "n": n,
        "r": float(r) if r is not None else np.nan,
        "p": float(p) if p is not None else np.nan,
        "x_norm_p": x_p, "x_norm_test": x_test,
        "y_norm_p": y_p, "y_norm_test": y_test,
        "note": "non_normal",
    }

def strength_label(r: float) -> tuple:
    """Возвращает (название силы, CSS-класс)"""
    abs_r = abs(r)
    if abs_r < 0.3:
        return "Слабая", "weak"
    elif abs_r < 0.6:
        return "Средняя", "moderate"
    else:
        return "Сильная", "strong"

# =============================================================================
# Загрузка данных
# =============================================================================
@st.cache_data
def load_data():
    kpi_path = "data_new_500.csv"
    srok_path = "srok_500.csv"
    cl1_path = "observation-checklist-1_500.csv"
    cl2_path = "observation-checklist-2_500.csv"
    zoom_path = "zoom_500.csv"
    ocenka_path = "learner-module-progress_500.csv"

    kpi_df = pd.read_csv(kpi_path)
    teacher_col_kpi = "TeacherName"
    ltv_col = "Сумма LTV_ValueNorm (Period)"

    srok_df = pd.read_csv(srok_path)
    srok_df["Teacher_Key"] = (srok_df["Фамилия"].astype(str) + " " + srok_df["Имя"].astype(str)).apply(two_words_key)
    tenure = pd.to_numeric(srok_df["Срок работы"], errors="coerce")
    srok_df["Стаж_дней_минус_30"] = tenure
    srok_df.loc[srok_df["Стаж_дней_минус_30"] <= 0, "Стаж_дней_минус_30"] = np.nan
    tenure_keys_set = set(srok_df["Teacher_Key"].dropna().unique())

    kpi_df["Teacher_Key"] = kpi_df[teacher_col_kpi].apply(lambda x: resolve_key_by_known_set(x, tenure_keys_set))
    kpi_df["Teacher_Key"] = kpi_df["Teacher_Key"].apply(apply_special_aliases)
    kpi_agg = kpi_df.groupby("Teacher_Key", as_index=False)[ltv_col].sum()

    ltv_tenure = pd.merge(
        kpi_agg, srok_df[["Teacher_Key", "Стаж_дней_минус_30"]],
        on="Teacher_Key", how="inner"
    ).dropna(subset=["Стаж_дней_минус_30"])
    ltv_tenure["LTV_norm_by_tenure"] = ltv_tenure[ltv_col] / ltv_tenure["Стаж_дней_минус_30"]
    teacher_keys_set = set(ltv_tenure["Teacher_Key"].dropna().unique())

    ocenka_df = pd.read_csv(ocenka_path, delimiter=";")
    ocenka_df["ocenka"] = 0
    for i in range(ocenka_df.shape[0]):
        if ocenka_df.iloc[i, 10] == "-":
            ocenka_df.loc[i, "ocenka"] = ocenka_df.iloc[i, 17]
        else:
            ocenka_df.loc[i, "ocenka"] = ocenka_df.iloc[i, 10]
    ocenka_df["ocenka"] = pd.to_numeric(ocenka_df["ocenka"], errors="coerce")

    teacher_col_oc = None
    for cand in ["Имя П", "Преподаватель", "Имя преподавателя", "Teacher", "TeacherName", "Имя пользователя"]:
        if cand in ocenka_df.columns:
            teacher_col_oc = cand
            break
    ocenka_df["Teacher_Key"] = ocenka_df[teacher_col_oc].apply(lambda x: resolve_key_by_known_set(x, teacher_keys_set))
    ocenka_df["Teacher_Key"] = ocenka_df["Teacher_Key"].apply(apply_special_aliases)
    ocenka_df = ocenka_df[ocenka_df["Teacher_Key"].isin(teacher_keys_set)].copy()

    def checklist_features(checklist_path: str, prefix: str):
        cl = pd.read_csv(checklist_path, sep=";")
        score_col = None
        for cand in ["Общий балл", "Средний балл"]:
            if cand in cl.columns:
                score_col = cand
                break
        cols_FL = cl.columns[5:12].tolist()
        use_cols = cols_FL.copy()
        if score_col:
            use_cols = [score_col] + use_cols
        for col in use_cols:
            cl[col] = cl[col].replace("-", np.nan)
            cl[col] = pd.to_numeric(cl[col], errors="coerce")
        cl["Teacher_Key"] = cl["Имя пользователя"].apply(lambda x: resolve_key_by_known_set(x, teacher_keys_set))
        cl["Teacher_Key"] = cl["Teacher_Key"].apply(apply_special_aliases)
        cl = cl[cl["Teacher_Key"].isin(teacher_keys_set)].copy()
        feat = cl.groupby("Teacher_Key", as_index=False)[use_cols].mean(numeric_only=True)
        rename_map = {}
        if score_col:
            rename_map[score_col] = f"{prefix}__{score_col}"
        for c in cols_FL:
            rename_map[c] = f"{prefix}__{c}"
        feat = feat.rename(columns=rename_map)
        return feat

    cl1_feat = checklist_features(cl1_path, "CL1")
    cl2_feat = checklist_features(cl2_path, "CL2")

    zoom = pd.read_csv(zoom_path)
    teacher_col_zoom = None
    for cand in ["Имя П", "Преподаватель", "Имя преподавателя", "Teacher", "TeacherName", "Имя пользователя"]:
        if cand in zoom.columns:
            teacher_col_zoom = cand
            break
    score_col_zoom = "Средняя оценка"
    zoom[score_col_zoom] = zoom[score_col_zoom].astype(str).str.replace(",", ".", regex=False)
    zoom[score_col_zoom] = zoom[score_col_zoom].replace("-", np.nan)
    zoom[score_col_zoom] = pd.to_numeric(zoom[score_col_zoom], errors="coerce")
    zoom["Teacher_Key"] = zoom[teacher_col_zoom].apply(lambda x: resolve_key_by_known_set(x, teacher_keys_set))
    zoom["Teacher_Key"] = zoom["Teacher_Key"].apply(apply_special_aliases)
    zoom = zoom[zoom["Teacher_Key"].isin(teacher_keys_set)].copy()
    zoom_feat = zoom.groupby("Teacher_Key", as_index=False)[score_col_zoom].mean(numeric_only=True)
    zoom_feat = zoom_feat.rename(columns={score_col_zoom: "Zoom__Средняя оценка"})

    df_all = ltv_tenure[["Teacher_Key", "LTV_norm_by_tenure", "Стаж_дней_минус_30"]].copy()
    df_all = df_all.merge(ocenka_df[["Teacher_Key", "ocenka"]], on="Teacher_Key", how="left")
    df_all = df_all.merge(cl1_feat, on="Teacher_Key", how="left")
    df_all = df_all.merge(cl2_feat, on="Teacher_Key", how="left")
    df_all = df_all.merge(zoom_feat, on="Teacher_Key", how="left")

    for col in df_all.select_dtypes(include=['float64', 'int64']).columns:
        df_all[col] = df_all[col].astype(float)

    return df_all, cl1_feat, cl2_feat, zoom_feat

df_all, cl1_feat, cl2_feat, zoom_feat = load_data()

# =============================================================================
# Формирование списка инсайтов с категориями
# =============================================================================
insights = []

# Категория 1: Стаж
category_stazh = "📅 Стаж работы"
insights.append({
    "category": category_stazh,
    "title": "Оценка знаний и Стаж",
    "description": "Проверка независимости: корреляция практически отсутствует.",
    "x_col": "Стаж_дней_минус_30", "y_col": "ocenka",
    "x_label": "Стаж (дни)", "y_label": "Оценка знаний"
})

cl2_cols = [c for c in cl2_feat.columns if c != "Teacher_Key"]
for target in ["средний балл", "обязательная информация", "после занятия", "внешний вид"]:
    for c in cl2_cols:
        if target in c.lower():
            insights.append({
                "category": category_stazh,
                "title": f"Стаж → {c.replace('CL2__', '')}",
                "description": f"Влияние стажа на качество проведения '{c.replace('CL2__', '')}'.",
                "x_col": "Стаж_дней_минус_30", "y_col": c,
                "x_label": "Стаж (дни)", "y_label": c.replace("CL2__", "")
            })
            break

insights.append({
    "category": category_stazh,
    "title": "Стаж → Средняя оценка Zoom",
    "description": "С увеличением стажа оценка уроков в Zoom имеет тенденцию к снижению.",
    "x_col": "Стаж_дней_минус_30", "y_col": "Zoom__Средняя оценка",
    "x_label": "Стаж (дни)", "y_label": "Средняя оценка Zoom"
})

# Категория 2: Знания
category_znaniya = "🎓 Оценка знаний"
for target in ["первое занятие", "начало занятия", "конец занятия"]:
    for c in cl2_cols:
        if target in c.lower():
            insights.append({
                "category": category_znaniya,
                "title": f"Знания → {c.replace('CL2__', '')}",
                "description": f"Связь уровня знаний с пунктом '{c.replace('CL2__', '')}' чек-листа 2.",
                "x_col": c, "y_col": "ocenka",
                "x_label": c.replace("CL2__", ""), "y_label": "Оценка знаний"
            })
            break

# Категория 3: LTV
category_ltv = "💰 LTV преподавателя"
insights.append({
    "category": category_ltv,
    "title": "Знания → LTV (нормированный)",
    "description": "Высокий уровень знаний преподавателя коррелирует с большим LTV на единицу стажа.",
    "x_col": "ocenka", "y_col": "LTV_norm_by_tenure",
    "x_label": "Оценка знаний", "y_label": "LTV / стаж"
})

insights.append({
    "category": category_ltv,
    "title": "Стаж → LTV (нормированный)",
    "description": "Опытные преподаватели приносят больше дохода на единицу стажа.",
    "x_col": "Стаж_дней_минус_30", "y_col": "LTV_norm_by_tenure",
    "x_label": "Стаж (дни)", "y_label": "LTV / стаж"
})

for target in ["внешний вид", "проверка дз"]:
    for c in cl2_cols:
        if target in c.lower():
            insights.append({
                "category": category_ltv,
                "title": f"LTV → {c.replace('CL2__', '')}",
                "description": f"Влияние качества '{c.replace('CL2__', '')}' на доходность преподавателя.",
                "x_col": c, "y_col": "LTV_norm_by_tenure",
                "x_label": c.replace("CL2__", ""), "y_label": "LTV / стаж"
            })
            break

cl1_cols = [c for c in cl1_feat.columns if c != "Teacher_Key"]
for target in ["теория", "начало", "неожиданности"]:
    for c in cl1_cols:
        if target in c.lower():
            insights.append({
                "category": category_ltv,
                "title": f"LTV → {c.replace('CL1__', '')}",
                "description": f"Влияние качества '{c.replace('CL1__', '')}' на доходность преподавателя.",
                "x_col": c, "y_col": "LTV_norm_by_tenure",
                "x_label": c.replace("CL1__", ""), "y_label": "LTV / стаж"
            })
            break

# =============================================================================
# Вычисление всех корреляций (один раз)
# =============================================================================
@st.cache_data
def compute_all_correlations(_df, _insights):
    results = []
    for ins in _insights:
        x_col, y_col = ins["x_col"], ins["y_col"]
        data = _df[[x_col, y_col]].copy()
        data[x_col] = pd.to_numeric(data[x_col], errors="coerce")
        data[y_col] = pd.to_numeric(data[y_col], errors="coerce")
        data = data.dropna()
        data = data[np.isfinite(data[x_col]) & np.isfinite(data[y_col])]
        if len(data) < 3:
            continue
        corr_res = corr_auto(data[x_col], data[y_col])
        if corr_res.get("method") is None:
            continue
        p = corr_res.get("p", np.nan)
        if np.isnan(p) or p > 0.1:
            continue
        results.append({**ins, **corr_res})
    return results

all_results = compute_all_correlations(df_all, insights)

# =============================================================================
# Сайдбар с фильтрами
# =============================================================================
with st.sidebar:
    st.markdown("## ⚙️ Фильтры")
    
    # Фильтр по силе корреляции
    strength_options = ["Все", "Слабая (|r| < 0.3)", "Средняя (0.3 ≤ |r| < 0.6)", "Сильная (|r| ≥ 0.6)"]
    selected_strength = st.selectbox("💪 Сила корреляции", strength_options, index=0)
    
    # Фильтр по направлению
    direction_options = ["Все", "Положительная", "Отрицательная"]
    selected_direction = st.selectbox("📈 Направление", direction_options, index=0)
    
    # Фильтр по методу
    method_options = ["Все", "Pearson", "Spearman"]
    selected_method = st.selectbox("🧮 Метод", method_options, index=0)
    
    # Фильтр по категории
    categories = sorted(list(set(ins["category"] for ins in insights)))
    selected_category = st.multiselect("📂 Категория", categories, default=categories)
    
    # Порог p-value
    p_threshold = st.slider("🎯 Порог значимости (p ≤)", 0.01, 0.1, 0.1, 0.01)
    
    st.markdown("---")
    st.markdown("### 📖 О методологии")
    st.markdown("""
    **Автоматический выбор метода:**
    - **Pearson** — если оба признака нормально распределены
    - **Spearman** — в остальных случаях
    
    **Проверка нормальности:** Shapiro-Wilk (n ≤ 5000) или D'Agostino.
    """)

# =============================================================================
# Применение фильтров
# =============================================================================
filtered_results = []
for res in all_results:
    r = res["r"]
    p = res["p"]
    method = res["method"]
    category = res["category"]
    
    # Фильтр по p-value
    if p > p_threshold:
        continue
    
    # Фильтр по категории
    if category not in selected_category:
        continue
    
    # Фильтр по методу
    if selected_method != "Все" and method != selected_method:
        continue
    
    # Фильтр по направлению
    if selected_direction == "Положительная" and r <= 0:
        continue
    if selected_direction == "Отрицательная" and r >= 0:
        continue
    
    # Фильтр по силе
    if selected_strength != "Все":
        abs_r = abs(r)
        if "Слабая" in selected_strength and abs_r >= 0.3:
            continue
        if "Средняя" in selected_strength and (abs_r < 0.3 or abs_r >= 0.6):
            continue
        if "Сильная" in selected_strength and abs_r < 0.6:
            continue
    
    filtered_results.append(res)

# =============================================================================
# HERO-секция
# =============================================================================
st.markdown("""
<div class="hero-section animate-in">
    <div class="hero-title">📊 Эффективность преподавателей</div>
    <div class="hero-subtitle">
        Интерактивный сториборд значимых взаимосвязей между ключевыми метриками.
        Найдено <b>{n}</b> статистически значимых корреляций (p ≤ {p:.2f}) среди {total} проверенных гипотез.
    </div>
</div>
""".format(n=len(filtered_results), p=p_threshold, total=len(insights)), unsafe_allow_html=True)

# =============================================================================
# KPI-панель
# =============================================================================
if filtered_results:
    rs = [res["r"] for res in filtered_results]
    avg_r = np.mean(rs)
    pos_count = sum(1 for r in rs if r > 0)
    neg_count = sum(1 for r in rs if r < 0)
    strong_count = sum(1 for r in rs if abs(r) >= 0.6)
    
    st.markdown(f"""
    <div class="kpi-grid animate-in">
        <div class="kpi-card">
            <div class="kpi-icon">🎯</div>
            <div class="kpi-value">{len(filtered_results)}</div>
            <div class="kpi-label">Значимых связей</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">📊</div>
            <div class="kpi-value">{avg_r:+.2f}</div>
            <div class="kpi-label">Средний r</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">🟢</div>
            <div class="kpi-value">{pos_count}</div>
            <div class="kpi-label">Положительных</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">🔴</div>
            <div class="kpi-value">{neg_count}</div>
            <div class="kpi-label">Отрицательных</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">💎</div>
            <div class="kpi-value">{strong_count}</div>
            <div class="kpi-label">Сильных (|r|≥0.6)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-state-icon">🔍</div>
        <div class="empty-state-title">Ничего не найдено</div>
        <div class="empty-state-text">Попробуйте изменить параметры фильтров в боковой панели</div>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# Обзорный heatmap всех корреляций
# =============================================================================
if filtered_results:
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">🗺️</div>
        <div class="section-title">Карта взаимосвязей</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Создаем матрицу для heatmap
    heatmap_data = []
    for res in filtered_results:
        heatmap_data.append({
            "Пара": f"{res['x_label']} ↔ {res['y_label']}",
            "r": res["r"],
            "p": res["p"],
            "Категория": res["category"]
        })
    
    heatmap_df = pd.DataFrame(heatmap_data)
    
    # Bubble chart: размер = |r|, цвет = знак r
    fig_bubble = px.scatter(
        heatmap_df,
        x="Пара",
        y="r",
        size=np.abs(heatmap_df["r"]) * 50,
        color="r",
        color_continuous_scale=[[0, "#ef4444"], [0.5, "#f1f5f9"], [1, "#10b981"]],
        range_color=[-1, 1],
        hover_data=["p", "Категория"],
        labels={"r": "Коэффициент r", "Пара": "Пара показателей"},
        title="Сила и направление корреляций"
    )
    fig_bubble.update_layout(
        height=450,
        template="plotly_white",
        xaxis_tickangle=-45,
        margin=dict(l=40, r=40, t=60, b=120),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    # fig_bubble.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
    st.plotly_chart(fig_bubble, use_container_width=True)

# =============================================================================
# Детальные карточки инсайтов по категориям
# =============================================================================
if filtered_results:
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">🔬</div>
        <div class="section-title">Детальный анализ</div>
        <div class="section-count">{n} инсайтов</div>
    </div>
    """.format(n=len(filtered_results)), unsafe_allow_html=True)
    
    # Группируем по категориям
    from collections import defaultdict
    grouped = defaultdict(list)
    for res in filtered_results:
        grouped[res["category"]].append(res)
    
    for category, items in grouped.items():
        st.markdown(f"""
        <div class="category-header">
            <div class="category-title">{category}</div>
            <div class="category-count">Найдено связей: {len(items)}</div>
        </div>
        """, unsafe_allow_html=True)
        
        for res in items:
            r = res["r"]
            p = res["p"]
            method = res["method"]
            n = res["n"]
            strength_name, strength_class = strength_label(r)
            direction = "positive" if r > 0 else "negative"
            emoji = "🟢" if r > 0 else "🔴"
            dir_badge = "direction-pos" if r > 0 else "direction-neg"
            
            # Подготовка данных для графика
            data = df_all[[res["x_col"], res["y_col"]]].copy()
            data[res["x_col"]] = pd.to_numeric(data[res["x_col"]], errors="coerce")
            data[res["y_col"]] = pd.to_numeric(data[res["y_col"]], errors="coerce")
            data = data.dropna()
            data = data[np.isfinite(data[res["x_col"]]) & np.isfinite(data[res["y_col"]])]
            
            fig = px.scatter(
                data_frame=data,
                x=res["x_col"],
                y=res["y_col"],
                trendline="ols",
                trendline_color_override="#667eea",
                opacity=0.6,
                labels={res["x_col"]: res["x_label"], res["y_col"]: res["y_label"]},
            )
            fig.update_traces(marker=dict(size=9, color="#667eea", line=dict(width=1, color="white")))
            fig.update_layout(
                height=340,
                template="plotly_white",
                margin=dict(l=40, r=20, t=20, b=40),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.markdown(f"""
            <div class="insight-card {direction} animate-in">
                <div class="insight-header">
                    <div class="insight-title">{emoji} {res["title"]}</div>
                    <div class="insight-badges">
                        <span class="badge badge-strength-{strength_class}">💪 {strength_name}</span>
                        <span class="badge badge-{dir_badge}">📈 {'+' if r > 0 else ''}{r:.3f}</span>
                        <span class="badge badge-method">🧮 {method}</span>
                    </div>
                </div>
                <div class="insight-description">{res["description"]}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.plotly_chart(fig, use_container_width=True)
            # fig_bubble.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
            
            st.markdown(f"""
            <div class="insight-stats">
                <span class="stat-pill">📊 <b>r</b> = {r:.3f}</span>
                <span class="stat-pill">🎯 <b>p-value</b> = {p:.4f}</span>
                <span class="stat-pill">👥 <b>n</b> = {n}</span>
                <span class="stat-pill">🧮 <b>Нормальность X</b> = {res.get('x_norm_test', 'N/A')} (p={res.get('x_norm_p', np.nan):.3f})</span>
                <span class="stat-pill">🧮 <b>Нормальность Y</b> = {res.get('y_norm_test', 'N/A')} (p={res.get('y_norm_p', np.nan):.3f})</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div>
                <br>
                <br>
            </div>
            """, unsafe_allow_html=True)

# =============================================================================
# Итоговые выводы и рекомендации
# =============================================================================
if filtered_results:
    # Формируем автоматические выводы на основе данных
    conclusions = []
    
    # # 1. Самый сильный инсайт
    # strongest = max(filtered_results, key=lambda x: abs(x["r"]))
    # direction_word = "положительная" if strongest["r"] > 0 else "отрицательная"
    # conclusions.append({
    #     "title": f"🏆 Самая сильная связь: {strongest['title']}",
    #     "text": f"Коэффициент r = {strongest['r']:+.3f} ({direction_word}). Это наиболее выраженная взаимосвязь в наборе данных."
    # })
    
    # # 2. Баланс направлений
    # pos = sum(1 for r in [x["r"] for x in filtered_results] if r > 0)
    # neg = len(filtered_results) - pos
    # if pos > neg * 1.5:
    #     conclusions.append({
    #         "title": "📈 Преобладание положительных связей",
    #         "text": f"Большинство значимых корреляций ({pos} из {len(filtered_results)}) положительные — рост одних метрик сопровождается ростом других."
    #     })
    # elif neg > pos * 1.5:
    #     conclusions.append({
    #         "title": "📉 Преобладание отрицательных связей",
    #         "text": f"Большинство значимых корреляций ({neg} из {len(filtered_results)}) отрицательные — есть системные противоречия в метриках."
    #     })
    
    # # 3. Вывод по стажу
    # stazh_insights = [x for x in filtered_results if "Стаж" in x["title"] and "Стаж" in x["x_label"]]
    # if stazh_insights:
    #     avg_stazh_r = np.mean([x["r"] for x in stazh_insights])
    #     if avg_stazh_r < -0.1:
    #         conclusions.append({
    #             "title": "📅 Стаж и качество — обратная связь",
    #             "text": f"Стаж в среднем отрицательно коррелирует с качеством (средний r = {avg_stazh_r:+.2f}). Возможно, требуется программа переобучения опытных преподавателей."
    #         })
    #     elif avg_stazh_r > 0.1:
    #         conclusions.append({
    #             "title": "📅 Стаж повышает качество",
    #             "text": f"Стаж положительно связан с ключевыми метриками (средний r = {avg_stazh_r:+.2f}). Опыт работает на результат."
    #         })
    
    # # 4. Вывод по LTV
    # ltv_insights = [x for x in filtered_results if "LTV" in x["y_label"] or "LTV" in x["title"]]
    # if ltv_insights:
    #     avg_ltv_r = np.mean([x["r"] for x in ltv_insights])
    #     conclusions.append({
    #         "title": "💰 Драйверы LTV",
    #         "text": f"На нормированный LTV значимо влияют {len(ltv_insights)} факторов (средний r = {avg_ltv_r:+.2f}). Это точки приложения управленческих усилий."
    #     })
    
    # # 5. Рекомендация
    # conclusions.append({
    #     "title": "💡 Общие выводы",
    #     "text": "Наиболее влиятельным показателем на LTV оказался фактор стажа. Второй по важности - оценка знаний. Дополнительно было выявлено, что NPS имеет сильную отрицательную корреляцию с оценкой знаний. LTV и NPS связаны средне-слабой отрицательноый корреляцией."
    # })

    conclusions.append({
        "title": "Стаж преподавателя является главным фактором, связанным с ростом LTV учеников",
        "text": "Чем дольше преподаватель работает в компании, тем выше средний LTV его учеников. Это указывает на накопительный эффект опыта: преподаватели со временем лучше удерживают учеников, эффективнее работают с возражениями и точнее подбирают образовательную траекторию. Таким образом, инвестиции в удержание и развитие преподавателей могут давать больший эффект для выручки, чем постоянный набор новых сотрудников."
    })

    conclusions.append({
        "title": "Оценка знаний преподавателей является вторым по значимости фактором для LTV",
        "text": "Преподаватели, которые демонстрируют более высокий уровень знания продуктов и процессов компании, связаны с более высоким LTV учеников. Регулярное обучение и контроль знаний влияют не только на качество работы сотрудников, но и на долгосрочную коммерческую эффективность."
    })
    conclusions.append({
        "title": "Между NPS и оценкой знаний выявлена сильная отрицательная связь",
        "text": "Рост результатов тестирования знаний сопровождается снижением NPS. Это может говорить о том, что преподаватели, ориентированные на строгое следование стандартам и глубокое погружение в продукт, не всегда обеспечивают лучший клиентский опыт. Существует риск, что программы обучения и аттестации развивают знания сотрудников быстрее, чем навыки коммуникации и работы с клиентским опытом. Важно: корреляция не означает причинно-следственную связь. Для подтверждения причин потребуется дополнительный анализ."
    })
    conclusions.append({
        "title": "Между LTV и NPS наблюдается слабая или средне-слабая отрицательная связь",
        "text": "Высокие оценки удовлетворенности клиентов не обязательно приводят к более высокому LTV. В отдельных случаях преподаватели с более высоким NPS показывают меньшие показатели долгосрочной монетизации. NPS не может рассматриваться как единственный или ключевой показатель эффективности преподавателя. Высокая удовлетворенность клиентов не гарантирует удержание и повторные покупки."
    })
    conclusions.append({
    "title": "Рекомендации",
    "text": """1. Сделать удержание преподавателей одним из приоритетов.
- Проанализировать причины ухода сотрудников.
- Усилить программы адаптации и карьерного развития.
- Рассмотреть специальные меры поддержки преподавателей после первого года работы когда начинает накапливаться наиболее ценный опыт.
2. Продолжать развивать систему оценки знаний.
- Сохранить регулярное тестирование.
- Использовать результаты для персонализированного обучения.
- Связать программы развития знаний с бизнес-метриками, включая LTV.
3. Проверить причины отрицательной связи между знаниями и NPS.
- Проанализировать уроки преподавателей с высокими знаниями и низким NPS.
- Оценить их коммуникативные навыки, стиль преподавания и взаимодействие с учениками.
- Дополнить обучение модулями по клиентскому сервису, эмпатии и вовлечению учеников.
4. Пересмотреть систему KPI преподавателей. В нее могут входить LTV, NPS, результаты оценки знаний."""
    })

    conclusions.append({
        "title": "Ключевой вывод",
        "text": "Наибольший вклад в долгосрочную ценность учеников вносят опыт преподавателя и его знание продукта. При этом показатели удовлетворенности клиентов (NPS) не только не демонстрируют положительной связи с LTV, но и имеют отрицательную связь как с LTV, так и с уровнем знаний. Это говорит о необходимости рассматривать качество сервиса и коммерческую эффективность как разные управленческие задачи и искать баланс между ними, а не управлять преподавателями исключительно через NPS."
    })
    
    # st.markdown('<div class="conclusion-card animate-in">', unsafe_allow_html=True)
    st.markdown('<div class="conclusion-title">🎯 Ключевые выводы</div>', unsafe_allow_html=True)
    
    for c in conclusions:
        st.markdown(f"""
        <div class="conclusion-item">
            <div class="conclusion-item-title">{c["title"]}</div>
            <div class="conclusion-item-text">{c["text"]}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# Футер
# =============================================================================
st.markdown("""
<div class="footer">
    📊 Сториборд построен на основе {n} преподавателей • Методы: Pearson / Spearman • Порог значимости: p ≤ {p:.2f}<br>
    <span style="opacity: 0.7;">Данные: KPI, стаж, чек-листы наблюдений, Zoom, прогресс учащихся</span>
</div>
""".format(n=len(df_all), p=p_threshold), unsafe_allow_html=True)