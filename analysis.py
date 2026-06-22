import re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, shapiro, normaltest

# =========================
# Пути к файлам
# =========================
kpi_path = "data_new_500.csv"  # было: "data.csv"
srok_path = "srok_500.csv"
# cl1_path = "observation-checklist-employees-results-20260404-1714-GMT 0300.csv"
# cl2_path = "observation-checklist-employees-results-20260404-1716-GMT 0300.csv"
cl1_path = "observation-checklist-1_500.csv"
cl2_path = "observation-checklist-2_500.csv"
zoom_path = "zoom_500.csv"
ocenka_znanii="learner-module-progress_500.csv"

# =========================
# Порог значимости для тестов
# =========================
ALPHA_NORMALITY = 0.05


# =========================
# Нормализация/матчинг имён
# =========================
def _clean_name(s: str) -> str:
    """lower, ё->е, убрать лишние символы, нормализовать пробелы."""
    if pd.isna(s):
        return ""
    s = str(s).strip().lower().replace("ё", "е")
    # оставляем рус/лат буквы, дефис и пробел
    s = re.sub(r"[^a-zа-я\- ]+", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def two_words_key(name: str) -> str:
    """Первые 2 слова после чистки: 'word1 word2'."""
    base = _clean_name(name)
    parts = base.split()
    if len(parts) < 2:
        return ""
    return f"{parts[0]} {parts[1]}"


def resolve_key_by_known_set(name: str, known_keys: set) -> str:
    """
    Делает ключ по первым 2 словам и учитывает возможный порядок:
    'фамилия имя' vs 'имя фамилия'. Возвращает тот вариант, который есть в known_keys.
    Если ни один не найден — возвращает базовый вариант.
    """
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
    """
    Уникальные алиасы/исправления (после clean):
    Катя Автушко (в любом порядке) -> автушко екатерина
    """
    if key in {"катя автушко", "автушко катя"}:
        return "автушко екатерина"
    return key


# =========================
# Нормальность: тест и решение про Пирсона
# =========================
def normality_test(series: pd.Series, alpha: float = ALPHA_NORMALITY):
    """
    Возвращает (is_normal, p_value, test_name).
    Используем:
      - Shapiro-Wilk для n<=5000
      - D'Agostino K^2 (normaltest) для n>5000 (и вообще для n>=8)
    Примечание: если n слишком мал, тесты ненадёжны — вернём False (т.е. Пирсона не считаем).
    """
    x = pd.to_numeric(series, errors="coerce").dropna().values
    n = len(x)

    # Для очень малых выборок нормальность проверять бессмысленно
    if n < 8:
        return (False, np.nan, "n<8 (skip)")

    try:
        if n <= 5000:
            stat, p = shapiro(x)
            return (p > alpha, float(p), "Shapiro-Wilk")
        else:
            stat, p = normaltest(x)  # требует n>=8
            return (p > alpha, float(p), "D'Agostino K^2")
    except Exception:
        return (False, np.nan, "test_error")


def corr_auto(x: pd.Series, y: pd.Series, alpha: float = ALPHA_NORMALITY):
    """
    Автоматически выбирает:
      - Pearson, если обе серии "не отвергают нормальность" (p>alpha) и есть вариативность
      - иначе Spearman
    Возвращает dict с метаданными.
    """
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    n = len(df)
    if n < 3:
        return {"method": None, "n": n, "r": np.nan, "p": np.nan, "note": "n<3"}

    if df["x"].nunique() < 2 or df["y"].nunique() < 2:
        return {"method": None, "n": n, "r": np.nan, "p": np.nan, "note": "constant_input"}

    x_norm, x_p, x_test = normality_test(df["x"], alpha=alpha)
    y_norm, y_p, y_test = normality_test(df["y"], alpha=alpha)

    # Pearson только если обе "нормальные" по тесту
    if x_norm and y_norm:
        r, p = pearsonr(df["x"], df["y"])
        return {
            "method": "Pearson",
            "n": n,
            "r": float(r),
            "p": float(p),
            "x_norm_p": x_p,
            "x_norm_test": x_test,
            "y_norm_p": y_p,
            "y_norm_test": y_test,
            "note": "ok",
        }

    # иначе Spearman
    r, p = spearmanr(df["x"], df["y"])
    return {
        "method": "Spearman",
        "n": n,
        "r": float(r) if r is not None else np.nan,
        "p": float(p) if p is not None else np.nan,
        "x_norm_p": x_p,
        "x_norm_test": x_test,
        "y_norm_p": y_p,
        "y_norm_test": y_test,
        "note": "non_normal_or_small_n",
    }


# =========================
# 1) KPI / LTV: читаем и готовим
# =========================
kpi_df = pd.read_csv(kpi_path)

teacher_col_kpi = "TeacherName"
ltv_col = "Сумма LTV_ValueNorm (Period)"

if teacher_col_kpi not in kpi_df.columns:
    raise ValueError(f"В файле KPI нет колонки '{teacher_col_kpi}'. Доступные: {list(kpi_df.columns)}")
if ltv_col not in kpi_df.columns:
    raise ValueError(f"В файле KPI нет колонки '{ltv_col}'. Доступные: {list(kpi_df.columns)}")


# =========================
# 2) Стаж: читаем и готовим (с тем же ключом)
# =========================
srok_df = pd.read_csv(srok_path)

if "Фамилия" not in srok_df.columns or "Имя" not in srok_df.columns or "Срок работы" not in srok_df.columns:
    raise ValueError(
        "В файле srok.csv ожидаются колонки: 'Фамилия', 'Имя', 'Срок работы'. "
        f"Доступные: {list(srok_df.columns)}"
    )

# Нормализованный ключ (Фамилия + Имя)
srok_df["Teacher_Key"] = (srok_df["Фамилия"].astype(str) + " " + srok_df["Имя"].astype(str)).apply(two_words_key)

# Стаж: -30 и clip(upper=393) как у вас
tenure = pd.to_numeric(srok_df["Срок работы"], errors="coerce")# - 30
srok_df["Стаж_дней_минус_30"] = tenure
srok_df.loc[srok_df["Стаж_дней_минус_30"] <= 0, "Стаж_дней_минус_30"] = np.nan

# Множество ключей из стажа — по нему нормализуем KPI и остальные таблицы
tenure_keys_set = set(srok_df["Teacher_Key"].dropna().unique())


# =========================
# 3) KPI ключ: выбираем порядок по known_set стажа
# =========================
kpi_df["Teacher_Key"] = kpi_df[teacher_col_kpi].apply(lambda x: resolve_key_by_known_set(x, tenure_keys_set))
kpi_df["Teacher_Key"] = kpi_df["Teacher_Key"].apply(apply_special_aliases)

# Агрегируем LTV по Teacher_Key
kpi_agg = kpi_df.groupby("Teacher_Key", as_index=False)[ltv_col].sum()


# =========================
# 4) Мёрдж LTV + стаж и нормирование
# =========================
ltv_tenure = (
    pd.merge(
        kpi_agg,
        srok_df[["Teacher_Key", "Стаж_дней_минус_30"]],
        on="Teacher_Key",
        how="inner",
    )
    .dropna(subset=["Стаж_дней_минус_30"])
)

ltv_tenure["LTV_norm_by_tenure"] = ltv_tenure[ltv_col] / ltv_tenure["Стаж_дней_минус_30"]

teacher_keys_set = set(ltv_tenure["Teacher_Key"].dropna().unique())
print("Преподавателей с нормированным LTV (после мёрджа LTV+стаж):", len(teacher_keys_set))

# =========================
# 5) Данные по оценке знаний
# =========================

ocenka_df=pd.read_csv(ocenka_znanii, delimiter=";")

ocenka_df["ocenka"]=0
for i in range(ocenka_df.shape[0]):
    if ocenka_df.iloc[i,10]=="-":
        ocenka_df["ocenka"].iloc[i]=ocenka_df.iloc[i,17]
    else:
        ocenka_df["ocenka"].iloc[i]=ocenka_df.iloc[i,10]
# print(ocenka_df)

# =========================
# Корреляция с чек-листами + диагностика несматченных
# =========================
def corr_with_checklist(checklist_path: str, label: str):
    cl = pd.read_csv(checklist_path, sep=";")

    # итоговый столбец (если есть)
    score_col = None
    for cand in ["Общий балл", "Средний балл"]:
        if cand in cl.columns:
            score_col = cand
            break

    # столбцы F–L (6–12 в Excel): как в вашем исходном подходе
    cols_FL = cl.columns[5:12].tolist()

    # чистим значения: '-' -> NaN, всё -> float
    for col in cols_FL:
        cl[col] = cl[col].replace("-", np.nan)
        cl[col] = pd.to_numeric(cl[col], errors="coerce")
    if score_col:
        cl[score_col] = cl[score_col].replace("-", np.nan)
        cl[score_col] = pd.to_numeric(cl[score_col], errors="coerce")

    # Ключ преподавателя в чек-листе: учитываем "ИФ/ФИ" относительно teacher_keys_set
    cl["Teacher_Key"] = cl["Имя пользователя"].apply(lambda x: resolve_key_by_known_set(x, teacher_keys_set))
    cl["Teacher_Key"] = cl["Teacher_Key"].apply(apply_special_aliases)

    # merge по ключу
    use_cols = ["Имя пользователя", "Teacher_Key"] + ([score_col] if score_col else []) + cols_FL
    merged = pd.merge(
        ltv_tenure[["Teacher_Key", "LTV_norm_by_tenure"]],
        cl[use_cols],
        on="Teacher_Key",
        how="inner",
    )

    print(f"\n=== {label} ===")
    print("Пересечение преподавателей:", merged["Teacher_Key"].nunique())

    # ----- Диагностика несматченных из чек-листа -----
    not_matched_from_cl = (
        cl.loc[~cl["Teacher_Key"].isin(teacher_keys_set), ["Имя пользователя", "Teacher_Key"]]
        .drop_duplicates()
        .sort_values("Имя пользователя")
    )
    if len(not_matched_from_cl) > 0:
        print(f"\nНе сматчились из чек-листа ({len(not_matched_from_cl)}):")
        print(not_matched_from_cl.to_string(index=False))
    else:
        print("\nВсе преподаватели из чек-листа сматчились с LTV+стаж.")

    # ----- Корреляции (Pearson только если нормальность, иначе Spearman) -----
    results = []
    test_cols = ([score_col] if score_col else []) + cols_FL

    for col in test_cols:
        out = corr_auto(merged["LTV_norm_by_tenure"], merged[col], alpha=ALPHA_NORMALITY)
        if out["method"] is None:
            continue
        results.append(
            {
                "Показатель": col,
                "N": out["n"],
                "Метод": out["method"],
                "r": out["r"],
                "p-value": out["p"],
                "Нормальность X p": out.get("x_norm_p", np.nan),
                "Тест X": out.get("x_norm_test", ""),
                "Нормальность Y p": out.get("y_norm_p", np.nan),
                "Тест Y": out.get("y_norm_test", ""),
                "Примечание": out.get("note", ""),
            }
        )

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df = res_df.sort_values(["Метод", "r"], ascending=[True, False])  # Pearson/ Spearman, затем r
    print(res_df)

    return merged, res_df

def corr_with_ocenka():
    global ocenka_df
    print("LTV+Оценка знаний")
    # колонка преподавателя
    teacher_col = None
    for cand in ["Имя П", "Преподаватель", "Имя преподавателя", "Teacher", "TeacherName", "Имя пользователя"]:
        if cand in ocenka_df.columns:
            teacher_col = cand
            break
    if teacher_col is None:
        raise ValueError(f"Не найдена колонка с преподавателем в {zoom_path}. Доступные колонки: {list(ocenka_df.columns)}")

    score_col = "ocenka"

    # оценка: иногда "4,8" -> заменим запятую
    ocenka_df[score_col] = ocenka_df[score_col].astype(str).str.replace(",", ".", regex=False)
    ocenka_df[score_col] = ocenka_df[score_col].replace("-", np.nan)
    ocenka_df[score_col] = pd.to_numeric(ocenka_df[score_col], errors="coerce")

    # ключ преподавателя: учитываем "ИФ/ФИ" относительно teacher_keys_set
    ocenka_df["Teacher_Key"] = ocenka_df[teacher_col].apply(lambda x: resolve_key_by_known_set(x, teacher_keys_set))
    ocenka_df["Teacher_Key"] = ocenka_df["Teacher_Key"].apply(apply_special_aliases)

    merged = pd.merge(
        ltv_tenure[["Teacher_Key", "LTV_norm_by_tenure"]],
        ocenka_df[[teacher_col, "Teacher_Key", score_col]],
        on="Teacher_Key",
        how="inner",
    )

    print("Пересечение преподавателей:", merged["Teacher_Key"].nunique())

    # диагностика: что не сматчилось из zoom
    not_matched = (
        ocenka_df.loc[~ocenka_df["Teacher_Key"].isin(teacher_keys_set), [teacher_col, "Teacher_Key"]]
        .drop_duplicates()
        .sort_values(teacher_col)
    )
    if len(not_matched) > 0:
        print(f"\nНе сматчились из zoom ({len(not_matched)}). Примеры (первые 20):")
        print(not_matched.head(20).to_string(index=False))

    # Pearson только если нормальность, иначе Spearman
    out = corr_auto(merged["LTV_norm_by_tenure"], merged[score_col], alpha=ALPHA_NORMALITY)
    if out["method"] is None:
        res_df = pd.DataFrame([{
            "Показатель": score_col,
            "N": out["n"],
            "Метод": None,
            "r": np.nan,
            "p-value": np.nan,
            "Комментарий": out.get("note", "no_data")
        }])
    else:
        res_df = pd.DataFrame([{
            "Показатель": score_col,
            "N": out["n"],
            "Метод": out["method"],
            "r": out["r"],
            "p-value": out["p"],
            "Нормальность X p": out.get("x_norm_p", np.nan),
            "Тест X": out.get("x_norm_test", ""),
            "Нормальность Y p": out.get("y_norm_p", np.nan),
            "Тест Y": out.get("y_norm_test", ""),
            "Примечание": out.get("note", ""),
        }])

    print(res_df)
    return merged, res_df

# =========================
# Корреляция стажа с чек-листами + диагностика несматченных
# =========================
def corr_srok_with_checklist(checklist_path: str, label: str):
    cl = pd.read_csv(checklist_path, sep=";")

    if "Имя пользователя" not in cl.columns:
        raise ValueError(f"В чек-листе нет колонки 'Имя пользователя'. Доступные: {list(cl.columns)}")

    # итоговый столбец (если есть)
    score_col = None
    for cand in ["Общий балл", "Средний балл"]:
        if cand in cl.columns:
            score_col = cand
            break

    # столбцы F–L (6–12 в Excel): как в вашем исходном подходе
    cols_FL = cl.columns[5:12].tolist()

    # чистим значения: '-' -> NaN, всё -> float
    for col in cols_FL:
        cl[col] = cl[col].replace("-", np.nan)
        cl[col] = pd.to_numeric(cl[col], errors="coerce")
    if score_col:
        cl[score_col] = cl[score_col].replace("-", np.nan)
        cl[score_col] = pd.to_numeric(cl[score_col], errors="coerce")

    # Ключ преподавателя в чек-листе: учитываем "ИФ/ФИ" относительно teacher_keys_set
    cl["Teacher_Key"] = cl["Имя пользователя"].apply(lambda x: resolve_key_by_known_set(x, teacher_keys_set))
    cl["Teacher_Key"] = cl["Teacher_Key"].apply(apply_special_aliases)

    # merge по ключу
    use_cols = ["Имя пользователя", "Teacher_Key"] + ([score_col] if score_col else []) + cols_FL
    merged = pd.merge(
        ltv_tenure[["Teacher_Key", "Стаж_дней_минус_30"]],
        cl[use_cols],
        on="Teacher_Key",
        how="inner",
    )

    print(f"\n=== {label} ===")
    print("Пересечение преподавателей:", merged["Teacher_Key"].nunique())

    # ----- Диагностика несматченных из чек-листа -----
    not_matched_from_cl = (
        cl.loc[~cl["Teacher_Key"].isin(teacher_keys_set), ["Имя пользователя", "Teacher_Key"]]
        .drop_duplicates()
        .sort_values("Имя пользователя")
    )
    if len(not_matched_from_cl) > 0:
        print(f"\nНе сматчились из чек-листа ({len(not_matched_from_cl)}):")
        print(not_matched_from_cl.to_string(index=False))
    else:
        print("\nВсе преподаватели из чек-листа сматчились с LTV+стаж.")

    # ----- Корреляции (Pearson только если нормальность, иначе Spearman) -----
    results = []
    test_cols = ([score_col] if score_col else []) + cols_FL

    for col in test_cols:
        out = corr_auto(merged["Стаж_дней_минус_30"], merged[col], alpha=ALPHA_NORMALITY)
        if out["method"] is None:
            continue
        results.append(
            {
                "Показатель": col,
                "N": out["n"],
                "Метод": out["method"],
                "r": out["r"],
                "p-value": out["p"],
                "Нормальность X p": out.get("x_norm_p", np.nan),
                "Тест X": out.get("x_norm_test", ""),
                "Нормальность Y p": out.get("y_norm_p", np.nan),
                "Тест Y": out.get("y_norm_test", ""),
                "Примечание": out.get("note", ""),
            }
        )

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df = res_df.sort_values(["Метод", "r"], ascending=[True, False])  # Pearson/ Spearman, затем r
    print(res_df)

    return merged, res_df


# =========================
# Корреляция с Zoom (Средняя оценка) + диагностика
# =========================
def corr_with_zoom(zoom_path: str, label: str = "Оценки занятий (Zoom)"):
    zoom = pd.read_csv(zoom_path)

    # колонка преподавателя
    teacher_col = None
    for cand in ["Имя П", "Преподаватель", "Имя преподавателя", "Teacher", "TeacherName", "Имя пользователя"]:
        if cand in zoom.columns:
            teacher_col = cand
            break
    if teacher_col is None:
        raise ValueError(f"Не найдена колонка с преподавателем в {zoom_path}. Доступные колонки: {list(zoom.columns)}")

    score_col = "Средняя оценка"
    if score_col not in zoom.columns:
        raise ValueError(f"Не найдена колонка '{score_col}' в {zoom_path}. Доступные колонки: {list(zoom.columns)}")

    # оценка: иногда "4,8" -> заменим запятую
    zoom[score_col] = zoom[score_col].astype(str).str.replace(",", ".", regex=False)
    zoom[score_col] = zoom[score_col].replace("-", np.nan)
    zoom[score_col] = pd.to_numeric(zoom[score_col], errors="coerce")

    # ключ преподавателя: учитываем "ИФ/ФИ" относительно teacher_keys_set
    zoom["Teacher_Key"] = zoom[teacher_col].apply(lambda x: resolve_key_by_known_set(x, teacher_keys_set))
    zoom["Teacher_Key"] = zoom["Teacher_Key"].apply(apply_special_aliases)

    merged = pd.merge(
        ltv_tenure[["Teacher_Key", "LTV_norm_by_tenure"]],
        zoom[[teacher_col, "Teacher_Key", score_col]],
        on="Teacher_Key",
        how="inner",
    )

    print(f"\n=== {label} ===")
    print("Пересечение преподавателей:", merged["Teacher_Key"].nunique())

    # диагностика: что не сматчилось из zoom
    not_matched = (
        zoom.loc[~zoom["Teacher_Key"].isin(teacher_keys_set), [teacher_col, "Teacher_Key"]]
        .drop_duplicates()
        .sort_values(teacher_col)
    )
    if len(not_matched) > 0:
        print(f"\nНе сматчились из zoom ({len(not_matched)}). Примеры (первые 20):")
        print(not_matched.head(20).to_string(index=False))

    # Pearson только если нормальность, иначе Spearman
    out = corr_auto(merged["LTV_norm_by_tenure"], merged[score_col], alpha=ALPHA_NORMALITY)
    if out["method"] is None:
        res_df = pd.DataFrame([{
            "Показатель": score_col,
            "N": out["n"],
            "Метод": None,
            "r": np.nan,
            "p-value": np.nan,
            "Комментарий": out.get("note", "no_data")
        }])
    else:
        res_df = pd.DataFrame([{
            "Показатель": score_col,
            "N": out["n"],
            "Метод": out["method"],
            "r": out["r"],
            "p-value": out["p"],
            "Нормальность X p": out.get("x_norm_p", np.nan),
            "Тест X": out.get("x_norm_test", ""),
            "Нормальность Y p": out.get("y_norm_p", np.nan),
            "Тест Y": out.get("y_norm_test", ""),
            "Примечание": out.get("note", ""),
        }])

    print(res_df)
    return merged, res_df

# =========================
# Корреляция с Zoom (Средняя оценка) + диагностика
# =========================
def corr_srok_with_zoom(zoom_path: str, label: str):
    zoom = pd.read_csv(zoom_path)

    # колонка преподавателя
    teacher_col = None
    for cand in ["Имя П", "Преподаватель", "Имя преподавателя", "Teacher", "TeacherName", "Имя пользователя"]:
        if cand in zoom.columns:
            teacher_col = cand
            break

    score_col = "Средняя оценка"

    # оценка: иногда "4,8" -> заменим запятую
    zoom[score_col] = zoom[score_col].astype(str).str.replace(",", ".", regex=False)
    zoom[score_col] = zoom[score_col].replace("-", np.nan)
    zoom[score_col] = pd.to_numeric(zoom[score_col], errors="coerce")

    # ключ преподавателя: учитываем "ИФ/ФИ" относительно teacher_keys_set
    zoom["Teacher_Key"] = zoom[teacher_col].apply(lambda x: resolve_key_by_known_set(x, teacher_keys_set))
    zoom["Teacher_Key"] = zoom["Teacher_Key"].apply(apply_special_aliases)

    merged = pd.merge(
        ltv_tenure[["Teacher_Key", "Стаж_дней_минус_30"]],
        zoom[[teacher_col, "Teacher_Key", score_col]],
        on="Teacher_Key",
        how="inner",
    )

    print(f"\n=== {label} ===")
    print("Пересечение преподавателей:", merged["Teacher_Key"].nunique())

    # диагностика: что не сматчилось из zoom
    not_matched = (
        zoom.loc[~zoom["Teacher_Key"].isin(teacher_keys_set), [teacher_col, "Teacher_Key"]]
        .drop_duplicates()
        .sort_values(teacher_col)
    )
    if len(not_matched) > 0:
        print(f"\nНе сматчились из zoom ({len(not_matched)}). Примеры (первые 20):")
        print(not_matched.head(20).to_string(index=False))

    # Pearson только если нормальность, иначе Spearman
    out = corr_auto(merged["Стаж_дней_минус_30"], merged[score_col], alpha=ALPHA_NORMALITY)
    if out["method"] is None:
        res_df = pd.DataFrame([{
            "Показатель": score_col,
            "N": out["n"],
            "Метод": None,
            "r": np.nan,
            "p-value": np.nan,
            "Комментарий": out.get("note", "no_data")
        }])
    else:
        res_df = pd.DataFrame([{
            "Показатель": score_col,
            "N": out["n"],
            "Метод": out["method"],
            "r": out["r"],
            "p-value": out["p"],
            "Нормальность X p": out.get("x_norm_p", np.nan),
            "Тест X": out.get("x_norm_test", ""),
            "Нормальность Y p": out.get("y_norm_p", np.nan),
            "Тест Y": out.get("y_norm_test", ""),
            "Примечание": out.get("note", ""),
        }])

    print(res_df)
    return merged, res_df

# =========================
# Корреляция LTV и стажа
# =========================
def corr_with_srok():
    global ltv_tenure
    print("=== LTV+Стаж ===")
    # Pearson только если нормальность, иначе Spearman
    out = corr_auto(ltv_tenure["LTV_norm_by_tenure"], ltv_tenure["Стаж_дней_минус_30"], alpha=ALPHA_NORMALITY)
    if out["method"] is None:
        res_df = pd.DataFrame([{
            "Показатель": "Стаж_дней_минус_30",
            "N": out["n"],
            "Метод": None,
            "r": np.nan,
            "p-value": np.nan,
            "Комментарий": out.get("note", "no_data")
        }])
    else:
        res_df = pd.DataFrame([{
            "Показатель": "Стаж_дней_минус_30",
            "N": out["n"],
            "Метод": out["method"],
            "r": out["r"],
            "p-value": out["p"],
            "Нормальность X p": out.get("x_norm_p", np.nan),
            "Тест X": out.get("x_norm_test", ""),
            "Нормальность Y p": out.get("y_norm_p", np.nan),
            "Тест Y": out.get("y_norm_test", ""),
            "Примечание": out.get("note", ""),
        }])

    print(res_df)
    return res_df

# =========================
# Корреляция Оценки знаний с чек-листами + диагностика несматченных
# =========================
def corr_ocenka_with_checklist(checklist_path: str, label: str):
    global ocenka_df
    cl = pd.read_csv(checklist_path, sep=";")

    # итоговый столбец (если есть)
    score_col = None
    for cand in ["Общий балл", "Средний балл"]:
        if cand in cl.columns:
            score_col = cand
            break

    # столбцы F–L (6–12 в Excel): как в вашем исходном подходе
    cols_FL = cl.columns[5:12].tolist()

    # чистим значения: '-' -> NaN, всё -> float
    for col in cols_FL:
        cl[col] = cl[col].replace("-", np.nan)
        cl[col] = pd.to_numeric(cl[col], errors="coerce")
    if score_col:
        cl[score_col] = cl[score_col].replace("-", np.nan)
        cl[score_col] = pd.to_numeric(cl[score_col], errors="coerce")

    # Ключ преподавателя в чек-листе: учитываем "ИФ/ФИ" относительно teacher_keys_set
    cl["Teacher_Key"] = cl["Имя пользователя"].apply(lambda x: resolve_key_by_known_set(x, teacher_keys_set))
    cl["Teacher_Key"] = cl["Teacher_Key"].apply(apply_special_aliases)

    # merge по ключу
    use_cols = ["Имя пользователя", "Teacher_Key"] + ([score_col] if score_col else []) + cols_FL
    merged = pd.merge(
        ocenka_df[["Teacher_Key", "ocenka"]],
        cl[use_cols],
        on="Teacher_Key",
        how="inner",
    )

    print(f"\n=== {label} ===")
    print("Пересечение преподавателей:", merged["Teacher_Key"].nunique())

    # ----- Диагностика несматченных из чек-листа -----
    not_matched_from_cl = (
        cl.loc[~cl["Teacher_Key"].isin(teacher_keys_set), ["Имя пользователя", "Teacher_Key"]]
        .drop_duplicates()
        .sort_values("Имя пользователя")
    )
    if len(not_matched_from_cl) > 0:
        print(f"\nНе сматчились из чек-листа ({len(not_matched_from_cl)}):")
        print(not_matched_from_cl.to_string(index=False))
    else:
        print("\nВсе преподаватели из чек-листа сматчились с LTV+стаж.")

    # ----- Корреляции (Pearson только если нормальность, иначе Spearman) -----
    results = []
    test_cols = ([score_col] if score_col else []) + cols_FL

    for col in test_cols:
        out = corr_auto(merged["ocenka"], merged[col], alpha=ALPHA_NORMALITY)
        if out["method"] is None:
            continue
        results.append(
            {
                "Показатель": col,
                "N": out["n"],
                "Метод": out["method"],
                "r": out["r"],
                "p-value": out["p"],
                "Нормальность X p": out.get("x_norm_p", np.nan),
                "Тест X": out.get("x_norm_test", ""),
                "Нормальность Y p": out.get("y_norm_p", np.nan),
                "Тест Y": out.get("y_norm_test", ""),
                "Примечание": out.get("note", ""),
            }
        )

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df = res_df.sort_values(["Метод", "r"], ascending=[True, False])  # Pearson/ Spearman, затем r
    print(res_df)

    return merged, res_df


# =========================
# Корреляция Оценки знаний с Zoom (Средняя оценка) + диагностика
# =========================
def corr_ocenka_with_zoom(zoom_path: str, label: str = "Оценки занятий (Zoom)"):
    global ocenka_df
    zoom = pd.read_csv(zoom_path)

    # колонка преподавателя
    teacher_col = None
    for cand in ["Имя П", "Преподаватель", "Имя преподавателя", "Teacher", "TeacherName", "Имя пользователя"]:
        if cand in zoom.columns:
            teacher_col = cand
            break

    score_col = "Средняя оценка"

    # оценка: иногда "4,8" -> заменим запятую
    zoom[score_col] = zoom[score_col].astype(str).str.replace(",", ".", regex=False)
    zoom[score_col] = zoom[score_col].replace("-", np.nan)
    zoom[score_col] = pd.to_numeric(zoom[score_col], errors="coerce")

    # ключ преподавателя: учитываем "ИФ/ФИ" относительно teacher_keys_set
    zoom["Teacher_Key"] = zoom[teacher_col].apply(lambda x: resolve_key_by_known_set(x, teacher_keys_set))
    zoom["Teacher_Key"] = zoom["Teacher_Key"].apply(apply_special_aliases)

    merged = pd.merge(
        ocenka_df[["Teacher_Key", "ocenka"]],
        zoom[[teacher_col, "Teacher_Key", score_col]],
        on="Teacher_Key",
        how="inner",
    )

    print(f"\n=== {label} ===")
    print("Пересечение преподавателей:", merged["Teacher_Key"].nunique())

    # диагностика: что не сматчилось из zoom
    not_matched = (
        zoom.loc[~zoom["Teacher_Key"].isin(teacher_keys_set), [teacher_col, "Teacher_Key"]]
        .drop_duplicates()
        .sort_values(teacher_col)
    )
    if len(not_matched) > 0:
        print(f"\nНе сматчились из zoom ({len(not_matched)}). Примеры (первые 20):")
        print(not_matched.head(20).to_string(index=False))

    # Pearson только если нормальность, иначе Spearman
    out = corr_auto(merged["ocenka"], merged[score_col], alpha=ALPHA_NORMALITY)
    if out["method"] is None:
        res_df = pd.DataFrame([{
            "Показатель": score_col,
            "N": out["n"],
            "Метод": None,
            "r": np.nan,
            "p-value": np.nan,
            "Комментарий": out.get("note", "no_data")
        }])
    else:
        res_df = pd.DataFrame([{
            "Показатель": score_col,
            "N": out["n"],
            "Метод": out["method"],
            "r": out["r"],
            "p-value": out["p"],
            "Нормальность X p": out.get("x_norm_p", np.nan),
            "Тест X": out.get("x_norm_test", ""),
            "Нормальность Y p": out.get("y_norm_p", np.nan),
            "Тест Y": out.get("y_norm_test", ""),
            "Примечание": out.get("note", ""),
        }])

    print(res_df)
    return merged, res_df

# =========================
# Корреляция оценки знаний и стажа
# =========================
def corr_ocenka_with_srok():
    global ltv_tenure, ocenka_df
    print("=== Оценка знаний+Стаж ===")
    # Pearson только если нормальность, иначе Spearman
    out = corr_auto(ltv_tenure["LTV_norm_by_tenure"], ocenka_df["ocenka"], alpha=ALPHA_NORMALITY)
    if out["method"] is None:
        res_df = pd.DataFrame([{
            "Показатель": "ocenka",
            "N": out["n"],
            "Метод": None,
            "r": np.nan,
            "p-value": np.nan,
            "Комментарий": out.get("note", "no_data")
        }])
    else:
        res_df = pd.DataFrame([{
            "Показатель": "ocenka",
            "N": out["n"],
            "Метод": out["method"],
            "r": out["r"],
            "p-value": out["p"],
            "Нормальность X p": out.get("x_norm_p", np.nan),
            "Тест X": out.get("x_norm_test", ""),
            "Нормальность Y p": out.get("y_norm_p", np.nan),
            "Тест Y": out.get("y_norm_test", ""),
            "Примечание": out.get("note", ""),
        }])

    print(res_df)
    return res_df

from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

def checklist_features(checklist_path: str, prefix: str, teacher_keys_set: set) -> pd.DataFrame:
    """
    Возвращает wide-таблицу: Teacher_Key + признаки чек-листа (агрегированы mean по преподавателю).
    prefix нужен, чтобы отличать колонки чек-листа 1 и 2.
    """
    cl = pd.read_csv(checklist_path, sep=";")

    if "Имя пользователя" not in cl.columns:
        raise ValueError(f"В чек-листе нет колонки 'Имя пользователя'. Доступные: {list(cl.columns)}")

    # Итоговый столбец (если есть)
    score_col = None
    for cand in ["Общий балл", "Средний балл"]:
        if cand in cl.columns:
            score_col = cand
            break

    # F–L (6–12 в Excel): как в вашем исходном подходе
    cols_FL = cl.columns[5:12].tolist()

    # чистим значения: '-' -> NaN, всё -> float
    use_cols = cols_FL.copy()
    if score_col:
        use_cols = [score_col] + use_cols

    for col in use_cols:
        cl[col] = cl[col].replace("-", np.nan)
        cl[col] = pd.to_numeric(cl[col], errors="coerce")

    # Teacher_Key с учетом порядка ФИ/ИФ
    cl["Teacher_Key"] = cl["Имя пользователя"].apply(lambda x: resolve_key_by_known_set(x, teacher_keys_set))
    cl["Teacher_Key"] = cl["Teacher_Key"].apply(apply_special_aliases)

    # Оставляем только тех, кто есть в ltv_tenure
    cl = cl[cl["Teacher_Key"].isin(teacher_keys_set)].copy()

    # Агрегируем по преподавателю (если несколько наблюдений)
    feat = cl.groupby("Teacher_Key", as_index=False)[use_cols].mean(numeric_only=True)

    # Переименуем колонки с префиксом
    rename_map = {}
    if score_col:
        rename_map[score_col] = f"{prefix}__{score_col}"
    for c in cols_FL:
        rename_map[c] = f"{prefix}__{c}"
    feat = feat.rename(columns=rename_map)

    return feat


def zoom_features(zoom_path: str, teacher_keys_set: set) -> pd.DataFrame:
    """Teacher_Key + Средняя оценка (mean по преподавателю)."""
    zoom = pd.read_csv(zoom_path)

    # колонка преподавателя
    teacher_col = None
    for cand in ["Имя П", "Преподаватель", "Имя преподавателя", "Teacher", "TeacherName", "Имя пользователя"]:
        if cand in zoom.columns:
            teacher_col = cand
            break
    if teacher_col is None:
        raise ValueError(f"Не найдена колонка с преподавателем в {zoom_path}. Доступные колонки: {list(zoom.columns)}")

    score_col = "Средняя оценка"
    if score_col not in zoom.columns:
        raise ValueError(f"Не найдена колонка '{score_col}' в {zoom_path}. Доступные колонки: {list(zoom.columns)}")

    zoom[score_col] = zoom[score_col].astype(str).str.replace(",", ".", regex=False)
    zoom[score_col] = zoom[score_col].replace("-", np.nan)
    zoom[score_col] = pd.to_numeric(zoom[score_col], errors="coerce")

    zoom["Teacher_Key"] = zoom[teacher_col].apply(lambda x: resolve_key_by_known_set(x, teacher_keys_set))
    zoom["Teacher_Key"] = zoom["Teacher_Key"].apply(apply_special_aliases)

    zoom = zoom[zoom["Teacher_Key"].isin(teacher_keys_set)].copy()
    feat = zoom.groupby("Teacher_Key", as_index=False)[score_col].mean(numeric_only=True)
    feat = feat.rename(columns={score_col: "Zoom__Средняя оценка"})
    return feat


def build_scatter_matrix(
    ltv_tenure: pd.DataFrame,
    cl1_path: str,
    cl2_path: str,
    zoom_path: str,
    out_png: str = "scatter_matrix_all_metrics.png",
    diag: str = "hist",
    figsize_scale: float = 0.55,
):
    """
    Собирает общий датасет и строит матрицу рассеивания.
    diag: 'hist' (быстрее/проще) или 'kde' (если захотите, но может быть тяжелее).
    """
    base = ltv_tenure[["Teacher_Key", "LTV_norm_by_tenure"]].copy()
    teacher_keys_set = set(base["Teacher_Key"].dropna().unique())

    feat1 = checklist_features(cl1_path, prefix="CL1", teacher_keys_set=teacher_keys_set)
    feat2 = checklist_features(cl2_path, prefix="CL2", teacher_keys_set=teacher_keys_set)
    featz = zoom_features(zoom_path, teacher_keys_set=teacher_keys_set)

    # wide merge: всё на Teacher_Key
    df_all = base.merge(feat1, on="Teacher_Key", how="left").merge(feat2, on="Teacher_Key", how="left").merge(featz, on="Teacher_Key", how="left")

    # Для scatter_matrix нужны числовые колонки
    num_cols = [c for c in df_all.columns if c != "Teacher_Key"]
    df_num = df_all[num_cols].copy()

    # Можно убрать столбцы, где почти нет данных (чтобы график был читабельнее)
    # Например, оставляем колонки, где есть хотя бы 5 ненулевых наблюдений
    min_non_na = 5
    keep = [c for c in df_num.columns if df_num[c].notna().sum() >= min_non_na]
    df_num = df_num[keep]

    # Финальный dropna построчно: для scatter_matrix это нормально (он сам не умеет красиво с NaN)
    df_plot = df_num.dropna(axis=0, how="any")
    print("\n=== Scatter matrix: итоговый датасет ===")
    print("Колонок (метрик):", df_plot.shape[1])
    print("Строк (преподавателей после dropna):", df_plot.shape[0])

    if df_plot.shape[0] < 3 or df_plot.shape[1] < 2:
        print("Недостаточно данных для scatter matrix (нужно >=3 строк и >=2 метрик).")
        return df_all

    # Размер картинки растим от числа метрик
    k = df_plot.shape[1]
    figsize = (max(10, k / figsize_scale), max(10, k / figsize_scale))

    axarr = scatter_matrix(
        df_plot,
        figsize=figsize,
        diagonal=diag,
        alpha=0.6,
        marker=".",
    )

    # Повернём подписи осей, чтобы помещались
    for ax in axarr[-1, :]:
        ax.xaxis.label.set_rotation(90)
        ax.xaxis.label.set_ha("right")
    for ax in axarr[:, 0]:
        ax.yaxis.label.set_rotation(0)
        ax.yaxis.label.set_ha("right")

    plt.suptitle("Матрица рассеивания: LTV + чек-листы + оценки уроков", y=1.02)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Scatter matrix сохранена в файл: {out_png}")
    return df_all


# ====== В конец вашего блока "Запуск" добавьте: ======
df_all = build_scatter_matrix(
    ltv_tenure=ltv_tenure,
    cl1_path=cl1_path,
    cl2_path=cl2_path,
    zoom_path=zoom_path,
    out_png="scatter_matrix_all_metrics.png",
    diag="hist",
)

# =========================
# Запуск
# =========================
merged1, res1 = corr_with_checklist(cl1_path, "LTV+Чек-лист 1")
merged2, res2 = corr_with_checklist(cl2_path, "LTV+Чек-лист 2")
merged_zoom, res_zoom = corr_with_zoom(zoom_path, "LTV+Zoom")
res3 = corr_with_srok()
merged4, res4 = corr_srok_with_checklist(cl1_path, "Стаж+Чек-лист 1")
merged5, res5 = corr_srok_with_checklist(cl2_path, "Стаж+Чек-лист 2")
merged6,res6=corr_with_ocenka()
m,r=corr_srok_with_zoom(zoom_path, "Стаж+Zoom")
merged7,res7=corr_ocenka_with_checklist(cl1_path, "Оценка Знаний+Чек-лист 1")
merged8,res8=corr_ocenka_with_checklist(cl1_path, "Оценка Знаний+Чек-лист 2")
merged9,res9=corr_ocenka_with_zoom(zoom_path, "Оценка знаний+Zoom")
res10=corr_ocenka_with_srok()