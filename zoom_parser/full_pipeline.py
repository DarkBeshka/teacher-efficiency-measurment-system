"""
Единый скрипт, который последовательно выполняет 4 шага обработки Zoom-отчётов:

1) Логика:
   - из "много-секционного" CSV Zoom-отчёта извлекаются секции:
        * "Launch History"
        * "Как проходят ваши занятия?"
   - таблицы склеиваются по "Meeting/Webinar ID"
   - результат сохраняется в step1.csv и печатается в консоль

2) Логика:
   - все CSV внутри ZIP-архива объединяются (конкатенация строк + union колонок)
   - результат сохраняется в step2.csv и печатается в консоль

3) Логика:
   - left join: step1 + step2
   - ключи: "Meeting/Webinar ID" (step1) и "ID" (step2), перед сравнением нормализуются (оставляем только цифры)
   - добавляются поля: Host name, Host email, Participants, Duration (minutes), Group
   - результат сохраняется в step3.csv и печатается в консоль

4) Логика:
   - результаты шага 3 добавляются ВВЕРХ таблицы merged_with_feedback.csv (вертикальная склейка)
   - сопоставление делается по НАЗВАНИЯМ столбцов (порядок в файлах может отличаться)
   - нумерация в столбце "#" в существующей таблице смещается на N,
     где N = число добавленных новых строк (т.е. старая "1" станет "N+1" и т.д.)
   - результат сохраняется в step4.csv и печатается в консоль

Пример:
  python zoom_aggregate_all_steps_v2.py report.csv reports.zip

По умолчанию:
  step1.csv, step2.csv, step3.csv, step4.csv
"""

import argparse
import csv
import re
import zipfile
from io import TextIOWrapper
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# -----------------------------
# Utilities
# -----------------------------

def print_table_csv_like(header: List[str], rows: List[List[str]], title: str = "") -> None:
    """Печатает таблицу как CSV (для читаемости в консоли)."""
    if title:
        print("\n" + title)
        print("-" * len(title))
    print(",".join(str(x) for x in header))
    for r in rows:
        print(",".join("" if v is None else str(v) for v in r))


def write_csv_rows(path: Path, header: List[str], rows: List[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def write_csv_dict_rows(path: Path, header: List[str], rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def read_csv_as_dicts(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows: List[Dict[str, str]] = []
        for r in reader:
            if r is None:
                continue
            if None in r:
                r.pop(None, None)
            rows.append({k: (v if v is not None else "") for k, v in r.items()})
    return header, rows


def dicts_to_rowlists(header: List[str], rows: List[Dict[str, str]]) -> List[List[str]]:
    return [[r.get(h, "") for h in header] for r in rows]


def rowlists_to_dicts(header: List[str], rows: List[List[str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for r in rows:
        rr = r + [""] * max(0, len(header) - len(r))
        out.append({header[i]: rr[i] for i in range(len(header))})
    return out


def safe_int(s: str) -> Optional[int]:
    try:
        return int(str(s).strip())
    except Exception:
        return None


# -----------------------------
# STEP 1
# -----------------------------

SECTION_LAUNCH = "Launch History"
SECTION_SURVEY = "Как проходят ваши занятия?"


def extract_section(csv_path: Path, section_name: str) -> Tuple[List[str], List[List[str]]]:
    """
    Ищет секцию в "много-секционном" CSV Zoom-отчёта.
    Формат: строка с названием секции -> строка заголовков -> строки данных -> пустая строка/EOF.
    Возвращает (header, rows) где rows = list[list[str]].
    """
    header: Optional[List[str]] = None
    rows: List[List[str]] = []
    in_section = False

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) == 1 and row[0].strip() == section_name:
                in_section = True
                header = None
                rows = []
                continue

            if not in_section:
                continue

            if len(row) == 0 or (len(row) == 1 and row[0].strip() == ""):
                if header is not None and rows:
                    break
                continue

            if header is None:
                header = [c.strip() for c in row]
                continue

            if len(row) < len(header):
                row = row + [""] * (len(header) - len(row))
            elif len(row) > len(header):
                row = row[: len(header)]

            rows.append(row)

    if not in_section:
        raise ValueError(f"Секция не найдена: {section_name!r}")
    if header is None:
        raise ValueError(f"Секция найдена, но шапка таблицы отсутствует: {section_name!r}")

    return header, rows


def idx_or_fail(header: List[str], colname: str, section_name: str) -> int:
    try:
        return header.index(colname)
    except ValueError:
        raise ValueError(f"В секции {section_name!r} нет столбца {colname!r}. Доступные: {header}")


def step1_merge_sections(input_csv: Path) -> Tuple[List[str], List[List[str]]]:
    launch_header, launch_rows = extract_section(input_csv, SECTION_LAUNCH)
    survey_header, survey_rows = extract_section(input_csv, SECTION_SURVEY)

    lh_idx_num = idx_or_fail(launch_header, "#", SECTION_LAUNCH)
    lh_idx_topic = idx_or_fail(launch_header, "Topic", SECTION_LAUNCH)
    lh_idx_id = idx_or_fail(launch_header, "Meeting/Webinar ID", SECTION_LAUNCH)
    lh_idx_start = idx_or_fail(launch_header, "Actual Start Time", SECTION_LAUNCH)
    lh_idx_responses = idx_or_fail(launch_header, "Responses", SECTION_LAUNCH)

    sv_idx_id = idx_or_fail(survey_header, "Meeting/Webinar ID", SECTION_SURVEY)
    sv_idx_user = idx_or_fail(survey_header, "User Name", SECTION_SURVEY)
    sv_idx_submitted = idx_or_fail(survey_header, "Submitted Date and Time", SECTION_SURVEY)
    sv_idx_quality = idx_or_fail(survey_header, "Как тебе занятие?", SECTION_SURVEY)
    sv_idx_feedback = idx_or_fail(
        survey_header,
        "Что понравилось/не понравилось? (необязательный вопрос)",
        SECTION_SURVEY,
    )

    survey_by_id: Dict[str, List[Dict[str, str]]] = {}
    for r in survey_rows:
        mid = str(r[sv_idx_id]).strip()
        if not mid:
            continue
        survey_by_id.setdefault(mid, []).append(
            {
                "User Name": r[sv_idx_user],
                "Submitted Date and Time": r[sv_idx_submitted],
                "Оцените качество прошедшего занятия": r[sv_idx_quality],
                "feedback": r[sv_idx_feedback],
            }
        )

    out_header = [
        "#",
        "Topic",
        "Meeting/Webinar ID",
        "Actual Start Time",
        "User Name",
        "Submitted Date and Time",
        "Оцените качество прошедшего занятия",
        "feedback",
    ]
    out_rows: List[List[str]] = []

    for r in launch_rows:
        num = r[lh_idx_num]
        topic = r[lh_idx_topic]
        mid = str(r[lh_idx_id]).strip()
        start_time = r[lh_idx_start]

        responses_raw = str(r[lh_idx_responses]).strip()
        try:
            responses = int(float(responses_raw)) if responses_raw != "" else 0
        except ValueError:
            responses = 0

        base = [num, topic, mid, start_time]

        if responses == 0:
            out_rows.append(base + ["", "", "", ""])
        else:
            matches = survey_by_id.get(mid, [])
            if not matches:
                out_rows.append(base + ["", "", "", ""])
            else:
                for m in matches:
                    out_rows.append(
                        base
                        + [
                            m["User Name"],
                            m["Submitted Date and Time"],
                            m["Оцените качество прошедшего занятия"],
                            m["feedback"],
                        ]
                    )

    return out_header, out_rows


# -----------------------------
# STEP 2
# -----------------------------

def sniff_delimiter_sample(sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except Exception:
        return ","


def read_csv_from_zip(zf: zipfile.ZipFile, name: str) -> Tuple[List[str], List[Dict[str, str]]]:
    with zf.open(name, "r") as raw:
        txt = TextIOWrapper(raw, encoding="utf-8-sig", newline="")
        sample = txt.read(8192)
        delim = sniff_delimiter_sample(sample)
        txt.seek(0)

        reader = csv.DictReader(txt, delimiter=delim)
        header = reader.fieldnames or []
        rows: List[Dict[str, str]] = []
        for r in reader:
            if None in r:
                r.pop(None, None)
            rows.append({k: (v if v is not None else "") for k, v in r.items()})
        return header, rows


def step2_concat_zip_csv(zip_path: Path, include_subdirs: bool = True) -> Tuple[List[str], List[Dict[str, str]]]:
    all_columns: List[str] = []
    all_columns_set = set()
    out_rows: List[Dict[str, str]] = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not include_subdirs:
            csv_names = [n for n in csv_names if "/" not in n]

        for name in csv_names:
            header, rows = read_csv_from_zip(zf, name)
            for c in header:
                if c not in all_columns_set:
                    all_columns.append(c)
                    all_columns_set.add(c)
            out_rows.extend(rows)

    return all_columns, out_rows


# -----------------------------
# STEP 3
# -----------------------------

ADD_FIELDS = ["Host name", "Host email", "Participants", "Duration (minutes)", "Group"]


def normalize_id(value: str) -> str:
    return re.sub(r"\D", "", value or "")


def step3_enrich(
    step1_rows: List[List[str]],
    step1_header: List[str],
    step2_rows: List[Dict[str, str]],
) -> Tuple[List[str], List[List[str]]]:
    idx_id_1 = step1_header.index("Meeting/Webinar ID")

    agg_by_id: Dict[str, Dict[str, str]] = {}
    for r in step2_rows:
        nid = normalize_id(r.get("ID", ""))
        if nid and nid not in agg_by_id:
            agg_by_id[nid] = r

    out_header = step1_header + [c for c in ADD_FIELDS if c not in step1_header]
    out_rows: List[List[str]] = []

    for r in step1_rows:
        nid = normalize_id(r[idx_id_1])
        agg = agg_by_id.get(nid, {})
        out_rows.append(r + [agg.get(c, "") for c in ADD_FIELDS])

    return out_header, out_rows


# -----------------------------
# STEP 4
# -----------------------------

def step4_prepend_into_base(
    new_header: List[str],
    new_rows: List[List[str]],
    base_csv_path: Path,
) -> Tuple[List[str], List[Dict[str, str]]]:
    """
    Добавляет new_rows (с new_header) ВВЕРХ таблицы base_csv_path.

    Важно:
      - сопоставление колонок по имени;
      - "#" в base сдвигается на N (кол-во новых строк);
      - "#" в новых строках перенумеровывается с 1..N.
    """
    base_header, base_rows = read_csv_as_dicts(base_csv_path)

    out_header = list(base_header)
    for c in new_header:
        if c not in out_header:
            out_header.append(c)

    new_dicts = rowlists_to_dicts(new_header, new_rows)

    n_new = len(new_dicts)

    if "#" in out_header:
        for i, r in enumerate(new_dicts, start=1):
            r["#"] = str(i)

    if "#" in base_header:
        for r in base_rows:
            old = safe_int(r.get("#", ""))
            if old is None:
                continue
            r["#"] = str(old + n_new)

    out_rows: List[Dict[str, str]] = []

    for r in new_dicts:
        out_rows.append({c: r.get(c, "") for c in out_header})

    for r in base_rows:
        out_rows.append({c: r.get(c, "") for c in out_header})

    return out_header, out_rows


# -----------------------------
# main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csv", help="Zoom multi-section CSV (Launch History + survey section).")
    ap.add_argument("zip_path", help="ZIP with CSV reports to enrich with host/participants/duration/group.")
    ap.add_argument("--step1-out", default="step1.csv")
    ap.add_argument("--step2-out", default="step2.csv")
    ap.add_argument("--step3-out", default="step3.csv")

    # Step 4
    ap.add_argument(
        "--base-merged-csv",
        default="merged_with_feedback.csv",
        help="Путь к общей таблице, куда нужно добавить строки (вверх).",
    )

    ap.add_argument(
        "--no-subdirs",
        action="store_true",
        help="На шаге 2 не брать CSV из подпапок внутри ZIP.",
    )
    args = ap.parse_args()

    # STEP 1
    h1, r1 = step1_merge_sections(Path(args.input_csv))
    write_csv_rows(Path(args.step1_out), h1, r1)
    print_table_csv_like(h1, r1, title=f"STEP 1: {args.step1_out}")

    # STEP 2
    h2, r2 = step2_concat_zip_csv(Path(args.zip_path), include_subdirs=(not args.no_subdirs))
    write_csv_dict_rows(Path(args.step2_out), h2, r2)
    print_table_csv_like(h2, dicts_to_rowlists(h2, r2), title=f"STEP 2: {args.step2_out}")

    # STEP 3
    h3, r3 = step3_enrich(r1, h1, r2)
    write_csv_rows(Path(args.step3_out), h3, r3)
    print_table_csv_like(h3, r3, title=f"STEP 3: {args.step3_out}")

    # STEP 4
    base_path = Path(args.base_merged_csv)
    if not base_path.exists():
        raise FileNotFoundError(
            f"Не найден базовый файл для шага 4: {base_path}. "
            f"Укажите корректный путь через --base-merged-csv"
        )

    h4, r4_dicts = step4_prepend_into_base(h3, r3, base_path)

    # Перезаписываем исходный merged_with_feedback.csv обновлёнными данными
    write_csv_dict_rows(base_path, h4, r4_dicts)

    print_table_csv_like(
        h4,
        dicts_to_rowlists(h4, r4_dicts),
        title=f"STEP 4: обновлён {base_path}"
    )

if __name__ == "__main__":
    main()

# python full_pipeline.py fresh/report_survey.csv fresh/fresh.zip  (перед этим step4.csv -> merged_with_feedback.csv)