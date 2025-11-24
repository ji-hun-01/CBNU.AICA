import os
import json
import re
import math
import glob
import pickle
import hashlib
import unicodedata
import numpy as np
from datetime import datetime
from flask import Flask, request, render_template_string, jsonify,send_from_directory
import requests
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
try:
    # from plus_timetable import ko_tokenize as morph_tokenize
    # 기존 plus_timetable 대신, konlpy의 Okt 형태소 분석기를 사용
    try:
        from konlpy.tag import Okt
        _okt = Okt()
        def morph_tokenize(text):
            return _okt.morphs(text or "")
    except Exception:
        morph_tokenize = None
except Exception:
    morph_tokenize = None

app = Flask(__name__)


_NL_STOP_WORDS = [
    "수업", "강의", "과목", "보여줘", "알려줘", "추천", "해주세요", "해줘",
    "찾아줘", "목록", "리스트", "없는", "있는", "없는지", "중간고사", "기말고사",
    "평점", "정보", "주세요", "좀", "으로", "만", "위한", "관련", "대한", "중간", "기말",
    "이상", "이하", "초과", "미만", "이상인", "이하인", "교수", "교수님", "담당", "선생님",
    "나는야", "나는", "나", "듣고싶어", "듣고 싶어", "싶어", "원해", "듣자", "추천해",
    "추천해주세요", "추천해줘", "중에", "수강", "신청", "가능", "가능한"
]
NATURAL_LANGUAGE_STOP_WORDS = tuple(dict.fromkeys(_NL_STOP_WORDS))

PROF_TITLE_SUFFIX = r"(?:교수(?:님)?|담당|선생님)"
RULE_ONLY_FILTER_KEYS = {
    "rating_min", "rating_max", "final", "midterm", "grade", "day", "hour",
    "course_type", "course_type_contains", "department", "professor", "subject",
}
RATING_SORT_HINTS = ("높", "높은", "상위", "상위권", "top", "TOP", "Top")

ROMAN_TO_DIGIT = {
    "III": "3",
    "II": "2",
    "I": "1",
    "Ⅲ": "3",
    "Ⅱ": "2",
    "Ⅰ": "1",
}
DIGIT_TO_ROMAN = {
    "1": ("I", "Ⅰ"),
    "2": ("II", "Ⅱ"),
    "3": ("III", "Ⅲ"),
}


def _strip_professor_reference(text, prof_name):
    if not text or not prof_name:
        return text
    pattern = re.compile(rf"{re.escape(prof_name)}\s*(?:{PROF_TITLE_SUFFIX})?")
    return pattern.sub(" ", text)

def _normalize_text(value):
    if value is None:
        return ""
    return unicodedata.normalize("NFC", str(value)).strip()


def _subject_variant_tokens(name):
    base = _normalize_text(name)
    tokens = set()
    if not base:
        return tokens
    queue = [base]
    tokens.add(base)
    while queue:
        current = queue.pop()
        no_space = current.replace(" ", "")
        if no_space and no_space not in tokens:
            tokens.add(no_space)
            queue.append(no_space)
        for src, dst in ROMAN_TO_DIGIT.items():
            if src in current:
                candidate = current.replace(src, dst)
                if candidate not in tokens:
                    tokens.add(candidate)
                    queue.append(candidate)
        for digit, romans in DIGIT_TO_ROMAN.items():
            if digit in current:
                for roman in romans:
                    candidate = current.replace(digit, roman)
                    if candidate not in tokens:
                        tokens.add(candidate)
                        queue.append(candidate)
        if "원리" in current:
            candidate = current.replace("원리", "원론")
            if candidate not in tokens:
                tokens.add(candidate)
                queue.append(candidate)
        if "원론" in current:
            candidate = current.replace("원론", "원리")
            if candidate not in tokens:
                tokens.add(candidate)
                queue.append(candidate)
    return tokens


def _subject_matches(course_subject, query_text, *, course_tokens=None, query_tokens=None):
    norm_subject = _normalize_text(course_subject)
    norm_query = _normalize_text(query_text)
    if not norm_subject or not norm_query:
        return False
    if norm_query in norm_subject:
        return True
    subject_variants = course_tokens if course_tokens is not None else _subject_variant_tokens(norm_subject)
    query_variants = query_tokens if query_tokens is not None else _subject_variant_tokens(norm_query)
    if not subject_variants or not query_variants:
        return False
    return bool(subject_variants & query_variants)


def _rating_tokens(rating_value):
    tokens = ["평점", "별점"]
    if rating_value is None:
        return tokens
    text = str(rating_value).strip()
    if text:
        tokens.append(f"평점 {text}")
        tokens.append(f"별점 {text}")
    return tokens

MIN_CANDIDATES = 0
ELBOW_DROP_RATIO = 0.6

def _apply_elbow_cut(indices, scores, drop_ratio= ELBOW_DROP_RATIO):
    if not indices or not scores:
        return [], []
    keep = 1
    prev = scores[0]
    for idx in range(1, len(scores)):
        curr = scores[idx]
        if prev <= 0:
            break
        ratio = curr / prev if prev else 0
        if ratio < drop_ratio:
            break
        keep += 1
        prev = curr
    return indices[:keep], scores[:keep]


def _should_sort_by_rating(query_text):
    normalized = _normalize_text(query_text)
    if not normalized:
        return False
    if normalized == "평점":
        return True
    if "평점" in normalized:
        low = normalized.lower()
        return any(hint.lower() in low for hint in RATING_SORT_HINTS)
    return False


def _sort_results_by_rating(items):
    def rating_value(course):
        raw = course.get("평점")
        try:
            return float(raw)
        except (TypeError, ValueError):
            return float("-inf")
    return sorted(items, key=rating_value, reverse=True)

CODE_TO_HOUR = {
    1: 9, 2: 10, 3: 11, 4: 12, 5: 13, 6: 14, 7: 15,
    8: 16, 9: 17, 10: 18, 11: 19, 12: 20, 13: 21, 14: 22,
}
HOUR_TO_CODE = {hour: code for code, hour in CODE_TO_HOUR.items()}
DAY_MAP = {"월": 0, "화": 1, "수": 2, "목": 3, "금": 4, "토": 5, "일": 6}
DAY_INDEX_TO_SYMBOL = {idx: symbol for symbol, idx in DAY_MAP.items()}
TIME_CODE_TO_HOUR = {f"{code:02d}": hour for code, hour in CODE_TO_HOUR.items()}
TIME_CODE_TO_HOUR.update({str(code): hour for code, hour in CODE_TO_HOUR.items()})

# === 수업시간 파싱 함수 ===
def parse_timestr(timestr: str):
    slots = []
    if not timestr:
        return slots
    pattern = re.compile(r"(월|화|수|목|금|토|일)\s*([0-9 ,~]*)")
    for match in pattern.finditer(str(timestr)):
        day_symbol = match.group(1)
        day_idx = DAY_MAP.get(day_symbol)
        if day_idx is None:
            continue
        segment = match.group(2) or ""
        code_values = set()
        for start, end in re.findall(r"(\d{1,2})\s*~\s*(\d{1,2})", segment):
            a, b = int(start), int(end)
            if b < a:
                a, b = b, a
            for code in range(a, b + 1):
                code_values.add(code)
        for raw in re.findall(r"\d{1,2}", segment):
            code_values.add(int(raw.lstrip("0") or "0"))
        for code in sorted(code_values):
            slots.append((day_idx, code))
    return slots

# === 과목 인덱스 생성 ===
def _index_time_fields(course):
    slots = parse_timestr(str(course.get('수업시간', '')))
    code_set = {code for _, code in slots if isinstance(code, int) and code > 0}
    hour_set = {CODE_TO_HOUR.get(code) for code in code_set if CODE_TO_HOUR.get(code)}
    day_set = {DAY_INDEX_TO_SYMBOL.get(day_idx) for day_idx, _ in slots if DAY_INDEX_TO_SYMBOL.get(day_idx)}
    course['_time_codes'] = code_set
    course['_hours'] = hour_set
    course['_days'] = day_set
# 데이터 로드
with open("경영대학_과목_전체 복사본.json", "r", encoding="utf-8") as f:
    courses = json.load(f)
with open("학사일정 복사본.json", "r", encoding="utf-8") as f:
    schedule = json.load(f)

COURSE_SUBJECT_TOKENS = tuple(
    frozenset(_subject_variant_tokens(course.get("과목명", "")))
    for course in courses
)

# 학과/교수 목록 추출
departments = sorted(set(c.get("학과", "") for c in courses if c.get("학과")))
professors = sorted({
    name.strip()
    for c in courses
    for name in re.split(r"[,/&]", str(c.get("담당교수", "")))
    if name and name.strip()
})

# === NLP 인덱싱/검색 구성 ===
CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)

RANK_THRESHOLDS = {
    "bm25_min": 1.0,
    "tfidf_min": 0.06,
    "sem_min": 0.22,
}

MIN_CANDIDATES = 0
ELBOW_DROP_RATIO = 0.6


def ko_tokenize(text: str):
    """Korean tokenizer with morphology-aware fallback."""
    if morph_tokenize is not None:
        try:
            tokens = morph_tokenize(text or "")
            if tokens:
                return tokens
        except Exception:
            pass
    return re.findall(r"[가-힣A-Za-z0-9]+", str(text))


def _contains_dept(text, dept):
    if not text or not dept:
        return False
    pattern = re.compile(rf"(?<![가-힣]){re.escape(dept)}(?![가-힣])")
    return bool(pattern.search(text))


CURRICULUM_DIR = "./커리큘럼 복사본"


def _load_curriculum_data():
    catalog = {}
    if not os.path.isdir(CURRICULUM_DIR):
        return catalog
    for path in glob.glob(os.path.join(CURRICULUM_DIR, "*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        program = _normalize_text(raw.get("program"))
        plan = raw.get("plan")
        if not program or not isinstance(plan, list):
            continue
        
        # 별표 제거 로직 (요청 반영)
        for entry in plan:
            if isinstance(entry.get("courses"), list):
                for course in entry["courses"]:
                    if course.get("name"):
                        course["name"] = course["name"].replace("*", "")
        
        catalog[program] = {
            "catalog_year": raw.get("catalog_year"),
            "plan": plan,
        }
    return catalog


CURRICULUM_DATA = _load_curriculum_data()
CURRICULUM_PROGRAMS = sorted(CURRICULUM_DATA.keys())


WORD_SEP_PATTERN = re.compile(r"[^\w가-힣]+")
_word_re = re.compile(r"[가-힣A-Za-z0-9]+")


def _contains_word_boundary(text, needle):
    if not text or not needle:
        return False
    normalized_text = _normalize_text(text)
    normalized_needle = _normalize_text(needle)
    pattern = re.compile(rf"(?<![가-힣A-Za-z0-9]){re.escape(normalized_needle)}(?![가-힣A-Za-z0-9])")
    return bool(pattern.search(normalized_text))


def _has_day(timestr, day_symbol):
    if not timestr or not day_symbol:
        return False
    target = DAY_MAP.get(day_symbol)
    if target is None:
        return False
    for day_idx, _ in parse_timestr(timestr):
        if day_idx == target:
            return True
    return False


def _has_hour(timestr, hour):
    if not timestr or hour is None:
        return False
    for _, code in parse_timestr(timestr):
        if CODE_TO_HOUR.get(code) == hour:
            return True
    return False


def _resolve_hour_candidates(value):
    text = _normalize_text(value)
    if not text:
        return set()

    meridiem = None
    mode = None  # "code" or "clock"

    if text.startswith("오전"):
        meridiem = "오전"
        text = text[2:].strip()
        mode = "clock"
    elif text.startswith("오후"):
        meridiem = "오후"
        text = text[2:].strip()
        mode = "clock"

    if text.endswith("교시"):
        mode = "code"
        text = text[:-2].strip()
    elif text.endswith("시"):
        mode = "clock"
        text = text[:-1].strip()

    if not text or not re.fullmatch(r"[0-9]+", text):
        return set()

    val = int(text)
    if mode is None:
        if val <= 0:
            mode = "code"
        elif val >= 9:
            mode = "clock"
        else:
            mode = "code"

    codes = set()
    if mode == "code":
        if 1 <= val <= 14:
            codes.add(val)
        else:
            code = HOUR_TO_CODE.get(val)
            if code:
                codes.add(code)
    else:
        hour_val = val
        if meridiem == "오후" and hour_val < 12:
            hour_val += 12
        if meridiem == "오전" and hour_val == 12:
            hour_val = 0
        code = HOUR_TO_CODE.get(hour_val)
        if code:
            codes.add(code)
    return codes


def _matches_hour_filter(timestr, hour_value):
    if not timestr:
        return False
    candidates = _resolve_hour_candidates(hour_value)
    if not candidates:
        return False
    seen_days = set()
    for day_idx, code in parse_timestr(str(timestr)):
        if day_idx in seen_days:
            continue
        seen_days.add(day_idx)
        if code in candidates:
            return True
    return False

    
def _extract_day_token(text):
    if not text:
        return None, text
    pattern = re.compile(r"(월요일|화요일|수요일|목요일|금요일|토요일|일요일|월|화|수|목|금|토|일)")
    for match in pattern.finditer(text):
        token = match.group(1)
        if len(token) == 1:
            prev_char = text[match.start() - 1] if match.start() > 0 else ""
            next_char = text[match.end()] if match.end() < len(text) else ""
            if prev_char and re.match(r"[가-힣A-Za-z0-9]", prev_char):
                continue
            if next_char and re.match(r"[가-힣]", next_char):
                continue
        day_symbol = token[0]
        updated = text[:match.start()] + " " + text[match.end():]
        return day_symbol, updated
    return None, text


def course_time_tokens(timestr: str):
    """수업시간 문자열에서 요일/교시/실제 시각 토큰을 추출."""
    tokens = set()
    if not timestr:
        return tokens

    pattern = re.compile(r"(월|화|수|목|금|토|일)\s*([0-9 ,~]*)")
    for match in pattern.finditer(str(timestr)):
        day_symbol = match.group(1)
        tokens.add(day_symbol)
        tokens.add(f"{day_symbol}요일")

        code_segment = match.group(2) or ""
        code_values = set()
        for raw in re.findall(r"\d{1,2}", code_segment):
            if not raw:
                continue
            code_values.add(int(raw.lstrip("0") or "0"))
        for start, end in re.findall(r"(\d{1,2})\s*~\s*(\d{1,2})", code_segment):
            a, b = int(start), int(end)
            if b < a:
                a, b = b, a
            for code in range(a, b + 1):
                code_values.add(code)
        for code_int in sorted(code_values):
            if code_int <= 0:
                continue
            code_str = f"{code_int:02d}"
            tokens.add(code_str)
            tokens.add(str(code_int))
            tokens.add(f"{code_int}교시")
            tokens.add(f"{code_str}교시")
            tokens.add(f"{day_symbol}{code_int}교시")
            tokens.add(f"{day_symbol}{code_str}교시")
            hour = CODE_TO_HOUR.get(code_int)
            if hour:
                tokens.add(f"{hour}시")
                tokens.add(f"{hour:02d}시")
                tokens.add(f"{day_symbol}{hour}시")
                tokens.add(f"{day_symbol}요일{hour}시")
                tokens.add(f"{day_symbol} {hour}시")
    return tokens


def course_to_doc(course: dict) -> str:
    """검색 신호가 되는 필드를 문서로 결합."""
    subject_tokens = _subject_variant_tokens(course.get("과목명", ""))
    rating_tokens = _rating_tokens(course.get("평점"))
    doc_parts = [
        str(course.get("과목명", "")),
        " ".join(subject_tokens),
        str(course.get("학과", "")),
        str(course.get("담당교수", "")),
        str(course.get("이수구분", "")),
        str(course.get("수업시간", "")),
        str(course.get("중간고사", "")),
        str(course.get("기말고사", "")),
        str(course.get("별점", "")),
        str(course.get("강의계획서", "")),
        str(course.get("수강정원", "")),
        " ".join(course_time_tokens(course.get("수업시간", ""))),
        " ".join(rating_tokens),
    ]
    return " ".join(doc_parts)


DOCS = [course_to_doc(c) for c in courses]
TOK_DOCS = [ko_tokenize(doc) for doc in DOCS]


def _courses_digest(data):
    base = [
        {
            "과목명": c.get("과목명", ""),
            "학과": c.get("학과", ""),
            "담당교수": c.get("담당교수", ""),
            "수업시간": c.get("수업시간", ""),
            "수강정원": c.get("수강정원", ""),
            "평점": c.get("평점", ""),
        }
        for c in data
    ]
    serialized = json.dumps(base, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()


DATA_DIGEST = _courses_digest(courses)
digest_path = os.path.join(CACHE_DIR, "digest.txt")


def _cache_valid():
    try:
        with open(digest_path, "r", encoding="utf-8") as f:
            return f.read().strip() == DATA_DIGEST
    except FileNotFoundError:
        return False


def _write_digest():
    with open(digest_path, "w", encoding="utf-8") as f:
        f.write(DATA_DIGEST)

# --- BM25 (캐시) ---
bm25_cache = os.path.join(CACHE_DIR, "bm25.pkl")
cache_valid = _cache_valid()
bm25_rebuilt = False
if cache_valid and os.path.exists(bm25_cache):
    with open(bm25_cache, "rb") as f:
        BM25 = pickle.load(f)
else:
    BM25 = BM25Okapi(TOK_DOCS)
    with open(bm25_cache, "wb") as f:
        pickle.dump(BM25, f)
    bm25_rebuilt = True

# --- TF-IDF (캐시) ---
tfidf_cache = os.path.join(CACHE_DIR, "tfidf.pkl")
tfidf_mat_cache = os.path.join(CACHE_DIR, "tfidf.npz")
tfidf_rebuilt = False
if cache_valid and os.path.exists(tfidf_cache) and os.path.exists(tfidf_mat_cache):
    with open(tfidf_cache, "rb") as f:
        tfidf = pickle.load(f)
    from scipy import sparse
    TFIDF = sparse.load_npz(tfidf_mat_cache)
else:
    tfidf = TfidfVectorizer(tokenizer=ko_tokenize, ngram_range=(1, 2), min_df=1)
    TFIDF = tfidf.fit_transform(DOCS)
    from scipy import sparse
    with open(tfidf_cache, "wb") as f:
        pickle.dump(tfidf, f)
    sparse.save_npz(tfidf_mat_cache, TFIDF)
    tfidf_rebuilt = True

if not cache_valid or bm25_rebuilt or tfidf_rebuilt:
    _write_digest()


def rank_search_bm25(query: str, k: int = 10):
    """BM25 순위 기반 검색."""
    if not query.strip():
        return list(range(len(courses)))
    q_tokens = ko_tokenize(query)
    scores = BM25.get_scores(q_tokens)
    order = np.argsort(-scores)
    result = []
    for idx in order:
        score = float(scores[idx])
        if score <= 0:
            continue
        result.append(int(idx))
        if len(result) >= k:
            break
    return result


def rank_search_tfidf(query: str, k: int = 10):
    """TF-IDF 코사인 유사도 기반 검색."""
    if not query.strip():
        return list(range(len(courses)))
    q_vector = tfidf.transform([query])
    similarities = linear_kernel(q_vector, TFIDF).ravel()
    order = similarities.argsort()[::-1]
    result = []
    for idx in order:
        score = float(similarities[idx])
        if score <= 0:
            continue
        result.append(int(idx))
        if len(result) >= k:
            break
    return result


# --- Semantic (선택) ---
try:
    from sentence_transformers import SentenceTransformer

    sem_cache = os.path.join(CACHE_DIR, "doc_emb.npy")
    SEM = SentenceTransformer("jhgan/ko-sroberta-multitask")
    if cache_valid and os.path.exists(sem_cache):
        DOC_EMB = np.load(sem_cache)
    else:
        DOC_EMB = SEM.encode(DOCS, normalize_embeddings=True, show_progress_bar=False)
        np.save(sem_cache, DOC_EMB)

    def rank_search_semantic(query: str, k: int = 10):
        if not query.strip():
            return list(range(len(courses)))
        q_vec = SEM.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        similarities = DOC_EMB @ q_vec
        order = similarities.argsort()[::-1]
        result = []
        for idx in order:
            score = float(similarities[idx])
            if score <= 0.1:
                continue
            result.append(int(idx))
            if len(result) >= k:
                break
        return result

except Exception:
    SEM, DOC_EMB = None, None

    def rank_search_semantic(query: str, k: int = 10):
        return rank_search_bm25(query, k)

def get_bm25_rank(query: str, k: int = 10):
    if not query.strip():
        return list(range(len(courses))), [], 0.0
    q_tokens = ko_tokenize(query)
    scores = BM25.get_scores(q_tokens)
    order = np.argsort(-scores)
    ranked = []
    for idx in order:
        score = float(scores[idx])
        if score <= 0:
            continue
        ranked.append((int(idx), score))
        if len(ranked) >= k:
            break
    indices = [idx for idx, _ in ranked]
    score_list = [score for _, score in ranked]
    best = score_list[0] if score_list else 0.0
    return indices, score_list, best


def get_tfidf_rank(query: str, k: int = 10):
    if not query.strip():
        return list(range(len(courses))), [], 0.0
    q_vec = tfidf.transform([query])
    sims = linear_kernel(q_vec, TFIDF).ravel()
    order = sims.argsort()[::-1]
    ranked = []
    for idx in order:
        score = float(sims[idx])
        if score <= 0:
            continue
        ranked.append((int(idx), score))
        if len(ranked) >= k:
            break
    indices = [idx for idx, _ in ranked]
    score_list = [score for _, score in ranked]
    best = score_list[0] if score_list else 0.0
    return indices, score_list, best


def get_sem_rank(query: str, k: int = 10):
    if SEM is None or not query.strip():
        return get_bm25_rank(query, k)
    q_vec = SEM.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
    sims = DOC_EMB @ q_vec
    order = sims.argsort()[::-1]
    ranked = []
    for idx in order:
        score = float(sims[idx])
        if score <= 0.1:
            continue
        ranked.append((int(idx), score))
        if len(ranked) >= k:
            break
    indices = [idx for idx, _ in ranked]
    score_list = [score for _, score in ranked]
    best = score_list[0] if score_list else 0.0
    return indices, score_list, best


def nlp_then_rules(query_text: str, manual_filters: dict, notes: list, rank_k: int = 10):
    nl_filters, labels, kws = parse_natural_language(query_text)
    applied_labels = []
    for key, value in nl_filters.items():
        if key not in manual_filters:
            manual_filters[key] = value
            if labels.get(key):
                applied_labels.append(labels[key])
    if applied_labels:
        notes.append("자연어 조건 적용 → " + ", ".join(applied_labels))
    if kws:
        notes.append("키워드 추출 → " + ", ".join(kws))

    rating_sort_requested = _should_sort_by_rating(query_text)
    rule_only = (not kws) and any(field in manual_filters for field in RULE_ONLY_FILTER_KEYS)
    if rule_only:
        items = filter_courses(manual_filters, keywords=None, candidate_indices=None)
        if rating_sort_requested and items:
            items = _sort_results_by_rating(items)
            notes.append("요청 해석: 평점 상위 정렬 적용")
        notes.append("랭킹검증 -> 규칙(필터)으로 처리")
        diagnostic = "자연어 인식 OK"
        return items, notes, {"자연어 인식 OK"}
    rank_query = " ".join(kws) if kws else (query_text or "")

    modes = []
    if SEM is not None:
        modes.append("semantic")
    modes += ["bm25", "tfidf"]

    best_pick = None
    for m in modes:
        if m == "semantic":
            cand, score_list, best_score = get_sem_rank(rank_query, k=rank_k)
            threshold = RANK_THRESHOLDS["sem_min"]
        elif m == "bm25":
            cand, score_list, best_score = get_bm25_rank(rank_query, k=rank_k)
            threshold = RANK_THRESHOLDS["bm25_min"]
        else:
            cand, score_list, best_score = get_tfidf_rank(rank_query, k=rank_k)
            threshold = RANK_THRESHOLDS["tfidf_min"]

        cand, score_list = _apply_elbow_cut(cand, score_list)
        if not cand:
            continue
        pick = {
            "mode": m,
            "indices": cand,
            "scores": score_list,
            "best": best_score,
            "threshold": threshold,
        }
        if not best_pick or best_score > best_pick["best"]:
            best_pick = pick
        if best_score >= threshold and len(cand) >= MIN_CANDIDATES:
            best_pick = pick
            break

    keyword_args = kws or None
    if best_pick and best_pick["indices"]:
        mode_name = best_pick["mode"]
        indices = best_pick["indices"]
        best_score = best_pick["best"]
        threshold = best_pick["threshold"]
        if best_score >= threshold and len(indices) >= MIN_CANDIDATES:
            items = filter_courses(manual_filters, keywords=keyword_args, candidate_indices=indices)
            notes.append(f"NLP 랭킹 사용 → {mode_name} (top score={best_score:.3f}, candidates={len(indices)})")
            diag = {"mode": "nlp+filters", "score": best_score, "rank_mode": mode_name, "rank_ok": True}
        else:
            items = strict_filter_courses(manual_filters, keywords=keyword_args)
            notes.append("랭킹 후보 부족 → 정확/부분 일치 필터 적용")
            diag = {"mode": "rules_only", "score": best_score, "rank_mode": mode_name, "rank_ok": False}
        if rating_sort_requested and items:
            items = _sort_results_by_rating(items)
            notes.append("요청 해석: 평점 상위 정렬 적용")
        return items, notes, diag

    items = strict_filter_courses(manual_filters, keywords=keyword_args)
    if rating_sort_requested and items:
        items = _sort_results_by_rating(items)
        notes.append("요청 해석: 평점 상위 정렬 적용")
    notes.append("랭킹 생략 → 규칙(필터)만으로 처리")
    return items, notes, {"mode": "rules_only", "score": 0.0, "rank_mode": None, "rank_ok": False}

# HTML 템플릿 (좌우 배치, 확장성 고려)
HTML = """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>학과 AI 시간표 추천 시스템 (프로토타입)</title>
  <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --color-primary: #ac5372;
      --color-primary-light: #fbe6ee;
      --color-primary-dark: #8c3f5d;
      --color-secondary: #ffffff;
      --color-text-dark: #333333;
      --color-text-light: #666666;
      --color-border: #e0e0e0;
      --color-background-soft: #f8f8f8;
      --color-success: #38a169;
      --color-warning: #ecc94b;
      --color-danger: #c53030;
    }
    body {
      font-family: 'Noto Sans KR', Arial, sans-serif;
      margin: 0;
      min-height: 100vh;
      background: linear-gradient(145deg, #fef6fb 0%, #fdf0f2 40%, #f1f5ff 100%);
      color: var(--color-text-dark);
      line-height: 1.6;
    }
    h2, h3 { color: var(--color-primary-dark); border-left: 4px solid var(--color-primary); padding-left: 12px; margin: 24px 0 16px; }
    h2 { font-size: 1.8rem; }
    h3 { font-size: 1.4rem; margin-top: 32px; }

    /* --- [수정된 제목 스타일] --- */
    header {
        background: linear-gradient(100deg, rgba(172, 83, 114, 0.98) 0%, rgba(232, 122, 149, 0.9) 50%, rgba(255, 190, 200, 0.85) 100%);
        padding: 24px 0;
        margin-bottom: 32px;
        box-shadow: 0 10px 30px rgba(172, 83, 114, 0.25);
        display: flex; 
        align-items: center; 
        justify-content: center;
        position: relative; 
        overflow: hidden;
    }
    header h1 {
        color: var(--color-secondary);
        text-align: center;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        border-left: none; 
        padding-left: 0;
        position: relative;
        z-index: 2;
    }
    header img.logo {
        position: absolute; 
        left: 40px; 
        top: 50%;
        transform: translateY(-50%);
        height: 40px; 
        width: auto;
        object-fit: contain; 
    }
    .ai-title-mascot {
        position: absolute;
        right: 36px;
        top: 50%;
        transform: translateY(-50%);
        width: clamp(150px, 17vw, 240px);
        max-width: 22vw;
        height: clamp(200px, 22vw, 320px);
        object-fit: contain;
        z-index: 1;
        opacity: 0.98;
        pointer-events: none;
        filter: drop-shadow(0 12px 18px rgba(0,0,0,0.18));
    }
    /* --- [수정된 제목 스타일 끝] --- */
    
    /* Container padding adjusted to account for full-width header */
    
    .left, .right { padding: 0; }
    .left { flex: 1.35; min-width: 450px; }
    .right {flex: 1.15;
      min-width: 350px;
      background: linear-gradient(180deg, rgba(252, 247, 251, 0.9), rgba(242, 248, 255, 0.85));
      border-radius: 24px;
      padding: 0 20px 24px;
      box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.5);
    }
    .container {
      display: flex;
      flex-direction: row;
      gap: 32px;
      padding: 24px 36px 36px;
      max-width: 1400px;
      margin: 0 auto 40px;
      background: rgba(255, 255, 255, 0.92);
      border-radius: 28px;
      box-shadow: 0 20px 60px rgba(44, 45, 63, 0.12);
      backdrop-filter: blur(6px);
    }
    /* Search Bar & Filters */
    .search-main {
      background: linear-gradient(135deg, rgba(252, 235, 242, 0.95), rgba(240, 252, 255, 0.92));
      padding: 24px;
      border-radius: 22px;
      margin-bottom: 28px;
      box-shadow: 0 10px 30px rgba(149, 70, 110, 0.12);
      border: 1px solid rgba(255, 255, 255, 0.6);
    }
    .search-top { display: flex; align-items: flex-start; gap: 16px; flex-wrap: wrap; }
    .search-top form { flex: 1 1 320px; margin: 0; display: flex; flex-wrap: wrap; gap: 8px; }
    .search-mascot { flex: 0 0 auto; width: 70px; max-width: 15%; min-width: 70px; height:auto; object-fit: contain; align-self: flex-start; }
    .search-input { font-size: 1.2rem; padding: 14px 18px; border-radius: 8px; border: 2px solid var(--color-primary); transition: border-color 0.3s; flex: 1 1 260px; }
    .search-input:focus { border-color: var(--color-primary-dark); outline: none; }
    .search-btn { font-size: 1.1rem; padding: 12px 24px; border-radius: 8px; background: var(--color-primary); color: var(--color-secondary); border: none; cursor: pointer; font-weight: 500; transition: background-color 0.2s, transform 0.1s; }
    .search-btn:hover { background: var(--color-primary-dark); transform: translateY(-1px); }
    
    .row { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 8px; align-items: center; }
    .row label { font-size: 0.95rem; font-weight: 500; color: var(--color-text-dark); display: flex; align-items: center; gap: 6px; }
    .row select, .row input[type="text"] { padding: 8px 10px; border: 1px solid var(--color-border); border-radius: 6px; font-size: 0.9rem; transition: border-color 0.2s; }
    .row select:focus, .row input[type="text"]:focus { border-color: var(--color-primary); outline: none; }

    /* Cards & Buttons */
    .card { border: none; border-radius: 12px; padding: 16px; margin-bottom: 15px; background: var(--color-secondary); box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08); transition: transform 0.2s; }
    .card:hover { transform: translateY(-2px); box-shadow: 0 6px 15px rgba(0, 0, 0, 0.1); }
    .card b { font-weight: 700; color: var(--color-primary-dark); font-size: 1.1rem; }
    .btn { padding: 8px 16px; cursor: pointer; border-radius: 6px; font-weight: 500; transition: background-color 0.2s; border: 1px solid transparent; }
    .add-timetable { background: var(--color-primary); color: var(--color-secondary); border: none; }
    .add-timetable:hover { background: var(--color-primary-dark); }
    
    /* Selected Courses List */
    .selected-list { margin: 16px 0; min-height: 40px; border: 1px solid var(--color-border); border-radius: 8px; padding: 8px; background: var(--color-background-soft); }
    .selected-item { display: inline-block; background: var(--color-primary-light); border-radius: 4px; padding: 6px 12px; margin: 4px; font-size: 0.9rem; color: var(--color-primary-dark); font-weight: 500; }
    .selected-remove-btn { border: none; background: none; color: var(--color-danger); cursor: pointer; font-size: 14px; margin-left: 8px; padding: 0; line-height: 1; }
    .selected-remove-btn:hover { text-decoration: underline; }
    .selected-clear-btn { margin: 10px 0 16px; background: var(--color-danger); color: var(--color-secondary); border: 1px solid var(--color-danger); }
    .selected-clear-btn:hover:not(:disabled) { background: #a11b1b; }
    .selected-clear-btn:disabled { opacity: 0.6; cursor: not-allowed; background: var(--color-border); color: var(--color-text-light); }
    
    /* Timetable & Table */
    table { border-collapse: collapse; width: 100%; font-size: 13px; margin-top: 15px; }
    th, td { border: 1px solid var(--color-border); padding: 8px; text-align: center; }
    th { background: var(--color-primary-light); color: var(--color-primary-dark); font-weight: 600; }
    
    /* NL Info / Notes / Ratings */
    .nl-info { margin: 10px 0 16px; font-size: 0.85rem; }
    .nl-info .note-item { background: #f0fdf4; border-left: 4px solid var(--color-success); padding: 8px 12px; border-radius: 6px; margin-bottom: 6px; line-height: 1.4; color: #14532d; }
    .note-item strong { font-weight: 700; }
    
    .rating-badge { display: inline-flex; align-items: center; gap: 8px; padding: 6px 10px; border-radius: 999px; background: var(--color-background-soft); font-weight: 600; color: var(--color-text-dark); margin-top: 8px; font-size: 0.9rem; }
    .rating-badge.highlight { background: #fffde7; box-shadow: 0 0 0 1px var(--color-warning) inset; color: #78350f; }
    .rating-stars { color: var(--color-warning); letter-spacing: 1px; font-size: 16px; }
    .rating-score { font-size: 0.9rem; font-weight: 700; }
    .rating-missing { color: var(--color-text-light); font-size: 0.85rem; display: inline-block; margin-top: 8px; }
    
    .empty-state { padding: 30px; text-align: center; color: var(--color-text-light); background: var(--color-background-soft); border-radius: 10px; border: 1px dashed var(--color-border); margin-top: 10px; }
    .cross-note { margin-top: 8px; padding: 8px 10px; border-radius: 6px; background: #e0f2fe; color: #075985; font-size: 0.85rem; border-left: 4px solid #38bdf8; }
    
    /* Curriculum Panel */
    .curriculum-panel { border: 1px solid var(--color-border); border-radius: 12px; padding: 16px; background: var(--color-background-soft); margin-top: 16px; }
    .curriculum-controls { display: flex; gap: 10px; align-items: center; margin-bottom: 12px; }
    .curriculum-controls select { width: 100%; padding: 8px 10px; border-radius: 6px; border: 1px solid var(--color-border); }
    .curriculum-years { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
    .curriculum-year-btn { border: 1px solid var(--color-primary); background: var(--color-secondary); border-radius: 999px; padding: 6px 14px; font-size: 0.9rem; cursor: pointer; color: var(--color-primary); font-weight: 500; transition: all 0.2s ease; }
    .curriculum-year-btn:hover { background: var(--color-primary-light); }
    .curriculum-year-btn.active { background: var(--color-primary); color: var(--color-secondary); border-color: var(--color-primary); box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15); }
    
    .curriculum-result { display: flex; flex-direction: column; gap: 16px; font-size: 0.95rem; }
    .curriculum-year-card { border: 1px solid var(--color-border); border-radius: 10px; padding: 15px; background: var(--color-secondary); box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05); }
    .curriculum-year-card h4 { margin: 0 0 12px; font-size: 1.1rem; color: var(--color-primary-dark); display: flex; justify-content: space-between; align-items: center; gap: 8px; border-left: none; padding-left: 0; }
    .curriculum-year-card h4 span { font-size: 0.8rem; color: var(--color-text-light); font-weight: 400; }
    .curriculum-semesters { display: flex; flex-direction: column; gap: 12px; }
    .curriculum-semester-card { border: 1px solid var(--color-border); border-radius: 8px; padding: 12px; background: #fdfefe; }
    .curriculum-semester-title { font-weight: 700; margin-bottom: 8px; font-size: 1rem; color: var(--color-primary); border-bottom: 2px solid var(--color-primary-light); padding-bottom: 4px; }
    .curriculum-course { display: flex; justify-content: space-between; gap: 12px; padding: 6px 0; border-bottom: 1px dashed var(--color-border); align-items: center; }
    .curriculum-course:last-child { border-bottom: none; }
    .curriculum-course-name { font-weight: 500; color: var(--color-text-dark); }
    .curriculum-course-meta { color: var(--color-text-light); font-size: 0.8rem; text-align: right; }
    
    .curriculum-course-actions { display: flex; gap: 8px; align-items: center; }
    .curriculum-search-btn { background: var(--color-primary-dark); color: var(--color-secondary); border: none; padding: 5px 10px; border-radius: 4px; font-size: 0.7rem; cursor: pointer; transition: background-color 0.15s ease; }
    .curriculum-search-btn:hover { background: var(--color-primary); }
    
    /* Schedule Toggle */
    .schedule-month h4 { margin: 12px 0; font-size: 1.1rem; color: var(--color-primary-dark); display: flex; align-items: center; gap: 8px; cursor: pointer; user-select: none; transition: color 0.2s; border-left: none; padding-left: 0; }
    .schedule-month h4:hover { color: var(--color-primary); }
    .schedule-month h4::before { content: '▶'; margin-right: 8px; font-weight: normal; color: var(--color-primary); font-size: 0.75rem; transform: rotate(0deg); transition: transform 0.2s; }
    .schedule-month.open h4::before { content: '▼'; transform: rotate(0deg); } /* '▼' 사용 */
    .event-list { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 8px; padding-left: 20px; border-left: 1px solid var(--color-border); margin-bottom: 15px; }
    .schedule-month.open .event-list { display: flex; }
    
    /* Claude Recommendation */
    #claude-result pre { background: #f5f5f5; border: 1px solid #ddd; padding: 15px; border-radius: 8px; white-space: pre-wrap; word-wrap: break-word; font-size: 0.9rem; margin-top: 10px; color: var(--color-text-dark); }
    .ai-ask { display: flex; gap: 16px; align-items: center; padding: 12px; border-radius: 12px; border: 1px solid var(--color-border); background: var(--color-background-soft); margin-bottom: 12px; }
    .ai-mascot { width: 120px; max-width: 30%; object-fit: contain; }
    .ai-controls { flex: 1; }
    .ai-controls .btn { width: 100%; margin: 8px 0; }
    .ai-controls #claude-result { margin-top: 4px; }
    #user-input { padding: 10px; border: 1px solid var(--color-border); border-radius: 6px; font-size: 0.95rem; width: 100%; box-sizing: border-box; transition: border-color 0.2s; margin-bottom: 8px; }
    #user-input:focus { border-color: var(--color-primary); outline: none; }
    #claude-btn { background: var(--color-primary-dark); color: var(--color-secondary); border: none; }
    #claude-btn:hover { background: #4a5568; }

    /* --- [새로 추가된 검색 결과 토글 스타일] --- */
    .toggle-header { 
        display: flex; 
        justify-content: space-between; 
        align-items: center; 
        cursor: pointer; 
        user-select: none; 
        padding-right: 10px; 
        margin: 24px 0 16px; 
    }
    .toggle-header h3 { 
        margin: 0; /* Remove default h3 margin inside the toggle-header */
        border-left: 4px solid var(--color-primary); 
        padding-left: 12px;
    }
    .toggle-icon { 
        font-size: 1.2rem; 
        color: var(--color-primary); 
        transition: transform 0.3s; 
    }
    .toggle-header.closed .toggle-icon { 
        transform: rotate(0deg) scaleY(0.8); /* 닫힌 상태를 표시하기 위해 아이콘을 변경 */
        content: '◀';
    }
    .toggle-header.open .toggle-icon {
        transform: rotate(180deg) scaleY(0.8); /* 열린 상태를 표시하기 위해 아이콘을 변경 */
    }
    .toggle-header.closed + .toggle-content {
        display: none; 
    }
    .toggle-header.closed { 
        margin-bottom: 12px; /* 닫혔을 때 아래쪽 마진 조정 */
    }
    /* --- [토글 스타일 끝] --- */

    /* Responsive Design */
    @media (max-width: 1024px) {
      .container { flex-direction: column; gap: 24px; padding: 20px; }
      .left, .right { min-width: unset; flex: 1 1 100%; }
      .ai-title-mascot { width: 150px; height: 190px; right: 16px; }
      .search-top { flex-direction: column; }
      .search-mascot { align-self: flex-end; }
      .search-input { width: 100%; }
      .search-btn { margin-left: 0; margin-top: 8px; width: 100%; }
      .row { gap: 10px; }
      .row label { width: 48%; }
      .row select, .row input[type="text"] { flex-grow: 1; }
      .ai-ask { flex-direction: column; align-items: flex-start; }
      .ai-mascot { align-self: flex-end; }
      #user-input { width: 100%; }
      .toggle-header { margin-top: 16px; margin-bottom: 12px; }
    }
    @media (max-width: 600px) {
        header h1 { font-size: 2.0rem; }
        header img.logo {
            height: 30px;
            left: 10px;
        }
        .ai-title-mascot { width: 110px; height: 150px; right: 10px; opacity: 0.9; }
        .row label { width: 100%; margin-bottom: 8px; }
        .row select, .row input[type="text"] { width: 100%; box-sizing: border-box; }
        .search-main { padding: 15px; }
        .search-input { font-size: 1rem; padding: 10px 14px; }
        .search-btn { font-size: 1rem; padding: 10px; }
        h2, h3 { padding-left: 8px; }
    }
  </style>
</head>
<body>
<header>
    <img src="CBNU_LOGO2.png" alt="충북대학교 로고" class="logo">
    <h1>충북대 경영대학 시간표 챗봇</h1>
    <img src="CBNU_CHA3.png" alt="AI 시간표 우왕 캐릭터" class="ai-title-mascot">
</header>
<div class="container">
  <div class="left">
    <div class="search-main">
      <div class="search-top">
        <form id="main-search-form">
          <input type="text" id="main-search-input" class="search-input" placeholder="자연어로 검색하세요! (예: 경영학과 1학년 평점 4점 이상)" />
          <button type="submit" class="search-btn">검색</button>
        </form>
        <img src="CBNU_CHA1.png" alt="과목 검색을 안내하는 우왕 캐릭터" class="search-mascot">
      </div>
      <div class="row">
        <label>학과:
          <select id="filter-department">
            <option value="">전체</option>
            {% for d in departments %}
              <option value="{{d}}">{{d}}</option>
            {% endfor %}
          </select>
        </label>
        <label>요일:
          <select id="filter-day">
            <option value="">전체</option>
            {% for d in ["월","화","수","목","금","토","일"] %}
              <option value="{{d}}">{{d}}</option>
            {% endfor %}
          </select>
        </label>
        <label>교시/시간: <input type="text" id="filter-hour" style="width:60px;" placeholder="예: 2 또는 13"></label>
        <label>과목명: <input type="text" id="filter-subject" style="width:120px;" placeholder="예: 회계"></label>
        <label>중간고사:
          <select id="filter-midterm">
            <option value="">전체</option>
            <option value="있음">있음</option>
            <option value="없음">없음</option>
          </select>
        </label>
        <label>기말고사:
          <select id="filter-final">
            <option value="">전체</option>
            <option value="있음">있음</option>
            <option value="없음">없음</option>
          </select>
        </label>
      </div>
      <div id="nl-info" class="nl-info"></div>
    </div>
    
    <div id="results-toggle-header" class="toggle-header open">
        <h3>🔍 검색 결과</h3>
        <span class="toggle-icon">▼</span>
    </div>
    <div id="results" class="toggle-content"></div>
    <div id="more-results-container"></div>

  </div>
  <div class="right">
    <h2>🗓️ 시간표</h2>
    <div id="timetable"></div>
    <h3>📝 선택한 과목</h3>
    <div class="selected-list" id="selected-list"></div>
    <button class="btn selected-clear-btn" type="button" id="clear-selected-btn">선택 과목 전체 삭제</button>
    
    <h3>💡 우왕이에게 물어보기 (AI에게 물어보기)</h3>
    <div class="ai-ask">
      <img src="CBNU_CHA2.png" alt="우왕 캐릭터" class="ai-mascot">
      <div class="ai-controls">
        <input type="text" id="user-input" placeholder="추천 설명용 입력(예: 2학년, 발표 없는 수업)">
        <button class="btn" id="claude-btn">우왕이에게 질문하기</button>
        <div id="claude-result"></div>
      </div>
    </div>
    
    <h3>📚 학과별 커리큘럼</h3>
    <div class="curriculum-panel">
      <div class="curriculum-controls">
        <select id="curriculum-dept">
          <option value="">학과 선택</option>
          {% for d in curriculum_departments %}
            <option value="{{d}}">{{d}}</option>
          {% endfor %}
        </select>
      </div>
      <div id="curriculum-year-buttons" class="curriculum-years"></div>
      <div id="curriculum-result" class="curriculum-result"></div>
    </div>
    
    <h3>📅 학사일정</h3>
    <div id="schedule-html"></div>
  </div>
</div>
<script>
// 기존 JavaScript 코드 시작
let selectedCourses = [];
const curriculumState = {
  grouped: [],
  currentDept: '',
  catalogYear: '',
  activeYear: '',
  defaultMessage: ''
};

// [새로 추가된 전역 변수]
let allSearchResults = [];
const RESULTS_LIMIT = 10;
let currentResultsCount = 0; // 새로 추가: 현재 로드된 항목 수 추적
// [새로 추가된 전역 변수 끝]

// [시간 충돌 확인을 위한 상수/함수 추가]
const DAY_MAP = {"월": 0, "화": 1, "수": 2, "목": 3, "금": 4, "토": 5, "일": 6};

function parseTimeSlots(timestr) {
    const slots = [];
    if (!timestr) return new Set();
    const pattern = /(월|화|수|목|금|토|일)\s*([0-9 ,~]*)/g;
    let match;
    while ((match = pattern.exec(timestr)) !== null) {
        const daySymbol = match[1];
        const dayIdx = DAY_MAP[daySymbol];
        if (dayIdx === undefined) continue;
        const segment = match[2] || "";
        const codeValues = new Set();
        
        // 범위 (예: 02~04) 파싱
        for (const [start, end] of Array.from(segment.matchAll(/(\d{1,2})\s*~\s*(\d{1,2})/g), m => [parseInt(m[1]), parseInt(m[2])])) {
            const a = Math.min(start, end);
            const b = Math.max(start, end);
            for (let code = a; code <= b; code++) {
                codeValues.add(code);
            }
        }
        // 단일 교시 (예: 05, 06) 파싱
        for (const raw of Array.from(segment.matchAll(/\d{1,2}/g), m => parseInt(m[0]))) {
            if (raw) codeValues.add(raw);
        }
        
        for (const code of codeValues) {
            slots.push(`${dayIdx}-${code}`); // 예: "0-6" (월요일 6교시)
        }
    }
    return new Set(slots);
}

function checkConflict(newCourse, currentCourses) {
    const newSlots = parseTimeSlots(newCourse.수업시간);
    
    // 새 과목에 시간이 없으면 충돌 없음
    if (newSlots.size === 0) return { isConflict: false };

    for (const existingCourse of currentCourses) {
        const existingSlots = parseTimeSlots(existingCourse.수업시간);
        
        for (const newSlot of newSlots) {
            if (existingSlots.has(newSlot)) {
                // 충돌 발생
                const [dayIdx, code] = newSlot.split('-');
                const daySymbol = Object.keys(DAY_MAP).find(key => DAY_MAP[key] === parseInt(dayIdx));
                return {
                    isConflict: true,
                    conflictingCourse: existingCourse.과목명,
                    time: `${daySymbol}요일 ${code}교시`
                };
            }
        }
    }
    return { isConflict: false };
}
// [시간 충돌 확인을 위한 상수/함수 추가 끝]

function normalizeCourseCode(value) {
  if (value === null || value === undefined) {
    return '';
  }
  return String(value).replace(/\.0$/, '');
}
function renderSelectedList() {
  const list = document.getElementById('selected-list');
  if (!list) {
    return;
  }
  if (!selectedCourses.length) {
    list.innerHTML = `<div class="empty-state" style="padding:10px;margin:0; min-height: 40px;">선택된 과목이 없습니다.</div>`;
  } else {
    list.innerHTML = selectedCourses
      .map(c => {
        const code = normalizeCourseCode(c.과목코드);
        return `<span class='selected-item'>${c.과목명 || ''} (${code}) <button type='button' class='selected-remove-btn' onclick='removeCourse(\"${code}\")'>X</button></span>`;
      })
      .join('');
  }
  const clearBtn = document.getElementById('clear-selected-btn');
  if (clearBtn) {
    clearBtn.disabled = selectedCourses.length === 0;
  }
}
function removeCourse(code) {
  const normalized = normalizeCourseCode(code);
  selectedCourses = selectedCourses.filter(c => normalizeCourseCode(c.과목코드) !== normalized);
  renderSelectedList();
  updateTimetable();
}
function updateTimetable() {
  fetch('/timetable', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ courses: selectedCourses })
  }).then(r => r.text()).then(html => {
    document.getElementById('timetable').innerHTML = html;
  });
}
function initScheduleToggle() {
    document.getElementById('schedule-html').addEventListener('click', function(e) {
        // h4 클릭 시 토글되도록 수정
        if (e.target.tagName === 'H4' || e.target.closest('.schedule-month h4')) { 
            const monthContainer = e.target.closest('.schedule-month');
            if (monthContainer) {
                // .open 클래스를 토글하여 내용을 숨기거나 보여줍니다.
                monthContainer.classList.toggle('open');
            }
        }
    });
}
function updateSchedule() {
  fetch('/schedule_html').then(r => r.text()).then(html => {
    document.getElementById('schedule-html').innerHTML = html;
    initScheduleToggle(); // HTML 로드 후 토글 기능 초기화
  });
}

document.addEventListener('click', function(e) {
  if (e.target.classList.contains('add-timetable')) {
    const idx = e.target.getAttribute('data-idx');
    fetch('/course_by_idx?idx=' + idx)
      .then(r => r.json())
      .then(c => {
        const candidateCode = normalizeCourseCode(c.과목코드);
        
        // 1. 이미 추가된 과목인지 확인
        if (selectedCourses.find(x => normalizeCourseCode(x.과목코드) === candidateCode)) {
            alert(`${c.과목명}은(는) 이미 시간표에 추가된 과목입니다.`);
            return;
        }

        // 2. [추가된 로직] 시간 충돌 확인
        const conflictResult = checkConflict(c, selectedCourses);

        if (conflictResult.isConflict) {
            // 충돌 발생 시 알림 표시 및 추가 차단
            alert(`
🚨 시간표 충돌 발생! 🚨
추가하려는 과목: ${c.과목명}
충돌 과목: ${conflictResult.conflictingCourse}
충돌 시간: ${conflictResult.time}
(과목이 시간표에 추가되지 않았습니다.)
            `);
            return; // 추가를 막고 함수 종료
        }
        
        // 3. 충돌이 없으면 정상적으로 추가
        selectedCourses.push(c);
        renderSelectedList();
        updateTimetable();
      });
    return;
  }
  
  // --- 커리큘럼 검색 버튼 로직 시작 ---
  if (e.target.classList.contains('curriculum-search-btn')) {
    const subjectName = e.target.getAttribute('data-subject');
    const deptName = document.getElementById('curriculum-dept').value || '';

    if (subjectName) {
      // 1. 메인 검색 창과 과목 필터에 값 설정
      document.getElementById('main-search-input').value = ''; 
      document.getElementById('filter-subject').value = subjectName;

      // 2. 학과 필터는 커리큘럼 학과로 설정 
      document.getElementById('filter-department').value = deptName;

      // 3. 나머지 수동 필터 초기화
      document.getElementById('filter-day').value = '';
      document.getElementById('filter-hour').value = '';
      document.getElementById('filter-midterm').value = '';
      document.getElementById('filter-final').value = '';
      
      // 4. 검색 실행 (즉시)
      runSearch(true);
      
      // 5. 검색 결과 창으로 스크롤
      const resultsToggleHeader = document.getElementById('results-toggle-header');
      if (resultsToggleHeader) {
          resultsToggleHeader.scrollIntoView({ behavior: 'smooth' });
      }
    }
    return;
  }
  // --- 커리큘럼 검색 버튼 로직 끝 ---

  if (e.target.classList.contains('curriculum-year-btn')) {
    const yearKey = e.target.getAttribute('data-year');
    showCurriculumYear(yearKey);
  }
});
document.getElementById('main-search-form').onsubmit = function(e) {
  e.preventDefault();
  runSearch(true);
};

function renderRating(value) {
  if (value === null || value === undefined || value === '') {
    return `<span class="rating-missing">평점 정보 없음</span>`;
  }
  const rating = Number(value);
  if (Number.isNaN(rating)) {
    return `<span class="rating-missing">평점: ${value}</span>`;
  }
  const fullStars = Math.max(0, Math.min(5, Math.floor(rating)));
  const hasHalf = rating - fullStars >= 0.5;
  let stars = '★'.repeat(Math.min(fullStars, 5));
  if (hasHalf && stars.length < 5) {
    stars += '☆';
  }
  if (stars.length < 5) {
    stars = stars.padEnd(5, '☆');
  }
  const highlightClass = rating >= 4 ? 'rating-badge highlight' : 'rating-badge';
  return `<span class="${highlightClass}"><span class="rating-stars">${stars}</span><span class="rating-score">${rating.toFixed(2)}</span></span>`;
}

function sanitizeNumeric(value) {
  if (value === null || value === undefined || value === '') {
    return '-';
  }
  const num = Number(value);
  if (!Number.isNaN(num)) {
    return Number.isInteger(num) ? String(num) : String(num.toFixed(2)).replace(/\.00$/, '');
  }
  return value;
}

function formatExam(value) {
  if (value === null || value === undefined) {
    return '정보 없음';
  }
  const text = String(value).trim();
  return text ? text : '정보 없음';
}

function buildCourseCard(course) {
  const courseCode = course.과목코드 != null ? String(course.과목코드).replace(/\.0$/, '') : '';
  const grade = course.학년 || '-';
  const division = course.이수구분 || '-';
  const displayDivision = course._treated_as_elective ? '전공선택 (타 학과 전필)' : division;
  const credit = sanitizeNumeric(course.학점);
  const midterm = formatExam(course.중간고사);
  const finalExam = formatExam(course.기말고사);
  const ratingMarkup = renderRating(course.평점);
  const professor = course.담당교수 || '정보 없음'; // 교수명 변수 추가
  const electiveNote = course._treated_as_elective
    ? `<div class='cross-note'>${course._treated_source || '타 학과 전공필수'} → 경영정보학과 전공선택으로 인정</div>`
    : '';

  return `
    <div class='card'>
      <div><b>${course.과목명 || ''}</b> (${courseCode})</div>
      <div>${course.학과 || ''} · 학년: ${grade} · 이수구분: ${displayDivision} · 학점: ${credit}</div>
      <div>🧑‍🏫 담당교수: ${professor}</div> <div>⏰ 수업시간: ${course.수업시간 || '-'}</div>
      <div>📝 중간고사: ${midterm} / 🗓️ 기말고사: ${finalExam}</div>
      ${ratingMarkup}
      ${electiveNote}
      <div style='margin-top:12px;'>
        <button class='btn add-timetable' type='button' data-idx='${course._idx}'>➕ 시간표 추가</button>
      </div>
    </div>
  `;
}

function resetCurriculumView(message, clearYears = true) {
  const yearWrap = document.getElementById('curriculum-year-buttons');
  if (clearYears && yearWrap) {
    yearWrap.innerHTML = '';
  }
  const resultEl = document.getElementById('curriculum-result');
  if (resultEl) {
    resultEl.innerHTML = message
      ? `<div class="empty-state" style="margin:0;">${message}</div>`
      : '';
  }
}

function toNumericValue(value) {
  if (value === null || value === undefined) return null;
  if (typeof value === 'string' && value.trim() === '') return null;
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
}

function normalizeCurriculumPlan(plan) {
  const grouped = new Map();
  (plan || []).forEach(entry => {
    const yearNum = toNumericValue(entry.year);
    const yearKey = yearNum !== null ? String(yearNum) : String(entry.year || '기타');
    if (!grouped.has(yearKey)) {
      grouped.set(yearKey, {
        label: yearNum !== null ? `${yearNum}학년` : yearKey,
        order: yearNum !== null ? yearNum : 99,
        semesters: new Map()
      });
    }
    const semNum = toNumericValue(entry.semester);
    const semKey = semNum !== null ? String(semNum) : String(entry.semester || '학기');
    const semLabel = semNum !== null ? `${semNum}학기` : (entry.semester ? `${entry.semester}` : '학기 정보 없음');
    const yearBucket = grouped.get(yearKey).semesters;
    const courseList = Array.isArray(entry.courses) ? entry.courses : [];
    if (!yearBucket.has(semKey)) {
      yearBucket.set(semKey, {
        label: semLabel,
        order: semNum !== null ? semNum : 99,
        courses: courseList.slice()
      });
    } else {
      const stored = yearBucket.get(semKey);
      stored.courses = stored.courses.concat(courseList);
    }
  });
  return Array.from(grouped.entries())
    .map(([yearKey, info]) => ({
      yearKey,
      yearLabel: info.label,
      order: info.order,
      semesters: Array.from(info.semesters.entries())
        .map(([semKey, semInfo]) => ({
          semesterKey: semKey,
          semesterLabel: semInfo.label,
          order: semInfo.order,
          courses: semInfo.courses
        }))
        .sort((a, b) => a.order - b.order)
    }))
    .sort((a, b) => a.order - b.order);
}

function renderCurriculumYearButtons(groups) {
  const yearWrap = document.getElementById('curriculum-year-buttons');
  if (!yearWrap) return;
  if (!groups.length) {
    yearWrap.innerHTML = '';
    return;
  }
  yearWrap.innerHTML = groups
    .map(group => `<button type="button" class="curriculum-year-btn" data-year="${group.yearKey}">${group.yearLabel}</button>`)
    .join('');
}

function showCurriculumYear(yearKey) {
  if (!curriculumState.grouped.length) {
    return;
  }
  const targetKey = String(yearKey);
  curriculumState.activeYear = targetKey;
  document.querySelectorAll('.curriculum-year-btn').forEach(btn => {
    if (btn.getAttribute('data-year') === targetKey) {
      btn.classList.add('active');
    } else {
      btn.classList.remove('active');
    }
  });
  const resultEl = document.getElementById('curriculum-result');
  if (!resultEl) {
    return;
  }
  const target = curriculumState.grouped.find(item => String(item.yearKey) === targetKey);
  if (!target) {
    resetCurriculumView('선택된 학년에 대한 커리큘럼이 없습니다.', false);
    return;
  }
  let html = `<div class="curriculum-year-card"><h4>${target.yearLabel}`;
  if (curriculumState.catalogYear) {
    html += `<span>(${curriculumState.catalogYear} 기준)</span>`;
  }
  html += `</h4><div class="curriculum-semesters">`;
  target.semesters.forEach(sem => {
    html += `<div class="curriculum-semester-card"><div class="curriculum-semester-title">${sem.semesterLabel}</div>`;
    if (!sem.courses || !sem.courses.length) {
      html += `<div class="empty-state" style="padding:10px; margin:0;">등록된 과목이 없습니다.</div>`;
    } else {
      html += sem.courses.map(course => {
        const parts = [];
        if (course.type) parts.push(course.type);
        const creditValue = String(course.credit || '').split('-')[0].trim();
        if (creditValue) parts.push(creditValue + '학점');
        const meta = parts.join(' · ');
        
        // 과목명에서 영어명을 제외한 한글 이름만 추출
        const koreanName = (course.name || '').split('(')[0].trim();
        
        return `
          <div class="curriculum-course">
            <div class="curriculum-course-name">${koreanName}</div>
            <div class="curriculum-course-actions">
              <div class="curriculum-course-meta">${meta}</div>
              <button type="button" 
                      class="curriculum-search-btn" 
                      data-subject="${koreanName}">검색</button>
            </div>
          </div>
        `;
      }).join('');
    }
    html += `</div>`;
  });
  html += `</div></div>`;
  resultEl.innerHTML = html;
}

function fetchCurriculumPlan(department) {
  if (!department) {
    curriculumState.grouped = [];
    curriculumState.catalogYear = '';
    curriculumState.activeYear = '';
    curriculumState.currentDept = '';
    resetCurriculumView(curriculumState.defaultMessage || '학과를 선택해 커리큘럼을 확인하세요.');
    return;
  }
  curriculumState.grouped = [];
  curriculumState.activeYear = '';
  curriculumState.catalogYear = '';
  curriculumState.currentDept = department;
  resetCurriculumView(`${department} 커리큘럼을 불러오는 중입니다...`);
  fetch(`/curriculum?department=${encodeURIComponent(department)}`)
    .then(r => r.json())
    .then(data => {
      curriculumState.catalogYear = data.catalog_year || '';
      curriculumState.grouped = normalizeCurriculumPlan(data.plan || []);
      if (!curriculumState.grouped.length) {
        resetCurriculumView('선택한 학과의 커리큘럼 데이터가 없습니다.');
        return;
      }
      renderCurriculumYearButtons(curriculumState.grouped);
      showCurriculumYear(curriculumState.grouped[0].yearKey);
    })
    .catch(() => {
      resetCurriculumView('커리큘럼을 불러오지 못했습니다. 잠시 후 다시 시도하세요.');
    });
}

function initCurriculumSection() {
  const deptSelect = document.getElementById('curriculum-dept');
  if (!deptSelect) {
    return;
  }
  const hasOptions = deptSelect.options.length > 1;
  curriculumState.defaultMessage = hasOptions
    ? '학과를 선택해 커리큘럼을 확인하세요.'
    : '커리큘럼 데이터가 아직 준비되지 않았습니다.';
  resetCurriculumView(curriculumState.defaultMessage);
  deptSelect.addEventListener('change', () => {
    fetchCurriculumPlan(deptSelect.value);
  });
  if (deptSelect.value) {
    fetchCurriculumPlan(deptSelect.value);
  }
}

// [수정된 함수] 검색 결과를 10개씩 나눠서 렌더링하고 더보기 버튼을 처리합니다.
function renderPaginatedResults() {
  const resultsEl = document.getElementById('results');
  const moreContainer = document.getElementById('more-results-container');
  const totalCount = allSearchResults.length;
  
  if (!totalCount) {
    resultsEl.innerHTML = `<div class="empty-state">조건에 맞는 과목이 없습니다.</div>`;
    moreContainer.innerHTML = '';
    return;
  }
  
  // 새로 로드할 시작점과 끝점 계산
  const start = currentResultsCount;
  const end = Math.min(totalCount, start + RESULTS_LIMIT);
  
  const coursesToAppend = allSearchResults.slice(start, end);
  
  // 결과를 추가 (덮어쓰기가 아님)
  if (start === 0) {
      // 첫 로드인 경우 (전체 덮어쓰기)
      resultsEl.innerHTML = coursesToAppend.map(buildCourseCard).join('');
  } else {
      // '더보기'를 눌러 추가하는 경우
      resultsEl.insertAdjacentHTML('beforeend', coursesToAppend.map(buildCourseCard).join(''));
  }

  // 로드된 항목 수 업데이트
  currentResultsCount = end;

  const remainingCount = totalCount - currentResultsCount;
  
  // '더보기' 버튼 처리
  if (remainingCount > 0) {
    // 다음 로드될 항목 수 계산 (10개 또는 남은 항목 중 더 작은 수)
    const nextLoadCount = Math.min(remainingCount, RESULTS_LIMIT);

    moreContainer.innerHTML = `
      <button id="load-more-results" class="btn" style="width:100%; background:var(--color-primary-dark); color:var(--color-secondary); padding:10px; margin-top:15px; border:none; border-radius:8px;">
        더보기 (${nextLoadCount}개 추가 / 총 ${remainingCount}개 남음)
      </button>
    `;
    // '더보기' 버튼 클릭 시 다음 페이지 로드
    document.getElementById('load-more-results').onclick = () => {
      renderPaginatedResults();
    };
  } else {
    moreContainer.innerHTML = '';
  }
}


let searchTimer;
function runSearch(immediate = false) {
  if (searchTimer) {
    clearTimeout(searchTimer);
  }
  const exec = () => {
    // 검색 결과가 로드될 때 토글이 열려 있도록 설정
    const resultsToggleHeader = document.getElementById('results-toggle-header');
    if (resultsToggleHeader) {
        resultsToggleHeader.classList.remove('closed');
        resultsToggleHeader.classList.add('open');
        document.getElementById('results').style.display = '';
    }

    // 이전 검색 결과 및 더보기 버튼 초기화
    document.getElementById('results').innerHTML = '';
    document.getElementById('more-results-container').innerHTML = '';
    allSearchResults = []; 
    currentResultsCount = 0; // 새로 추가: 검색 시 로드 카운트 초기화

    const nlQuery = document.getElementById('main-search-input').value.trim();
    const department = document.getElementById('filter-department').value;
    const day = document.getElementById('filter-day').value;
    const hour = document.getElementById('filter-hour').value.trim();
    const midterm = document.getElementById('filter-midterm').value;
    const finalExam = document.getElementById('filter-final').value;
    const subject = document.getElementById('filter-subject').value.trim();
    const params = new URLSearchParams();
    if (nlQuery) params.append('nl_query', nlQuery);
    if (department) params.append('department', department);
    if (day) params.append('day', day);
    if (hour) params.append('hour', hour);
    if (midterm) params.append('midterm', midterm);
    if (finalExam) params.append('final', finalExam);
    if (subject) params.append('subject', subject);

    const queryString = params.toString();
    const url = queryString ? '/search?' + queryString : '/search';

    fetch(url)
      .then(r => r.json())
      .then(data => {
        const infoEl = document.getElementById('nl-info');
        const notes = data.notes || [];
        infoEl.innerHTML = notes.length ? notes.map(note => `<div class="note-item">${note}</div>`).join('') : '';

        // 모든 결과를 저장하고 페이지네이션 함수 호출
        allSearchResults = data.results || []; 
        renderPaginatedResults(); // isInitialLoad 파라미터 제거
      })
      .catch(() => {
        document.getElementById('results').innerHTML = `<div class="empty-state">검색 중 문제가 발생했습니다. 잠시 후 다시 시도해 주세요.</div>`;
        document.getElementById('more-results-container').innerHTML = '';
      });
  };
  if (immediate) {
    exec();
  } else {
    searchTimer = setTimeout(exec, 200);
  }
}

document.getElementById('filter-department').onchange = () => runSearch(true);
document.getElementById('filter-day').onchange = () => runSearch(true);
document.getElementById('filter-hour').oninput = () => runSearch();
document.getElementById('filter-midterm').onchange = () => runSearch(true);
document.getElementById('filter-final').onchange = () => runSearch(true);
document.getElementById('filter-subject').oninput = () => runSearch();
const clearSelectedBtn = document.getElementById('clear-selected-btn');
if (clearSelectedBtn) {
  clearSelectedBtn.onclick = function() {
    if (!selectedCourses.length) {
      return;
    }
    selectedCourses = [];
    renderSelectedList();
    updateTimetable();
  };
}
document.getElementById('claude-btn').onclick = function() {
  const userInput = document.getElementById('user-input').value;
  document.getElementById('claude-result').innerHTML = `<pre>AI가 추천을 생성하는 중입니다...</pre>`; // 로딩 표시
  fetch('/claude_reco', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ courses: selectedCourses, user_input: userInput })
  }).then(r => r.text()).then(txt => {
    document.getElementById('claude-result').innerHTML = `<pre>${txt}</pre>`;
  }).catch(() => {
    document.getElementById('claude-result').innerHTML = `<pre>추천 요청 중 오류가 발생했습니다.</pre>`;
  });
};

// [새로 추가된 부분] 검색 결과 토글 이벤트 리스너
const resultsToggleHeader = document.getElementById('results-toggle-header');
if (resultsToggleHeader) {
    resultsToggleHeader.addEventListener('click', function() {
        const content = document.getElementById('results');
        const moreBtnContainer = document.getElementById('more-results-container');
        if (this.classList.contains('open')) {
            this.classList.remove('open');
            this.classList.add('closed');
            content.style.display = 'none';
            moreBtnContainer.style.display = 'none'; // 버튼도 함께 숨김
        } else {
            this.classList.remove('closed');
            this.classList.add('open');
            content.style.display = '';
            // 내용이 표시될 때 더보기 버튼도 다시 표시 (단, 버튼 내용이 있어야 함)
            if (moreBtnContainer.innerHTML.trim() !== '') {
                moreBtnContainer.style.display = '';
            }
        }
    });
}
// [새로 추가된 부분 끝]

renderSelectedList();
updateSchedule();
initCurriculumSection();
runSearch(true);
// 기존 JavaScript 코드 끝
</script>
</body>
</html>
"""

CROSS_MAJOR_ELECTIVES = {
    "경영정보학과": {"국제경영학과", "경영학과"},
}

MONTH_ORDER = {f"{i}월": i for i in range(1, 13)}


def _clean_for_keyword(text):
    cleaned = text
    for stop in NATURAL_LANGUAGE_STOP_WORDS:
        cleaned = cleaned.replace(stop, " ")
    cleaned = re.sub(r"\d+학년", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def normalize_exam_value(value):
    return _normalize_text(value)


def parse_natural_language(query):
    filters = {}
    labels = {}
    keywords = []

    text = _normalize_text(query)
    if not text:
        return filters, labels, keywords

    normalized = text
    # Department detection
    for dept in departments:
        if dept and _contains_dept(normalized, dept):
            filters["department"] = dept
            labels["department"] = f"학과: {dept}"
            pattern = re.compile(rf"(?<![가-힣]){re.escape(dept)}(?![가-힣])")
            normalized = pattern.sub(" ", normalized)
            break

    # Professor detection
    for prof in professors:
        if prof and prof in normalized:
            filters["professor"] = prof
            labels["professor"] = f"담당 교수: {prof}"
            normalized = _strip_professor_reference(normalized, prof)
            break

    if "professor" not in filters:
        prof_match = re.search(rf"([가-힣]{{2,4}})\s*(?:{PROF_TITLE_SUFFIX})", normalized)
        if prof_match:
            candidate = prof_match.group(1)
            for prof in professors:
                if candidate in prof:
                    filters["professor"] = prof
                    labels["professor"] = f"담당 교수: {prof}"
                    normalized = normalized.replace(prof_match.group(0), " ")
                    break

    # Day detection
    day_symbol, normalized = _extract_day_token(normalized)
    if day_symbol:
        filters["day"] = day_symbol
        labels["day"] = f"요일: {day_symbol}"

    # Time detection (교시 우선)
    slot_match = re.search(r"([0-1]?\d)\s*(?:교시|교)", normalized)
    if slot_match:
        code = int(slot_match.group(1))
        filters["hour"] = f"{code:02d}교시"
        labels["hour"] = f"{code}교시"
        normalized = normalized.replace(slot_match.group(0), " ")
    else:
        time_match = re.search(r"((오전|오후)?\s*(\d{1,2}))\s*시", normalized)
        if time_match:
            raw = time_match.group(0)
            hour = int(time_match.group(3))
            meridiem = time_match.group(2)
            if meridiem:
                meridiem = meridiem.strip()
            if meridiem == "오후" and hour < 12:
                hour += 12
            filters["hour"] = f"{hour}시"
            labels["hour"] = f"{hour}시"
            normalized = normalized.replace(raw, " ")

    # Grade detection (1~6학년 등)
    grade_match = re.search(r"([1-6])\s*학년", normalized)
    if grade_match:
        gnum = grade_match.group(1)
        filters["grade"] = gnum
        labels["grade"] = f"학년: {gnum}학년"
        normalized = normalized.replace(grade_match.group(0), " ")

    # Course type detection (이수구분)
    course_type_map = {
        "전공필수": "전공필수",
        "전공 필수": "전공필수",
        "전공선택": "전공선택",
        "전공 선택": "전공선택",
        "교양필수": "교양필수",
        "교양 필수": "교양필수",
        "교양선택": "교양선택",
        "교양 선택": "교양선택",
        "자유선택": "자유선택",
        "자유 선택": "자유선택",
    }
    for key, value in course_type_map.items():
        if key in normalized:
            filters["course_type"] = value
            labels["course_type"] = f"이수구분: {value}"
            normalized = normalized.replace(key, " ")
            break

    # General 전공/교양 keyword when specific type not found
    if "course_type" not in filters:
        if "전공" in normalized:
            filters["course_type_contains"] = "전공"
            labels["course_type_contains"] = "이수구분 포함: 전공"
            normalized = normalized.replace("전공", " ")
        elif "교양" in normalized:
            filters["course_type_contains"] = "교양"
            labels["course_type_contains"] = "이수구분 포함: 교양"
            normalized = normalized.replace("교양", " ")

    # Exam preferences
    if re.search(r"중간고사\s*없", normalized):
        filters["midterm"] = "없음"
        labels["midterm"] = "중간고사: 없음"
    elif re.search(r"중간고사\s*(있|유)", normalized):
        filters["midterm"] = "있음"
        labels["midterm"] = "중간고사: 있음"

    if re.search(r"기말고사\s*없", normalized):
        filters["final"] = "없음"
        labels["final"] = "기말고사: 없음"
    elif re.search(r"기말고사\s*(있|유)", normalized):
        filters["final"] = "있음"
        labels["final"] = "기말고사: 있음"

    # Rating thresholds
    rating_patterns = [
        (r"평점(?:이)?\s*([0-9]+(?:\.[0-9]+)?)\s*점?\s*(?:이상|초과|보다 높은)", "min"),
        (r"평점(?:이)?\s*([0-9]+(?:\.[0-9]+)?)\s*점?\s*(?:이하|미만|보다 낮은)", "max"),
        (r"([0-9]+(?:\.[0-9]+)?)\s*점\s*(?:이상|초과|보다 높은)", "min"),
        (r"([0-9]+(?:\.[0-9]+)?)\s*점\s*(?:이하|미만|보다 낮은)", "max"),
    ]
    for pattern, bound_type in rating_patterns:
        match = re.search(pattern, normalized)
        if match:
            try:
                value = float(match.group(1))
            except ValueError:
                continue
            if bound_type == "min":
                filters["rating_min"] = value
                labels["rating_min"] = f"평점 ≥ {value}"
            else:
                filters["rating_max"] = value
                labels["rating_max"] = f"평점 ≤ {value}"
            normalized = normalized.replace(match.group(0), " ")
            break

    # Keyword extraction for remaining meaningful terms
    cleaned = _clean_for_keyword(normalized)
    if cleaned:
        candidate_keywords = [token for token in re.split(r"[\s,]+", cleaned) if token]
        keywords = [tok for tok in candidate_keywords if len(tok) > 1]
        if keywords:
            labels["keywords"] = "키워드: " + ", ".join(keywords)

    return filters, labels, keywords


def highlight_rating(rating):
    """평점에 따른 간단한 강조 표시."""
    try:
        score = float(rating)
    except (TypeError, ValueError):
        return rating
    if score >= 4:
        return f'<span class="rating-star" style="color:#FFD700;">★</span> {score}'
    if score >= 3:
        return f'<span class="rating-star" style="color:#A0A0A0;">☆</span> {score}'
    return f"{score}"


def _normalize_grade(value):
    text = str(value).strip()
    match = re.search(r"([1-6])", text)
    return match.group(1) if match else text


def filter_courses(params, keywords=None, candidate_indices=None):
    result = []
    student_dept = params.get('department')
    course_type_filter = params.get('course_type')
    course_type_contains = params.get('course_type_contains')
    norm_keywords = [
        _normalize_text(k) for k in (keywords or [])
        if _normalize_text(k)
    ]
    single_keyword = len(norm_keywords) == 1
    multi_keyword_threshold = max(1, math.ceil(len(norm_keywords) * 0.6)) if len(norm_keywords) > 1 else 0
    subject_filter_query = _normalize_text(params.get('subject')) if params.get('subject') else ""
    subject_filter_tokens = _subject_variant_tokens(subject_filter_query) if subject_filter_query else None
    single_keyword_value = norm_keywords[0] if single_keyword else ""
    single_keyword_tokens = _subject_variant_tokens(single_keyword_value) if single_keyword_value else None

    indices = candidate_indices if candidate_indices is not None else range(len(courses))
    for i in indices:
        c = courses[i].copy()
        c['_idx'] = i
        c['_treated_as_elective'] = False
        c['_treated_source'] = ""
        course_subject = c.get('과목명', '')
        course_tokens = COURSE_SUBJECT_TOKENS[i] if i < len(COURSE_SUBJECT_TOKENS) else frozenset()

        course_dept = str(c.get('학과', '')).strip()
        division = str(c.get('이수구분', '')).strip()

        if student_dept:
            allowed_cross = CROSS_MAJOR_ELECTIVES.get(student_dept, set())
            eligible_as_elective = (
                student_dept == "경영정보학과"
                and course_dept in allowed_cross
                and division == "전공필수"
                and (
                    course_type_filter == "전공선택"
                    or (course_type_filter is None and course_type_contains and "전공" in course_type_contains)
                )
            )
            if course_dept != student_dept and not eligible_as_elective:
                continue
            if eligible_as_elective:
                c['_treated_as_elective'] = True
                c['_treated_source'] = f"{course_dept} {division}"

        if params.get('day'):
            if not _has_day(str(c.get('수업시간', '')), params['day']):
                continue
        if subject_filter_query:
            if not _subject_matches(course_subject, subject_filter_query, course_tokens=course_tokens, query_tokens=subject_filter_tokens):
                continue
        if params.get('professor'):
            if params['professor'] not in str(c.get('담당교수', '')):
                continue
        midterm_value = normalize_exam_value(c.get('중간고사'))
        final_value = normalize_exam_value(c.get('기말고사'))
        if params.get('midterm'):
            desired = params['midterm']
            if desired == '있음' and midterm_value != '있음':
                continue
            if desired == '없음' and midterm_value == '있음':
                continue
        if params.get('final'):
            desired = params['final']
            if desired == '있음' and final_value != '있음':
                continue
            if desired == '없음' and final_value == '있음':
                continue
        if params.get('hour'):
            if not _matches_hour_filter(c.get('수업시간', ''), params['hour']):
                continue
        if params.get('grade'):
            want = _normalize_grade(params['grade'])
            have = _normalize_grade(c.get('학년', ''))
            if not want or not have or want != have:
                continue
        if course_type_filter:
            if course_type_filter != division:
                if not (c['_treated_as_elective'] and course_type_filter == '전공선택'):
                    continue
        if course_type_contains:
            if course_type_contains not in division:
                if not (c['_treated_as_elective'] and course_type_contains == '전공'):
                    continue
        rating = c.get('평점')
        if params.get('rating_min'):
            try:
                if rating is None or float(rating) < float(params['rating_min']):
                    continue
            except (TypeError, ValueError):
                continue
        if params.get('rating_max'):
            try:
                if rating is None or float(rating) > float(params['rating_max']):
                    continue
            except (TypeError, ValueError):
                continue

        if norm_keywords:
            if single_keyword:
                if not _subject_matches(course_subject, single_keyword_value, course_tokens=course_tokens, query_tokens=single_keyword_tokens):
                    continue
            else:
                haystack = _normalize_text(" ".join(str(c.get(field, "")) for field in [
                    '과목명', '담당교수', '이수구분', '학과', '수업시간', '강의계획서', '중간고사', '기말고사'
                ]))
                hits = sum(1 for kw in norm_keywords if kw in haystack)
                if hits < multi_keyword_threshold:
                    continue

        c['평점표시'] = highlight_rating(c.get('평점', 0))
        c['중간고사'] = midterm_value
        c['기말고사'] = final_value
        result.append(c)
    return result


def strict_filter_courses(params, keywords=None):
    results = []
    kw = [_normalize_text(k) for k in (keywords or []) if _normalize_text(k)]
    subject_filter_query = _normalize_text(params.get("subject")) if params.get("subject") else ""
    subject_filter_tokens = _subject_variant_tokens(subject_filter_query) if subject_filter_query else None
    single_keyword = len(kw) == 1
    single_keyword_value = kw[0] if single_keyword else ""
    single_keyword_tokens = _subject_variant_tokens(single_keyword_value) if single_keyword_value else None

    for i, raw in enumerate(courses):
        c = raw.copy()
        c["_idx"] = i
        course_subject = c.get("과목명", "")
        course_tokens = COURSE_SUBJECT_TOKENS[i] if i < len(COURSE_SUBJECT_TOKENS) else frozenset()

        if params.get("department"):
            if _normalize_text(c.get("학과")) != _normalize_text(params["department"]):
                continue

        if params.get("day"):
            if not _has_day(str(c.get("수업시간", "")), params["day"]):
                continue

        if params.get("hour"):
            if not _matches_hour_filter(c.get("수업시간", ""), params["hour"]):
                continue

        if subject_filter_query:
            if not _subject_matches(course_subject, subject_filter_query, course_tokens=course_tokens, query_tokens=subject_filter_tokens):
                continue
        if params.get("professor"):
            if not _contains_word_boundary(str(c.get("담당교수", "")), params["professor"]):
                continue

        if params.get("course_type"):
            if _normalize_text(c.get("이수구분")) != _normalize_text(params["course_type"]):
                continue
        if params.get("course_type_contains"):
            if not _contains_word_boundary(str(c.get("이수구분", "")), params["course_type_contains"]):
                continue

        mid = normalize_exam_value(c.get("중간고사"))
        fin = normalize_exam_value(c.get("기말고사"))
        if params.get("midterm"):
            if _normalize_text(params["midterm"]) != _normalize_text(mid):
                continue
        if params.get("final"):
            if _normalize_text(params["final"]) != _normalize_text(fin):
                continue

        rating = c.get("평점")
        try:
            r = float(rating) if rating is not None and str(rating).strip() != "" else None
        except ValueError:
            r = None
        if params.get("rating_min") is not None:
            try:
                if r is None or r < float(params["rating_min"]):
                    continue
            except ValueError:
                continue
        if params.get("rating_max") is not None:
            try:
                if r is None or r > float(params["rating_max"]):
                    continue
            except ValueError:
                continue

        if kw:
            if single_keyword:
                if not _subject_matches(course_subject, single_keyword_value, course_tokens=course_tokens, query_tokens=single_keyword_tokens):
                    continue
            else:
                fields = " ".join([
                    _normalize_text(c.get("과목명", "")),
                    _normalize_text(c.get("담당교수", "")),
                    _normalize_text(c.get("학과", "")),
                    _normalize_text(c.get("이수구분", "")),
                    _normalize_text(c.get("강의계획서", "")),
                    _normalize_text(c.get("수업시간", "")),
                    _normalize_text(c.get("중간고사", "")),
                    _normalize_text(c.get("기말고사", "")),
                ])
                tokens = {_normalize_text(tok) for tok in _word_re.findall(fields)}
                if not all(token in tokens for token in kw):
                    continue

        c['중간고사'] = mid
        c['기말고사'] = fin
        c['평점표시'] = highlight_rating(c.get('평점', 0))
        results.append(c)

    return results


def parse_time_code(code):
    return f"{CODE_TO_HOUR.get(code, '')}시"


# 시간표 HTML 생성 함수
def make_timetable_html(subjects_df):
    html = '''
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Gowun+Dodum&display=swap');
    .chalk-table {border-collapse:collapse; width:85%; margin:10px auto; background:#164e28; border-radius:13px; overflow:hidden;font-size: 16px;}
    .chalk-table th, .chalk-table td {padding:12px 5px; text-align:center; min-width:80px; font-family:'Gowun Dodum', 'Courier New', monospace !important; font-size:17px !important; border:1.5px solid #dfe8df;}
    .chalk-table th {background:#2f4f3b; color:#fffbe2;}
    .chalk-table td {background:#2f4f3b; color:#fffbe2; font-weight:bold;}
    .chalk-table .subject {color:#dfe8df; background:rgba(255,255,255,0.08); border-radius:7px; display:inline-block; margin-bottom:2px;font-size:13px;}
    .chalk-table .prof {color:#fffbe2; font-size:13px;}
    </style>
    '''
    html += '<table class="chalk-table">'
    html += '<tr><th>시간</th>' + ''.join([f'<th>{d}</th>' for d in ["월", "화", "수", "목", "금"]]) + '</tr>'
    for time_code in range(1, 15):
        row = f'<tr><td>{time_code}<br><span style="font-family:font-size:13px">{parse_time_code(time_code)}</span></td>'
        for day in range(5):
            cell = ''
            if not subjects_df.empty:
                for _, row_sub in subjects_df.iterrows():
                    for cell_day, cell_code in parse_timestr(str(row_sub['수업시간'])):
                        if cell_day == day and cell_code == time_code:
                            cell = f'<div class="subject">{row_sub["과목명"]}<br><span class="prof">{row_sub["담당교수"]}</span></div>'
            row += f'<td>{cell}</td>'
        row += '</tr>'
        html += row
    html += '</table>'
    return html


# Claude 프롬프트 생성 함수
def build_claude_prompt(user_input, timetable_df):
    subject_lines = []
    for _, row in timetable_df.iterrows():
        subject_lines.append(f"{row['과목명']}({row['담당교수']}, {row['수업시간']}, {row['평점']})")
    subject_block = "\n".join(subject_lines)
    prompt = f"""아래는 충북대 경영대학 시간표 챗봇입니다.
학생 입력 조건: \"{user_input}\"
학생이 선택한 시간표 과목 목록은 다음과 같습니다:
{subject_block}
너의 패르소나는 이제부터 충북대학교 마스코트 "우왕"이야
아래 3단계로 반드시 답변해 주세요.  
**각 단계 사이에 반드시 '---'만 단독 줄로 넣어 구분해 주세요.**

1. 각 과목별 정보를 아래 예시처럼 요약해 주세요.

예시:
## 추천 과목 시험 정보 요약

| 경영학사소프트웨어비즈니스 | 서보성 | 
|중간 : 있음 | 기말 : 없음 | 팀플 : 있음 | 과제 : 있음 | 별점 : 2.7 |

---
2. 각 과목별 추천 사유를 아래 예시처럼 과목명으로 시작해 2~3줄씩 써 주세요.

예시:
## 추천 사유 설명

- 경영학사소프트웨어비즈니스: 팀플과 과제가 많지만 실무 경험을 쌓기에 좋음. 기말고사가 없어 부담이 적음.
- 경영정보분석: 프로그래밍 경험이 적은 학생에게도 적합. 과제와 발표가 있으나 팀플은 없음.

---
3. 전체 시간표에 대한 총평을 2~3줄로 써 주세요.

예시:
## 총평

이 시간표는 과목별로 실무 중심의 프로젝트가 많아 경험을 쌓기에 좋습니다. 시험 부담이 적고, 다양한 평가 방식이 조화롭게 배치되어 있습니다.

---
**꼭 위의 단계와 구분선을 지키고, 답변이 잘리지 않게 끝까지 작성해 주세요.**
**답변이 길어질 경우, 반드시 모든 단계가 포함되도록 요약해서라도 끝까지 작성해 주세요.**
**화**

"""
    return prompt


# Claude API 호출 함수
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL = "claude-3-7-sonnet-20250219"
CLAUDE_API_VERSION = "2023-06-01"
CLAUDE_MAX_TOKENS = 1000
# CLAUDE_API_KEY는 환경 변수에서 로드됨


def get_claude_response(prompt):
    CLAUDE_API_KEY = "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"
    if not CLAUDE_API_KEY:
        return "API 키가 설정되지 않았습니다. CLAUDE_API_KEY 환경변수를 확인하세요."
    headers = {
        "x-api-key": CLAUDE_API_KEY,
        "anthropic-version": CLAUDE_API_VERSION,
        "Content-Type": "application/json"
    }
    data = {
        "model": CLAUDE_MODEL,
        "max_tokens": CLAUDE_MAX_TOKENS,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    response = requests.post(CLAUDE_API_URL, headers=headers, json=data)
    if response.status_code == 200:
        resp_json = response.json()
        return resp_json['content'][0]['text']
    else:
        return f"API 요청 실패: {response.status_code} / {response.text}"


@app.route('/', methods=['GET'])
def index():
    return render_template_string(
        HTML,
        departments=departments,
        curriculum_departments=CURRICULUM_PROGRAMS,
    )


@app.route('/search', methods=['GET'])
def search_api():
    params = request.args.to_dict()
    strict = params.pop('strict', 'false').lower() in ('1', 'true', 'yes')
    mode = params.pop('mode', 'auto').lower()
    query_param = params.pop('query', '').strip()
    nl_query = params.pop('nl_query', '').strip()
    search_text = nl_query or query_param

    manual_filters = {}
    for key, value in params.items():
        if not value:
            continue
        manual_filters[key] = _normalize_text(value) if isinstance(value, str) else value
    notes = []

    if strict:
        nl_filters, labels, kws = parse_natural_language(search_text)
        for k, v in nl_filters.items():
            manual_filters.setdefault(k, v)
        if nl_filters:
            applied = [labels.get(k, f"{k}={v}") for k, v in nl_filters.items()]
            notes.append("STRICT: 자연어 조건(AND) 적용 → " + ", ".join(applied))
        if kws:
            notes.append("STRICT: 키워드(AND) → " + ", ".join(kws))
        items = strict_filter_courses(manual_filters, keywords=kws)
        if _should_sort_by_rating(search_text) and items:
            items = _sort_results_by_rating(items)
            notes.append("요청 해석: 평점 상위 정렬 적용")
        notes.append(f"STRICT: 랭킹 미사용, {len(items)}건")
        return jsonify({"results": items, "notes": notes})

    if not search_text:
        items = filter_courses(manual_filters, keywords=None, candidate_indices=None)
        return jsonify({"results": items, "notes": ["검색어 없음: 수동 필터만 적용"]})

    if mode == 'auto':
        items, notes, diag = nlp_then_rules(search_text, manual_filters, notes)
        notes.append(f"diagnostic: {diag}")
        return jsonify({"results": items, "notes": notes})

    if mode == 'semantic':
        cand, score_list, best_score = get_sem_rank(search_text, k=10)
    elif mode == 'tfidf':
        cand, score_list, best_score = get_tfidf_rank(search_text, k=10)
    else:
        cand, score_list, best_score = get_bm25_rank(search_text, k=10)
        mode = 'bm25'

    items = filter_courses(manual_filters, keywords=None, candidate_indices=cand)
    if _should_sort_by_rating(search_text) and items:
        items = _sort_results_by_rating(items)
        notes.append("요청 해석: 평점 상위 정렬 적용")
    notes.append(f"수동 모드: {mode}, top score={best_score:.3f}, candidates={len(cand)}")
    return jsonify({"results": items, "notes": notes})


@app.route('/course_by_idx', methods=['GET'])
def course_by_idx():
    idx = int(request.args.get('idx', 0))
    return jsonify(courses[idx])


@app.route('/timetable', methods=['POST'])
def timetable_api():
    data = request.get_json()
    selected = data.get('courses', [])
    if selected:
        df = pd.DataFrame(selected)
    else:
        df = pd.DataFrame(columns=['과목명', '담당교수', '수업시간'])
    return make_timetable_html(df)


@app.route('/claude_reco', methods=['POST'])
def claude_reco_api():
    data = request.get_json()
    selected = data.get('courses', [])
    user_input = data.get('user_input', '')
    if not selected:
        return "선택된 과목이 없습니다."
    timetable_df = pd.DataFrame(selected)
    prompt = build_claude_prompt(user_input, timetable_df)
    return get_claude_response(prompt)


@app.route('/schedule', methods=['GET'])
def schedule_api():
    return jsonify(schedule)

def parse_schedule_date(date_str):
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except (TypeError, ValueError):
        return None


def format_schedule_range(start_str, end_str):
    start_dt = parse_schedule_date(start_str)
    end_dt = parse_schedule_date(end_str)
    if start_dt and end_dt:
        if start_dt.date() == end_dt.date():
            return start_dt.strftime("%Y.%m.%d")
        return f"{start_dt.strftime('%Y.%m.%d')} ~ {end_dt.strftime('%Y.%m.%d')}"
    if start_dt:
        return start_dt.strftime("%Y.%m.%d")
    if end_dt:
        return end_dt.strftime("%Y.%m.%d")
    return "일정 미정"


def build_schedule_calendar(schedule_data):
    events_by_month = {}
    for item in schedule_data:
        month_label = item.get('month') or ''
        events_by_month.setdefault(month_label, []).append(item)

    today = datetime.today().date()

    html = '''
    <style>
    .schedule-calendar { display:grid; gap:16px; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); font-family:'Noto Sans KR', Arial, sans-serif; }
    .schedule-month { background:#f7fafc; border:1px solid #e2e8f0; border-radius:12px; padding:16px; box-shadow:0 2px 6px rgba(15, 23, 42, 0.06); }
    .schedule-month h4 { margin:0 0 12px; font-size:1.1rem; color:#1e3a8a; display:flex; align-items:center; gap:8px; cursor: pointer; user-select: none; }
    .schedule-month h4::before { content: '+'; margin-right: 8px; font-weight: bold; color: #64748b; }
    .schedule-month.open h4::before { content: '−'; }
    .schedule-month h4 span { font-size:0.8rem; color:#64748b; }
    .event-list { list-style:none; margin:0; padding:0; display:flex; flex-direction:column; gap:10px; display: none; }
    .schedule-month.open .event-list { display: flex; }
    .event-item { background:#fff; border-radius:10px; padding:10px 12px; border:1px solid #e2e8f0; }
    .event-title { font-weight:600; color:#0f172a; font-size:0.95rem; margin-bottom:6px; }
    .event-period { font-size:0.82rem; color:#475569; display:flex; align-items:center; gap:6px; }
    .event-badge { display:inline-flex; align-items:center; justify-content:center; font-size:0.7rem; padding:2px 6px; border-radius:999px; }
    .badge-upcoming { background:#ecfdf5; color:#047857; border:1px solid #34d399; }
    .badge-ongoing { background:#eff6ff; color:#2563eb; border:1px solid #60a5fa; }
    .badge-complete { background:#fef2f2; color:#b91c1c; border:1px solid #fca5a5; }
    .event-empty { color:#94a3b8; font-size:0.85rem; text-align:center; padding:16px 0; }
    </style>
    <div class="schedule-calendar">
    '''

    month_items = sorted(
        events_by_month.items(),
        key=lambda item: (MONTH_ORDER.get(item[0], 99), item[0])
    )

    # JavaScript를 이용해 토글 기능을 구현할 것이므로,
    # 월별 컨테이너에 'data-month' 속성과 'schedule-toggle' 클래스를 추가합니다.
    for month_label, items in month_items:
        # H4 태그에 클릭 이벤트를 줄 수 있도록 클래스를 추가합니다.
        html += '<div class="schedule-month">'
        html += f"<h4 class='schedule-toggle' data-month='{month_label}'>{month_label or '기타'}<span>{len(items)}건</span></h4>"
        if not items:
            # event-list 클래스에 display: none이 적용될 것이므로 별도로 닫힘/열림 상태를 제어하지 않습니다.
            html += '<div class="event-empty">등록된 일정이 없습니다.</div>'
        else:
            # event-list는 기본적으로 숨겨지고, JS를 통해 부모 div에 .open 클래스가 토글되면 보이게 됩니다.
            html += '<ul class="event-list">'
            items.sort(key=lambda x: parse_schedule_date(x.get('start')) or datetime.max)
            for event in items:
                start_dt = parse_schedule_date(event.get('start'))
                end_dt = parse_schedule_date(event.get('end'))
                event_status = "badge-complete"
                status_label = "종료"
                if start_dt and end_dt:
                    if start_dt.date() <= today <= end_dt.date():
                        event_status = "badge-ongoing"
                        status_label = "진행중"
                    elif today < start_dt.date():
                        event_status = "badge-upcoming"
                        status_label = "예정"
                elif start_dt:
                    if today < start_dt.date():
                        event_status = "badge-upcoming"
                        status_label = "예정"
                    elif today == start_dt.date():
                        event_status = "badge-ongoing"
                        status_label = "당일"
                html += "<li class='event-item'>"
                html += f"<div class='event-title'>{event.get('event', '무제')}</div>"
                html += "<div class='event-period'>"
                html += f"<span class='event-badge {event_status}'>{status_label}</span>"
                html += f"{format_schedule_range(event.get('start'), event.get('end'))}"
                html += "</div>"
                html += "</li>"
            html += '</ul>'
        html += '</div>'
    html += '</div>'
    return html


@app.route('/schedule_html', methods=['GET'])
def schedule_html_api():
    return build_schedule_calendar(schedule)


@app.route('/curriculum', methods=['GET'])
def curriculum_api():
    dept = request.args.get('department')
    list_only = request.args.get('list', '').lower() in ('1', 'true', 'yes')
    year_filter = request.args.get('year', type=int)

    payload = {
        "departments": CURRICULUM_PROGRAMS,
    }

    if not dept or list_only:
        return jsonify(payload)

    curriculum = CURRICULUM_DATA.get(dept)
    if not curriculum:
        payload.update({
            "department": dept,
            "plan": [],
            "catalog_year": None,
            "error": "해당 학과 커리큘럼을 찾을 수 없습니다.",
        })
        return jsonify(payload)

    plan = curriculum.get('plan', [])
    if year_filter is not None:
        filtered = []
        for entry in plan:
            try:
                entry_year = int(entry.get('year', 0))
            except (TypeError, ValueError):
                entry_year = None
            if entry_year == year_filter:
                filtered.append(entry)
        plan_to_send = filtered
    else:
        plan_to_send = plan

    payload.update({
        "department": dept,
        "catalog_year": curriculum.get('catalog_year'),
        "plan": plan_to_send,
        "has_plan": bool(plan),
    })
    return jsonify(payload)

# ... (기존 app.route('/curriculum') 등의 함수들이 끝나는 부분) ...

@app.route('/<filename>')
def serve_image(filename):
    """
    현재 디렉토리에서 'CBNU_LOGO2.png'와 같은 이미지 파일을 서빙합니다.
    (Flask 앱 실행 파일과 이미지가 같은 폴더에 있어야 합니다.)
    """
    # 보안을 위해 파일명이 이미지 파일("CBNU_LOGO2.png") 또는 허용된 확장자인지 확인하는 것이 좋습니다.
    return send_from_directory('.', filename)

if __name__ == "__main__":
    app.run(debug=True)
