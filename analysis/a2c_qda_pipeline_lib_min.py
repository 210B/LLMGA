import os, re, json, time, datetime as dt, csv
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

_CLIENT: Optional[OpenAI] = None

def make_client(dotenv_path: Optional[str] = None) -> OpenAI:
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    load_dotenv(dotenv_path)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found.")
    base_url = os.getenv("OPENAI_BASE_URL")
    _CLIENT = OpenAI(api_key=api_key) if not base_url else OpenAI(api_key=api_key, base_url=base_url)
    return _CLIENT

def chat_once(client: OpenAI, model: str, full_prompt: str) -> str:
    r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": full_prompt}], n=1)
    return r.choices[0].message.content

def now_ts() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True); return p

def write_text(path: str, text: str):
    with open(path, "w", encoding="utf-8") as f: f.write(text)

def write_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f: json.dump(obj, f, ensure_ascii=False, indent=2)

def save_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")

def _robust_read_csv(path: str, encoding: str = "utf-8") -> pd.DataFrame:
    encs = [encoding] + ([e for e in ["utf-8-sig","cp949"] if e != encoding])
    last = None
    for enc in encs:
        try:
            return pd.read_csv(path, encoding=enc, dtype=str)
        except Exception as e:
            last = e
        try:
            return pd.read_csv(path, encoding=enc, dtype=str, engine="python", sep=",",
                               quotechar='"', escapechar="\\", doublequote=True,
                               skipinitialspace=True, on_bad_lines="skip")
        except Exception as e:
            last = e
        try:
            with open(path, "r", encoding=enc, errors="replace") as f: sample = f.read(64000)
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            return pd.read_csv(path, encoding=enc, dtype=str, engine="python", sep=dialect.delimiter,
                               quotechar='"', escapechar="\\", doublequote=True,
                               skipinitialspace=True, on_bad_lines="skip")
        except Exception as e:
            last = e
    raise last or RuntimeError("CSV read failed.")

def canonicalize_id(x: Any) -> str:
    s = "" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)
    return re.sub(r"[^A-Za-z0-9]", "", s.strip()).upper()

def load_group_map_xlsx(path: str, id_col: str = "ID", group_col: str = "모델 구분", sheet_name: Optional[Any] = 0) -> Dict[str,str]:
    df = pd.read_excel(path, usecols=[id_col, group_col], sheet_name=sheet_name, dtype={id_col: str})
    df[id_col] = df[id_col].apply(canonicalize_id)
    df[group_col] = df[group_col].astype(str).str.strip()
    df = df[df[id_col].astype(bool)]
    return dict(zip(df[id_col], df[group_col]))

KOREAN_ID_COL = "참가자 ID를 입력해주세요."

def read_and_longify(path: str, encoding: str = "utf-8", group_map: Optional[Dict[str,str]] = None) -> pd.DataFrame:
    if not os.path.exists(path): raise FileNotFoundError(path)
    df = _robust_read_csv(path, encoding=encoding)
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    if KOREAN_ID_COL not in df.columns: raise ValueError(f"Missing '{KOREAN_ID_COL}'")
    id_col = KOREAN_ID_COL
    value_cols = [c for c in df.columns if c != id_col]
    long_df = df.melt(id_vars=[id_col], value_vars=value_cols, var_name="question", value_name="response")
    long_df.rename(columns={id_col: "participant_id"}, inplace=True)
    long_df["response"] = long_df["response"].astype(str).fillna("")
    long_df = long_df[long_df["response"].str.strip().astype(bool)].copy()
    rows = []
    for _, r in long_df.iterrows():
        txt = re.sub(r"\s+", " ", r["response"]).strip()
        parts = re.split(r'(?<=[.!?…])\s+|(?<=[\u3002\uFF01\uFF1F])\s+', txt) or [txt]
        parts = [p.strip() for p in parts if p.strip()] or [txt]
        g = group_map.get(canonicalize_id(r["participant_id"]), "") if group_map else ""
        for j, u in enumerate(parts):
            rows.append({"participant_id": r["participant_id"], "question": r["question"],
                         "segment_id": f'{r["participant_id"]}_{abs(hash(r["question"])) % 10000}_{j+1:02d}',
                         "text_ko": u, "group": g})
    return pd.DataFrame(rows).reset_index(drop=True)

GUIDE_STARTER = """Lenses:
(a) LLM/NPC consistency (repetition, contradictions, period-appropriate speech, historical plausibility),
(b) immersion signals (몰입/흥미 vs. 떨어짐; 'AI-ness'가 드러나는 순간),
(c) visual/design elements that support or undermine immersion.

Emotion families (primary): joy, interest, curiosity, awe, boredom, frustration, confusion, disappointment, mixed, neutral.
Intensity: 1=mild, 2=moderate, 3=strong.
Immersion: high, medium, low.
Consistency tags: dialogue_repetition, contradiction, period_appropriate_speech, historical_plausibility, visual_supports_immersion, flat_response, none.

Schema fields: participant_id, segment_id, text_ko, emotion_primary, valence, intensity_1to3, immersion_level, consistency_tag, descriptive_code, evidence_quote, notes.

Exemplar patterns (trigger → emotion → immersion):
- Contradiction → frustration/confusion → immersion↓
- Period-appropriate language → interest/joy → immersion↑
- Visual design supportive → positive affect → immersion↑
- Repetitive/templated answers → boredom → immersion↓
Always include a short Korean evidence quote.
"""

def pick_diverse_snippets(df: pd.DataFrame, k: int = 12) -> List[str]:
    cues = ["몰입","흥미","반복","말을 바꾸","고풍","시대","AI","인공지능","문법","어색","디자인","미사여구","오류","스크롤"]
    x = df.copy()
    x["score"] = x["text_ko"].apply(lambda t: sum(c in t for c in cues))
    x["len"] = x["text_ko"].str.len()
    top = x.sort_values(["score","len"], ascending=[False, False]).head(k*2)
    seen, out = set(), []
    for _, r in top.iterrows():
        key = r["text_ko"][:24]
        if key in seen: continue
        out.append(r["text_ko"]); seen.add(key)
        if len(out) >= k: break
    while len(out) < k and len(out) < len(x): out.append(x.iloc[len(out)]["text_ko"])
    return out[:k]

def prompt_build_guide(snippets: List[str]) -> str:
    h = "SYSTEM: You are an expert qualitative methodologist.\nTASK: Create ONE Analytical Guide Prompt tailored to Korean survey snippets about LLM/NPC behavior, consistency, and immersion in story-driven games.\n\n"
    b = f"""Provide exactly the following sections:
1) Core lens (what to pay attention to).
2) A small, stable set of emotion families.
3) Triggers → expected emotions → immersion shift patterns.
4) Consistency/Design checklists.
5) Output schema fields and allowed values (use the schema provided).
6) 3–5 short Korean exemplar quotes with how they'd be coded.

You MAY start from this starter but improve it to fit the data:
{GUIDE_STARTER}

Sample Korean snippets (use them to tailor the guide):
- """ + "\n- ".join(snippets)
    return h + b

def prompt_eval_guide(guide_text: str) -> str:
    h = "SYSTEM: You are a strict reviewer of analytical prompts for qualitative coding.\nTASK: Evaluate and, if needed, revise the guide. Return STRICT JSON only.\n\n"
    b = f"""Return strict JSON only:
{{
  "score": 1-5,
  "strengths": ["..."],
  "gaps": ["..."],
  "revise_prompt": "Rewrite a single improved Analytical Guide Prompt. If already perfect (5), repeat the same guide."
}}

Guide to evaluate:
<<<
{guide_text}
>>>
"""
    return h + b

def build_and_refine_guide(client: OpenAI, model: str, df_all: pd.DataFrame, outdir: str) -> str:
    ensure_dir(outdir); ensure_dir(os.path.join(outdir, "prompts_used"))
    snippets = pick_diverse_snippets(df_all, k=12)
    p_build = prompt_build_guide(snippets)
    write_text(os.path.join(outdir, "prompts_used", "stage1_build_prompt.txt"), p_build)
    draft = chat_once(client, model, p_build)
    write_text(os.path.join(outdir, "guide_draft.txt"), draft)
    p_eval1 = prompt_eval_guide(draft)
    write_text(os.path.join(outdir, "prompts_used", "stage1_eval1_prompt.txt"), p_eval1)
    e1_raw = chat_once(client, model, p_eval1)
    try:
        e1 = json.loads(re.search(r'\{.*\}', e1_raw, re.S).group(0))
    except Exception:
        e1 = {"score": 3, "strengths": [], "gaps": ["Non-JSON"], "revise_prompt": draft}
    write_json(os.path.join(outdir, "guide_eval1.json"), e1)
    guide = e1.get("revise_prompt", draft)
    if int(e1.get("score", 3)) < 5:
        p_eval2 = prompt_eval_guide(guide)
        write_text(os.path.join(outdir, "prompts_used", "stage1_eval2_prompt.txt"), p_eval2)
        e2_raw = chat_once(client, model, p_eval2)
        try:
            e2 = json.loads(re.search(r'\{.*\}', e2_raw, re.S).group(0))
        except Exception:
            e2 = {"score": 4, "strengths": [], "gaps": ["Non-JSON"], "revise_prompt": guide}
        write_json(os.path.join(outdir, "guide_eval2.json"), e2)
        guide = e2.get("revise_prompt", guide)
    write_text(os.path.join(outdir, "guide_final.txt"), guide)
    return guide

FIRST_CYCLE_SCHEMA = {
  "participant_id": "string",
  "segment_id": "string",
  "text_ko": "string",
  "emotion_primary": "joy|interest|curiosity|awe|boredom|frustration|confusion|disappointment|mixed|neutral",
  "valence": "positive|negative|mixed|neutral",
  "intensity_1to3": 1,
  "immersion_level": "high|medium|low",
  "consistency_tag": "dialogue_repetition|contradiction|period_appropriate_speech|historical_plausibility|visual_supports_immersion|flat_response|none",
  "descriptive_code": "short English noun-phrase about what the emotion is about",
  "evidence_quote": "short Korean fragment quoted from text_ko",
  "notes": "optional English note"
}

def _tsv_for_rows(rows: List[Dict[str,Any]]) -> str:
    ls = ["participant_id\tsegment_id\ttext_ko"]
    for r in rows:
        t = re.sub(r"\s+", " ", r["text_ko"]).strip()
        ls.append(f'{r["participant_id"]}\t{r["segment_id"]}\t{t}')
    return "\n".join(ls)

def prompt_first_cycle(guide_final: str, batch_rows: List[Dict[str,Any]]) -> str:
    schema_str = json.dumps(FIRST_CYCLE_SCHEMA, ensure_ascii=False, indent=2)
    table = _tsv_for_rows(batch_rows)
    h = "SYSTEM: You are a meticulous qualitative coder. Follow the Analytical Guide Prompt exactly.\nOUTPUT: JSON Lines (one JSON object per input row) matching the schema.\n\n"
    b = f"""Analytical Guide Prompt (frozen; do not rewrite):
<<<GUIDE>>>
{guide_final}
<<<END GUIDE>>>

Schema (allowed values/types):
{schema_str}

Rules:
- Choose ONE primary emotion; use "mixed" only if truly ambivalent.
- intensity_1to3: 1=mild, 2=moderate, 3=strong.
- immersion_level: infer from engagement vs detachment cues.
- consistency_tag: pick the best single tag; if none, "none".
- descriptive_code: concise English label of WHAT the emotion is about.
- evidence_quote: short Korean fragment directly quoted from text_ko.
- notes: optional English.
- Return ONLY JSONL, one object per input row.

Rows (TSV):
{table}
"""
    return h + b

def _parse_json_objects_from_text(text: str) -> List[Dict[str,Any]]:
    recs = []
    for line in text.splitlines():
        s = line.strip().strip("`")
        if not s: continue
        try:
            o = json.loads(s)
            if isinstance(o, dict): recs.append(o); continue
            if isinstance(o, list): recs.extend(o); continue
        except Exception: pass
        m = re.search(r'\{.*\}', s)
        if m:
            try:
                o = json.loads(m.group(0))
                if isinstance(o, dict): recs.append(o)
                elif isinstance(o, list): recs.extend(o)
            except Exception: pass
    return recs

def _coerce_first_cycle(rec: Dict[str,Any]) -> Dict[str,Any]:
    o = dict(rec)
    try: o["intensity_1to3"] = int(o.get("intensity_1to3", 2))
    except Exception: o["intensity_1to3"] = 2
    for k in ["participant_id","segment_id","text_ko","emotion_primary","valence","immersion_level","consistency_tag","descriptive_code","evidence_quote","notes"]:
        if k not in o or o[k] is None: o[k] = ""
    return o

def first_cycle_for_df(client: OpenAI, model: str, df: pd.DataFrame, guide_final: str, outdir: str, batch_size: int = 25) -> List[Dict[str,Any]]:
    ensure_dir(outdir); ensure_dir(os.path.join(outdir, "prompts_used"))
    rows = df[["participant_id","segment_id","text_ko"]].to_dict(orient="records")
    all_recs = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i+batch_size]
        p = prompt_first_cycle(guide_final, batch)
        write_text(os.path.join(outdir, "prompts_used", f"first_cycle_batch_{i//batch_size+1:02d}.txt"), p)
        out_text = chat_once(client, model, p)
        recs = [_coerce_first_cycle(r) for r in _parse_json_objects_from_text(out_text)]
        all_recs.extend(recs)
        time.sleep(0.4)
    save_jsonl(os.path.join(outdir, "first_cycle.jsonl"), all_recs)
    return all_recs

def build_codebook_markdown(client: OpenAI, model: str, first_cycle_records: List[Dict[str,Any]], outdir: str) -> str:
    ensure_dir(outdir); ensure_dir(os.path.join(outdir, "prompts_used"))
    p = ("SYSTEM: You are a codebook editor. Return a Markdown table only.\n\n"
         "From these first-cycle JSON objects, produce a compact codebook with columns:\n"
         "- code_name (descriptive_code)\n- definition (1–2 sentences)\n- when_to_use / when_not_to_use\n"
         "- positive/negative exemplars (Korean quotes)\n- common confusions\n\n"
         f"Data (sample up to 150):\n{json.dumps(first_cycle_records[:150], ensure_ascii=False)}\n")
    write_text(os.path.join(outdir, "prompts_used", "codebook_prompt.txt"), p)
    md = chat_once(client, model, p)
    write_text(os.path.join(outdir, "codebook.md"), md)
    return md

def prompt_pattern(guide_final: str, first_cycle_records: List[Dict[str,Any]]) -> str:
    h = "SYSTEM: You do Pattern Coding. Return markdown only.\n\n"
    b = f"""Cluster the first-cycle codes into 4–8 pattern themes of the form:
Trigger → Emotion → Immersion Shift

For each theme provide:
- name,
- short rationale,
- representative descriptive_code values,
- 2–3 Korean quotes (use evidence_quote if available),
- a 1–2 sentence explanation.

First-cycle sample (up to 150 items):
{json.dumps(first_cycle_records[:150], ensure_ascii=False)}

Guide (for context):
{guide_final}
"""
    return h + b

def prompt_axial(first_cycle_records: List[Dict[str,Any]]) -> str:
    h = "SYSTEM: You perform axial coding. Return markdown only.\n\n"
    b = f"""Build a relational map using first-cycle fields (consistency_tag, emotion_primary, immersion_level, descriptive_code).

Return:
1) A list of edges: source → relation → target (concise labels).
2) 3–5 brief propositions (If [condition], then [effect], mediated by [X]).
3) 3 Korean quotes illustrating distinct links.

Data (sample up to 150):
{json.dumps(first_cycle_records[:150], ensure_ascii=False)}
"""
    return h + b

def prompt_thematic(first_cycle_records: List[Dict[str,Any]]) -> str:
    h = "SYSTEM: You write final thematic findings. Return markdown only.\n\n"
    b = f"""Produce 3–5 final themes with:
- theme title,
- 3–4 sentence analytic memo,
- 2–3 Korean exemplar quotes each,
- note any disconfirming evidence.

Keep it neutral and tied to the data.

Data (sample up to 150):
{json.dumps(first_cycle_records[:150], ensure_ascii=False)}
"""
    return h + b

def run_pattern_axial_thematic(client: OpenAI, model: str, guide_final: str, first_cycle_records: List[Dict[str,Any]], outdir: str) -> Tuple[str,str,str]:
    ensure_dir(outdir); ensure_dir(os.path.join(outdir, "prompts_used"))
    p1 = prompt_pattern(guide_final, first_cycle_records)
    write_text(os.path.join(outdir, "prompts_used", "pattern_prompt.txt"), p1)
    patt_md = chat_once(client, model, p1)
    write_text(os.path.join(outdir, "pattern_coding.md"), patt_md)
    p2 = prompt_axial(first_cycle_records)
    write_text(os.path.join(outdir, "prompts_used", "axial_prompt.txt"), p2)
    axial_md = chat_once(client, model, p2)
    write_text(os.path.join(outdir, "axial_coding.md"), axial_md)
    p3 = prompt_thematic(first_cycle_records)
    write_text(os.path.join(outdir, "prompts_used", "thematic_prompt.txt"), p3)
    them_md = chat_once(client, model, p3)
    write_text(os.path.join(outdir, "thematic_findings.md"), them_md)
    return patt_md, axial_md, them_md

def quick_summaries(first_cycle_records: List[Dict[str,Any]], by_group: bool = False) -> Dict[str,Any]:
    df = pd.DataFrame(first_cycle_records); out = {}
    if df.empty: return out
    def base(dfx):
        d = {}
        if {"emotion_primary","immersion_level"} <= set(dfx.columns):
            d["emotion_x_immersion"] = pd.crosstab(dfx["emotion_primary"], dfx["immersion_level"]).sort_index().to_dict()
        if {"immersion_level","descriptive_code"} <= set(dfx.columns):
            low = dfx[dfx["immersion_level"]=="low"]
            d["top5_descriptive_low"] = low["descriptive_code"].value_counts().head(5).to_dict() if not low.empty else {}
        if {"consistency_tag","valence"} <= set(dfx.columns):
            mask = (dfx["consistency_tag"]=="visual_supports_immersion") & (dfx["valence"]=="negative")
            d["visual_supports_but_negative_examples"] = dfx.loc[mask, ["participant_id","segment_id","evidence_quote","text_ko"]].head(20).to_dict(orient="records")
        return d
    if by_group and "group" in df.columns:
        out["by_group"] = { (g or "ungrouped"): base(dfg) for g, dfg in df.groupby(df["group"].fillna("")) }
    else:
        out.update(base(df))
    return out

def run_full_pipeline(
    hamlet_path: str,
    venice_path: str,
    model: str = "gpt-4.1",
    batch_size: int = 25,
    dotenv_path: Optional[str] = None,
    outdir: Optional[str] = None,
    encoding: str = "utf-8",
    group_map_path: Optional[str] = None,
    group_map_id_col: str = "ID",
    group_map_group_col: str = "모델 구분",
    group_map_sheet_name: Optional[Any] = 0,
    by_group_summary: bool = False,
) -> Dict[str, Any]:
    client = make_client(dotenv_path)
    outdir = outdir or os.path.join("outputs", now_ts())
    ensure_dir(outdir)
    group_map = load_group_map_xlsx(group_map_path, group_map_id_col, group_map_group_col, group_map_sheet_name) if group_map_path else None
    df_h = read_and_longify(hamlet_path, encoding=encoding, group_map=group_map)
    df_v = read_and_longify(venice_path, encoding=encoding, group_map=group_map)
    df_all = pd.concat([df_h, df_v], ignore_index=True)
    guide_final = build_and_refine_guide(client, model, df_all, outdir)
    first_h = first_cycle_for_df(client, model, df_h, guide_final, os.path.join(outdir, "hamlet"), batch_size=batch_size)
    first_v = first_cycle_for_df(client, model, df_v, guide_final, os.path.join(outdir, "venice"), batch_size=batch_size)
    build_codebook_markdown(client, model, first_h + first_v, outdir)
    run_pattern_axial_thematic(client, model, guide_final, first_h + first_v, outdir)
    summary = quick_summaries(first_h + first_v, by_group=by_group_summary)
    summary_path = os.path.join(outdir, "summary.json"); write_json(summary_path, summary)
    return {
        "outdir": outdir,
        "guide_final_path": os.path.join(outdir, "guide_final.txt"),
        "hamlet_first_cycle_path": os.path.join(outdir, "hamlet", "first_cycle.jsonl"),
        "venice_first_cycle_path": os.path.join(outdir, "venice", "first_cycle.jsonl"),
        "summary_path": summary_path,
        "guide_final": guide_final,
        "first_cycle_counts": {"hamlet": len(first_h), "venice": len(first_v)}
    }
