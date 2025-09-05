"""
a2c_qda_pipeline_lib.py
A2C-style inductive qualitative analysis library you can import and call from Python code.

- Provides functions to run each stage programmatically (Stage 1 guide, Stage 2 first-cycle, etc.)
- Exposes a single `run_full_pipeline(...)` convenience function
- Uses the exact OpenAI call signature you asked for:
      response = client.chat.completions.create(
          model="gpt-4.1",
          messages=[{"role": "user", "content": full_prompt}],
          n=1
      )

Install deps:
    pip install openai pandas numpy python-dotenv

Minimal usage:
    from a2c_qda_pipeline_lib import run_full_pipeline
    run_full_pipeline(
        hamlet_path="text_hamlet.csv",
        venice_path="text_venice.csv",
        model="gpt-4.1",
        batch_size=25,
        dotenv_path=".env",                 # optional
        outdir="outputs/my_run"            # optional; default: outputs/<timestamp>
    )

Advanced usage:
    import pandas as pd
    from a2c_qda_pipeline_lib import (
        make_client, read_and_longify, build_and_refine_guide, first_cycle_for_df,
        build_codebook_markdown, run_pattern_axial_thematic, quick_summaries
    )
"""

import os, re, json, time, datetime as dt
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

# --- dotenv + OpenAI client ---
from dotenv import load_dotenv
from openai import OpenAI

_CLIENT: Optional[OpenAI] = None

def make_client(dotenv_path: Optional[str] = None) -> OpenAI:
    """
    Load .env (if provided/exists), pull OPENAI_API_KEY, and return a cached OpenAI client.
    """
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    # Load environment
    load_dotenv(dotenv_path)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found. Put it in your env or a .env file.")
    # Optional: base URL support if user wants to route requests
    base_url = os.getenv("OPENAI_BASE_URL")
    _CLIENT = OpenAI(api_key=api_key) if not base_url else OpenAI(api_key=api_key, base_url=base_url)
    return _CLIENT

def chat_once(client: OpenAI, model: str, full_prompt: str) -> str:
    """
    Single call to the chat.completions API using the requested signature.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": full_prompt}],
        n=1
    )
    return response.choices[0].message.content

# --- General I/O helpers ---
def now_ts() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def write_text(path: str, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def write_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def save_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# --- Data prep ---
KOREAN_ID_COL = "참가자 ID를 입력해주세요."

def read_and_longify(path: str, encoding: str = "utf-8") -> pd.DataFrame:
    """
    Read the wide CSV (first col = participant ID; other columns are long-form answers),
    reshape to long, and split into coarse meaning units.
    Adjust `encoding` if your files are 'utf-8-sig' or 'cp949'.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path, encoding=encoding)
    if KOREAN_ID_COL not in df.columns:
        raise ValueError(f"Expected column '{KOREAN_ID_COL}' in {path}")
    id_col = KOREAN_ID_COL
    value_cols = [c for c in df.columns if c != id_col]
    long_df = df.melt(id_vars=[id_col], value_vars=value_cols, var_name="question", value_name="response")
    long_df.rename(columns={id_col: "participant_id"}, inplace=True)
    long_df["response"] = long_df["response"].astype(str).fillna("")
    long_df = long_df[long_df["response"].str.strip().astype(bool)].copy()

    rows = []
    for _, r in long_df.iterrows():
        txt = re.sub(r"\s+", " ", r["response"]).strip()
        if not txt:
            continue
        parts = re.split(r'(?<=[.!?…])\s+|(?<=[\u3002\uFF01\uFF1F])\s+', txt)
        parts = [p.strip() for p in parts if p.strip()]
        if not parts:
            parts = [txt]
        for j, u in enumerate(parts):
            rows.append({
                "participant_id": r["participant_id"],
                "question": r["question"],
                "segment_id": f'{r["participant_id"]}_{abs(hash(r["question"])) % 10000}_{j+1:02d}',
                "text_ko": u
            })
    return pd.DataFrame(rows).reset_index(drop=True)

# --- Stage 1: Analytical Guide Prompt ---

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
    cues = ["몰입", "흥미", "반복", "말을 바꾸", "고풍", "시대", "AI", "인공지능", "문법", "어색", "디자인", "미사여구", "오류", "스크롤"]
    df = df.copy()
    df["score"] = df["text_ko"].apply(lambda t: sum(c in t for c in cues))
    df["len"] = df["text_ko"].str.len()
    top = df.sort_values(["score","len"], ascending=[False, False]).head(k*2)
    seen, picked = set(), []
    for _, r in top.iterrows():
        key = r["text_ko"][:24]
        if key in seen:
            continue
        picked.append(r["text_ko"])
        seen.add(key)
        if len(picked) >= k:
            break
    while len(picked) < k and len(picked) < len(df):
        picked.append(df.iloc[len(picked)]["text_ko"])
    return picked[:k]

def prompt_build_guide(snippets: List[str]) -> str:
    header = (
        "SYSTEM: You are an expert qualitative methodologist.\n"
        "TASK: Create ONE Analytical Guide Prompt tailored to Korean survey snippets about LLM/NPC behavior, consistency, and immersion in story-driven games.\n\n"
    )
    body = f"""Provide exactly the following sections:
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
    return header + body

def prompt_eval_guide(guide_text: str) -> str:
    header = (
        "SYSTEM: You are a strict reviewer of analytical prompts for qualitative coding.\n"
        "TASK: Evaluate and, if needed, revise the guide. Return STRICT JSON only.\n\n"
    )
    body = f"""Return strict JSON only:
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
    return header + body

def build_and_refine_guide(client: OpenAI, model: str, df_all: pd.DataFrame, outdir: str) -> str:
    ensure_dir(outdir)
    prompts_dir = ensure_dir(os.path.join(outdir, "prompts_used"))
    snippets = pick_diverse_snippets(df_all, k=12)

    p_build = prompt_build_guide(snippets)
    write_text(os.path.join(prompts_dir, "stage1_build_prompt.txt"), p_build)
    guide_draft = chat_once(client, model, p_build)
    write_text(os.path.join(outdir, "guide_draft.txt"), guide_draft)

    p_eval1 = prompt_eval_guide(guide_draft)
    write_text(os.path.join(prompts_dir, "stage1_eval1_prompt.txt"), p_eval1)
    eval1_raw = chat_once(client, model, p_eval1)
    try:
        eval1 = json.loads(re.search(r'\{.*\}', eval1_raw, re.S).group(0))
    except Exception:
        eval1 = {"score": 3, "strengths": [], "gaps": ["Non-JSON"], "revise_prompt": guide_draft}
    write_json(os.path.join(outdir, "guide_eval1.json"), eval1)
    guide_final = eval1.get("revise_prompt", guide_draft)

    if int(eval1.get("score", 3)) < 5:
        p_eval2 = prompt_eval_guide(guide_final)
        write_text(os.path.join(prompts_dir, "stage1_eval2_prompt.txt"), p_eval2)
        eval2_raw = chat_once(client, model, p_eval2)
        try:
            eval2 = json.loads(re.search(r'\{.*\}', eval2_raw, re.S).group(0))
        except Exception:
            eval2 = {"score": 4, "strengths": [], "gaps": ["Non-JSON"], "revise_prompt": guide_final}
        write_json(os.path.join(outdir, "guide_eval2.json"), eval2)
        guide_final = eval2.get("revise_prompt", guide_final)

    write_text(os.path.join(outdir, "guide_final.txt"), guide_final)
    return guide_final

# --- Stage 2: First-cycle coding ---

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
    lines = ["participant_id\tsegment_id\ttext_ko"]
    for r in rows:
        t = re.sub(r"\s+", " ", r["text_ko"]).strip()
        lines.append(f'{r["participant_id"]}\t{r["segment_id"]}\t{t}')
    return "\n".join(lines)

def prompt_first_cycle(guide_final: str, batch_rows: List[Dict[str,Any]]) -> str:
    schema_str = json.dumps(FIRST_CYCLE_SCHEMA, ensure_ascii=False, indent=2)
    table = _tsv_for_rows(batch_rows)
    header = (
        "SYSTEM: You are a meticulous qualitative coder. Follow the Analytical Guide Prompt exactly.\n"
        "OUTPUT: JSON Lines (one JSON object per input row) matching the schema.\n\n"
    )
    body = f"""Analytical Guide Prompt (frozen; do not rewrite):
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
    return header + body

def _parse_json_objects_from_text(text: str) -> List[Dict[str,Any]]:
    recs: List[Dict[str,Any]] = []
    for line in text.splitlines():
        s = line.strip().strip("`")
        if not s:
            continue
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                recs.append(obj)
                continue
            if isinstance(obj, list):
                recs.extend(obj)
                continue
        except Exception:
            pass
        m = re.search(r'\{.*\}', s)
        if m:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict):
                    recs.append(obj)
                elif isinstance(obj, list):
                    recs.extend(obj)
            except Exception:
                pass
    return recs

def _coerce_first_cycle(rec: Dict[str,Any]) -> Dict[str,Any]:
    out = dict(rec)
    # Types
    try:
        out["intensity_1to3"] = int(out.get("intensity_1to3", 2))
    except Exception:
        out["intensity_1to3"] = 2
    for k in ["participant_id","segment_id","text_ko","emotion_primary","valence","immersion_level","consistency_tag","descriptive_code","evidence_quote","notes"]:
        if k not in out or out[k] is None:
            out[k] = ""
    return out

def first_cycle_for_df(client: OpenAI, model: str, df: pd.DataFrame, guide_final: str, outdir: str, batch_size: int = 25) -> List[Dict[str,Any]]:
    ensure_dir(outdir)
    ensure_dir(os.path.join(outdir, "prompts_used"))
    rows = df[["participant_id","segment_id","text_ko"]].to_dict(orient="records")
    all_recs: List[Dict[str,Any]] = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i+batch_size]
        p = prompt_first_cycle(guide_final, batch)
        write_text(os.path.join(outdir, "prompts_used", f"first_cycle_batch_{i//batch_size+1:02d}.txt"), p)
        out_text = chat_once(client, model, p)
        recs = _parse_json_objects_from_text(out_text)
        recs = [_coerce_first_cycle(r) for r in recs]
        all_recs.extend(recs)
        time.sleep(0.4)
    save_jsonl(os.path.join(outdir, "first_cycle.jsonl"), all_recs)
    return all_recs

# --- Codebook + Stage 4 ---

def build_codebook_markdown(client: OpenAI, model: str, first_cycle_records: List[Dict[str,Any]], outdir: str) -> str:
    ensure_dir(outdir)
    ensure_dir(os.path.join(outdir, "prompts_used"))
    p = (
        "SYSTEM: You are a codebook editor. Return a Markdown table only.\n\n"
        "From these first-cycle JSON objects, produce a compact codebook with columns:\n"
        "- code_name (descriptive_code)\n- definition (1–2 sentences)\n- when_to_use / when_not_to_use\n"
        "- positive/negative exemplars (Korean quotes)\n- common confusions\n\n"
        f"Data (sample up to 150):\n{json.dumps(first_cycle_records[:150], ensure_ascii=False)}\n"
    )
    write_text(os.path.join(outdir, "prompts_used", "codebook_prompt.txt"), p)
    md = chat_once(client, model, p)
    write_text(os.path.join(outdir, "codebook.md"), md)
    return md

def prompt_pattern(guide_final: str, first_cycle_records: List[Dict[str,Any]]) -> str:
    header = "SYSTEM: You do Pattern Coding. Return markdown only.\n\n"
    body = f"""Cluster the first-cycle codes into 4–8 pattern themes of the form:
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
    return header + body

def prompt_axial(first_cycle_records: List[Dict[str,Any]]) -> str:
    header = "SYSTEM: You perform axial coding. Return markdown only.\n\n"
    body = f"""Build a relational map using first-cycle fields (consistency_tag, emotion_primary, immersion_level, descriptive_code).

Return:
1) A list of edges: source → relation → target (concise labels).
2) 3–5 brief propositions (If [condition], then [effect], mediated by [X]).
3) 3 Korean quotes illustrating distinct links.

Data (sample up to 150):
{json.dumps(first_cycle_records[:150], ensure_ascii=False)}
"""
    return header + body

def prompt_thematic(first_cycle_records: List[Dict[str,Any]]) -> str:
    header = "SYSTEM: You write final thematic findings. Return markdown only.\n\n"
    body = f"""Produce 3–5 final themes with:
- theme title,
- 3–4 sentence analytic memo,
- 2–3 Korean exemplar quotes each,
- note any disconfirming evidence.

Keep it neutral and tied to the data.

Data (sample up to 150):
{json.dumps(first_cycle_records[:150], ensure_ascii=False)}
"""
    return header + body

def run_pattern_axial_thematic(client: OpenAI, model: str, guide_final: str, first_cycle_records: List[Dict[str,Any]], outdir: str) -> Tuple[str,str,str]:
    ensure_dir(outdir)
    ensure_dir(os.path.join(outdir, "prompts_used"))
    # Pattern
    p1 = prompt_pattern(guide_final, first_cycle_records)
    write_text(os.path.join(outdir, "prompts_used", "pattern_prompt.txt"), p1)
    patt_md = chat_once(client, model, p1)
    write_text(os.path.join(outdir, "pattern_coding.md"), patt_md)
    # Axial
    p2 = prompt_axial(first_cycle_records)
    write_text(os.path.join(outdir, "prompts_used", "axial_prompt.txt"), p2)
    axial_md = chat_once(client, model, p2)
    write_text(os.path.join(outdir, "axial_coding.md"), axial_md)
    # Thematic
    p3 = prompt_thematic(first_cycle_records)
    write_text(os.path.join(outdir, "prompts_used", "thematic_prompt.txt"), p3)
    them_md = chat_once(client, model, p3)
    write_text(os.path.join(outdir, "thematic_findings.md"), them_md)
    return patt_md, axial_md, them_md

# --- Summaries ---

def quick_summaries(first_cycle_records: List[Dict[str,Any]]) -> Dict[str,Any]:
    df = pd.DataFrame(first_cycle_records)
    out: Dict[str,Any] = {}
    if not df.empty and {"emotion_primary","immersion_level"} <= set(df.columns):
        tab = pd.crosstab(df["emotion_primary"], df["immersion_level"]).sort_index()
        out["emotion_x_immersion"] = tab.to_dict()
    top_low = {}
    if not df.empty and {"immersion_level","descriptive_code"} <= set(df.columns):
        low = df[df["immersion_level"]=="low"]
        if not low.empty:
            top_low = low["descriptive_code"].value_counts().head(5).to_dict()
    out["top5_descriptive_low"] = top_low
    vis_neg = []
    if not df.empty and {"consistency_tag","valence"} <= set(df.columns):
        mask = (df["consistency_tag"]=="visual_supports_immersion") & (df["valence"]=="negative")
        if mask.any():
            vis_neg = df.loc[mask, ["participant_id","segment_id","evidence_quote","text_ko"]].head(20).to_dict(orient="records")
    out["visual_supports_but_negative_examples"] = vis_neg
    return out

# --- Convenience: full pipeline from two file paths ---

def run_full_pipeline(
    hamlet_path: str,
    venice_path: str,
    model: str = "gpt-4.1",
    batch_size: int = 25,
    dotenv_path: Optional[str] = None,
    outdir: Optional[str] = None,
    encoding: str = "utf-8"
) -> Dict[str, Any]:
    """
    Run the full A2C pipeline programmatically from Python.

    Returns a dict with paths and some objects:
    {
      "outdir": <path>,
      "guide_final_path": <path>,
      "hamlet_first_cycle_path": <path>,
      "venice_first_cycle_path": <path>,
      "summary_path": <path>,
      "guide_final": <str>,
      "first_cycle_counts": {"hamlet": N, "venice": M}
    }
    """
    client = make_client(dotenv_path)
    # Output directory
    if outdir is None:
        outdir = os.path.join("outputs", now_ts())
    ensure_dir(outdir)

    # Load dataframes
    df_h = read_and_longify(hamlet_path, encoding=encoding)
    df_v = read_and_longify(venice_path, encoding=encoding)
    df_all = pd.concat([df_h, df_v], ignore_index=True)

    # Stage 1 (guide)
    guide_final = build_and_refine_guide(client, model, df_all, outdir)

    # Stage 2 (first-cycle)
    first_h = first_cycle_for_df(client, model, df_h, guide_final, os.path.join(outdir, "hamlet"), batch_size=batch_size)
    first_v = first_cycle_for_df(client, model, df_v, guide_final, os.path.join(outdir, "venice"), batch_size=batch_size)

    # Codebook + Stage 4 outputs
    build_codebook_markdown(client, model, first_h + first_v, outdir)
    run_pattern_axial_thematic(client, model, guide_final, first_h + first_v, outdir)

    # Summary
    summary = quick_summaries(first_h + first_v)
    summary_path = os.path.join(outdir, "summary.json")
    write_json(summary_path, summary)

    return {
        "outdir": outdir,
        "guide_final_path": os.path.join(outdir, "guide_final.txt"),
        "hamlet_first_cycle_path": os.path.join(outdir, "hamlet", "first_cycle.jsonl"),
        "venice_first_cycle_path": os.path.join(outdir, "venice", "first_cycle.jsonl"),
        "summary_path": summary_path,
        "guide_final": guide_final,
        "first_cycle_counts": {"hamlet": len(first_h), "venice": len(first_v)}
    }