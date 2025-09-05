"""
A2C-Style Inductive Qualitative Analysis — Full Pipeline (Organized)

This script organizes the A2C steps discussed previously into one file and uses the
exact OpenAI API calling pattern requested by the user:

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": full_prompt}],
        n=1
    )

You can change --model on the CLI to use a different model (e.g., 'gpt-4o', a GPT‑5 endpoint).
The prompts for each stage are fully inlined below.

Stages implemented
------------------
0) Data prep (reshape to long + split into meaning units)
1) Stage 1 (A2C): Build Analytical Guide Prompt from YOUR data
   1a. Draft guide from diverse snippets
   1b. Self-evaluate (score 1–5) and refine once more if needed
2) Stage 2 (First-cycle): Emotional + Descriptive coding (JSONL)
3) Codebook (compact, from first-cycle)
4) Second-cycle: Pattern coding → Axial coding → Thematic findings
5) Quick descriptive summaries / cross-tabs
6) (Optional) Coverage-style alignment check (conceptual approximation)
"""

import os, re, json, time, argparse, datetime as dt
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from openai import OpenAI
import json
from dotenv import load_dotenv

# ---------- OpenAI client (new SDK) ----------
try:
    from openai import OpenAI
except Exception as e:
    raise SystemExit("Install OpenAI SDK: pip install openai\n" + str(e))

def make_client(dotenv_path: Optional[str] = None) -> OpenAI:
    """
    Load .env (if present), read OPENAI_API_KEY, and return a cached OpenAI client.
    Usage:
        client = make_client()                   # loads .env from CWD or parents
        client = make_client(".env.local")       # explicit path
    """
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    # Load .env into process env (no-op if file not found)
    load_dotenv(dotenv_path)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not found. "
            "Add it to your environment or a .env file, e.g.\n"
            "OPENAI_API_KEY=sk-********************************"
        )

    # If you also use a custom endpoint, uncomment the base_url lines.
    # base_url = os.getenv("OPENAI_BASE_URL")
    # _CLIENT = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

    _CLIENT = OpenAI(api_key=api_key)
    return _CLIENT

# ---------- I/O helpers ----------
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

# ---------- Data prep ----------
KOREAN_ID_COL = "참가자 ID를 입력해주세요."

SAMPLE_HAMLET = """참가자 ID를 입력해주세요.,해당 점수를 주신 이유는 무엇인가요?,해당 점수를 주신 이유는 무엇인가요?.1,해당 점수를 주신 이유는 무엇인가요?.2,해당 점수를 주신 이유는 무엇인가요?.3,해당 점수를 주신 이유는 무엇인가요?.4,해당 점수를 주신 이유는 무엇인가요?.5,추가로 좋았던 점이나 불편했던 점을 적어주세요.
P07,대부분 묻는 말에 적절한 대답과 소설에서 나오는 내용과 비슷하게 대답을 했기 때문이다. ,"클로어디스와 거트루드가 죽은 햄릿에 대한 얘기를 했음에도 불구하고 표정이나 목소리의 크기, 말투 등이 드러나지 않는 텍스트 형식으로만 했기 때문에 하는 말에 대한 진실 유무를 판단하기는 어려웠다. 그러나 지나치게 차분하거나 논리적인 방식으로 행동하여 어색한 부분은 적었던 것 같다.","대부분 캐릭터의 행동이 일관성이 있었다. 그러나 거트루드가 소설에서는 그녀의 의도가 모호하게 묘사된다고 게임 레퍼런스 안내 자료에 적혀있는데, 게임 플레이 중에는 조금 더 옹호하는 느낌이 들었던 것 같다.",대부분의 캐릭터가 작품 설정에 맞는 말을 하였다고 느꼈다. 이로 인해 게임을 하는 데 어려움이 있지는 않았다.,실제로 있을 법한 요소들이 이야기 속에 그리고 캐릭터의 대사 속에 들어가서 실제로 있을 법하게 느껴졌다.,"캐릭터들의 대사가 비문인 경우는 있었지만, 대부분 묻는 말에 적절히 대답한다고 느꼈다. 이로 인해 실제 내가 햄릿이 되어 소설 속 등장인물들과 상호작용 하는 느낌이 더 컸던 것 같다.",딱히 없음.
P06,여러 캐릭터들과 대화를 해보았을 때 예상했던 말투+멘트가 나와서,오필리아가 슬픔에 빠져 익사할 것 같지가 않았다..........,몰입에 방해되는 것이 없었다,여러 캐릭터들과 대화해봤을 때 예상한 말들이 나왔기 때문에,뭔가 조금 망설이다가 멘트가 나왔으면 진짜같게 느껴졌을 거 같은데 오히려 멘트가 너무 빨리나와서 와 진짜같다 싶다가도 와 ai같다...싶은느낌이 들었다,오히려 여러캐릭터와 대화를 더 해보고 싶은 느낌이 들었기 때문에,스크롤은 베니스의 상인만의 문제가 아닌 것 같습니다.!! ㅜㅜ
"""

SAMPLE_VENICE = """참가자 ID를 입력해주세요.,해당 점수를 주신 이유는 무엇인가요?,해당 점수를 주신 이유는 무엇인가요?.1,해당 점수를 주신 이유는 무엇인가요?.2,해당 점수를 주신 이유는 무엇인가요?.3,해당 점수를 주신 이유는 무엇인가요?.4,해당 점수를 주신 이유는 무엇인가요?.5,추가로 좋았던 점이나 불편했던 점을 적어주세요
P07,"일부 캐릭터는 상황에 맞는 말은 하였으나, 일부 캐릭터는 물어보는 말에 대답하기보다는 핀트를 엇나가는 느낌을 받았기 때문에","샤일록은 작품 내에서는 최대한 조약을 바꾸지 않고 1파운드의 살을 도려내려고 했던 캐릭터인데, 내가 조약을 바꾸는 건 어떻냐고 물었을 때 그에 순응했기 때문이다.",대부분 작품 내 캐릭터와는 조금 다른 면이 있었다. 너리사는 내가 묻는 말에 대답하고 나의 행동에 따라오기보다는 조금 과장되거나 질문에 대한 핀트를 비껴나가는 답을 하면서 과장되게 대답하였다. 반면 샤일록은 안토니오에 대한 적대감을 내비치기보다는 논리적으로 내가 하는 말에 순응하는 모습에서 원작과의 차이점을 느꼈다. ,"캐릭터가 원작의 캐릭터와는 다른 면이 있다보니 게임을 플레이하는 데에 있어서 몰입되지 않는 부분이 컸다. 또한 내가 캐릭터랑 한 대화로 문제를 해결할 수 있을 것 같았는데, 후의 상황 선택지에서는 그러한 내용이 반영되지 않은 느낌을 받아 스토리를 끼워맞추는 느낌이 들어 몰입이 어려웠다.",비슷한 이유로 캐릭터가 작품의 모든 내용을 꿰고 이에 따라서 행동하는 느낌이 들지 않아서 내가 어떻게 행동하는 게 좋을지에 대한 고민으로 인해 게임의 몰입이 어려웠다.,캐릭터들이 하는 말에 너무 미사여구나 문법적으로 올바르지 않거나 어려운 단어가 많아 캐릭터의 말과 그 말의 의도를 이해하는데 시간이 오래 걸렸다. 이에 따라 게임에서 내가 어떻게 행동해야할지 고민하는 시간으로 인해 흥미가 감소되는 느낌을 받았다.,위에서 말했던 것과 비슷한 의미로 상황에 맞지 않는 말을 하거나 문법적으로 오류가 있는 문장들로 인해 스토리 진행에서 어려움을 느꼈던 점이 가장 불편했던 것 같습니다.
P06,대부분 다 맞는 말이라서 .. ?,이건 저도 잘 모르겠습니다 ..,번호로 나열해서 내용을 작성해주는데 좀 너무 갑자기 많은 글들이 들어오는 느낌이 들었다 (첫번째 파트) 근데 이걸 두 번이나 연속으로 봐서 ..,그냥 내가 하고 싶은 말을 했어서 딱히 방해되는 부분은 없었던 것 같음,아까도 말했듯이 파트1에서 조금 과장된 느낌 (너무 많이 글이 나오는 게),한 5번?6번?문항 넘어가고 나서는 딱히 할 말이 없어서 thanks (?)같은 말을 하고 끝냈다,내용이 길면 스크롤로 잘 안 내려가는 것 같아요 (근데 제 문제일 수도..)
"""

def ensure_sample(path: str, sample_text: str):
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(sample_text)

def read_and_longify(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    if KOREAN_ID_COL not in df.columns:
        raise ValueError(f"Input file missing '{KOREAN_ID_COL}' column: {path}")
    id_col = KOREAN_ID_COL
    value_cols = [c for c in df.columns if c != id_col]
    long_df = df.melt(id_vars=[id_col], value_vars=value_cols, var_name="question", value_name="response")
    long_df.rename(columns={id_col: "participant_id"}, inplace=True)
    long_df["response"] = long_df["response"].astype(str).fillna("")
    long_df = long_df[long_df["response"].str.strip().astype(bool)].copy()
    # Split into meaning units (coarse)
    rows = []
    for _, r in long_df.iterrows():
        txt = re.sub(r"\s+", " ", r["response"]).strip()
        if not txt:
            continue
        # Split on typical sentence enders; keep as fallback the whole text
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

# ---------- A2C Stage 1: Build Analytical Guide Prompt ----------

GUIDE_STARTER = """Lenses:
(a) LLM/NPC consistency (repetition, contradictions, period-appropriate speech, historical plausibility),
(b) immersion signals (e.g., 몰입, 흥미, or noticing 'AI-ness'),
(c) visual/design elements that support or undermine immersion.

Emotion families (primary only): joy, interest, curiosity, awe, boredom, frustration, confusion, disappointment, mixed, neutral.
Intensity: 1=mild, 2=moderate, 3=strong.
Immersion: high, medium, low.
Consistency tags: dialogue_repetition, contradiction, period_appropriate_speech, historical_plausibility, visual_supports_immersion, flat_response, none.

Schema fields: participant_id, segment_id, text_ko, emotion_primary, valence, intensity_1to3, immersion_level, consistency_tag, descriptive_code, evidence_quote, notes.

Exemplar triggers → emotions → immersion:
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

def build_guide_prompt(snippets: List[str]) -> str:
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

def eval_guide_prompt(guide_text: str) -> str:
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

# ---------- Stage 2: First-cycle coding ----------

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

def make_tsv(rows: List[Dict[str,Any]]) -> str:
    lines = ["participant_id\tsegment_id\ttext_ko"]
    for r in rows:
        t = re.sub(r"\s+", " ", r["text_ko"]).strip()
        lines.append(f'{r["participant_id"]}\t{r["segment_id"]}\t{t}')
    return "\n".join(lines)

def first_cycle_prompt(guide_final: str, batch_rows: List[Dict[str,Any]]) -> str:
    header = (
        "SYSTEM: You are a meticulous qualitative coder. Follow the Analytical Guide Prompt exactly.\n"
        "OUTPUT: JSON Lines (one JSON object per input row) matching the schema.\n\n"
    )
    schema_str = json.dumps(FIRST_CYCLE_SCHEMA, ensure_ascii=False, indent=2)
    rows_tsv = make_tsv(batch_rows)
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
{rows_tsv}
"""
    return header + body

# ---------- Stage 4: Pattern / Axial / Thematic ----------

def pattern_prompt(guide_final: str, first_cycle_records: List[Dict[str,Any]]) -> str:
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

def axial_prompt(first_cycle_records: List[Dict[str,Any]]) -> str:
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

def thematic_prompt(first_cycle_records: List[Dict[str,Any]]) -> str:
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

# ---------- Stage 6 (Optional): Coverage-style alignment ----------

def coverage_prompt(my_codes: List[str], ref_codes: List[str]) -> str:
    header = "SYSTEM: You align two code sets by semantic similarity. Return a Markdown table.\n\n"
    body = f"""Given Set A (my codes) and Set B (reference codes), pair each A_i to the best B_j with a similarity explanation and flag low-similarity A_i (potential new coverage).
Return columns: A_i | best_B_j | similarity_reason | length_reasonableness | decision(ok/new).

Set A:
{json.dumps(my_codes, ensure_ascii=False)}

Set B:
{json.dumps(ref_codes, ensure_ascii=False)}
"""
    return header + body

# ---------- JSON utilities ----------

def parse_json_objects_from_text(text: str) -> List[Dict[str,Any]]:
    """Extracts JSON objects from text (line-wise or fenced)."""
    records = []
    for line in text.splitlines():
        s = line.strip().strip("`")
        if not s:
            continue
        # try whole line
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                records.append(obj)
                continue
            if isinstance(obj, list):
                records.extend(obj)
                continue
        except Exception:
            pass
        # fallback: search object in line
        m = re.search(r'\{.*\}', s)
        if m:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict):
                    records.append(obj)
                elif isinstance(obj, list):
                    records.extend(obj)
            except Exception:
                pass
    return records

def coerce_first_cycle_types(rec: Dict[str,Any]) -> Dict[str,Any]:
    """Ensure minimal type hygiene for intensity and strings."""
    out = dict(rec)
    # intensity
    if "intensity_1to3" in out:
        try:
            out["intensity_1to3"] = int(out["intensity_1to3"])
        except Exception:
            out["intensity_1to3"] = 2
    # strings
    for k in ["participant_id","segment_id","text_ko","emotion_primary","valence","immersion_level","consistency_tag","descriptive_code","evidence_quote","notes"]:
        if k in out and out[k] is None:
            out[k] = ""
        if k not in out:
            out[k] = ""
    return out

# ---------- LLM call (exact signature) ----------

def chat_once(client, model: str, full_prompt: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": full_prompt}],
        n=1
    )
    return response.choices[0].message.content

# ---------- Pipeline runners ----------

def run_stage1(client, model: str, df_all: pd.DataFrame, outdir: str) -> str:
    ensure_dir(outdir)
    prompts_dir = ensure_dir(os.path.join(outdir, "prompts_used"))
    # 1a) Build draft guide
    snippets = pick_diverse_snippets(df_all, k=12)
    p_guide = build_guide_prompt(snippets)
    write_text(os.path.join(prompts_dir, "stage1_build_prompt.txt"), p_guide)
    guide_draft = chat_once(client, model, p_guide)
    write_text(os.path.join(outdir, "guide_draft.txt"), guide_draft)

    # 1b) Evaluate and refine (up to 2 passes)
    p_eval1 = eval_guide_prompt(guide_draft)
    write_text(os.path.join(prompts_dir, "stage1_eval1_prompt.txt"), p_eval1)
    eval1_raw = chat_once(client, model, p_eval1)
    try:
        eval1 = json.loads(re.search(r'\{.*\}', eval1_raw, re.S).group(0))
    except Exception:
        eval1 = {"score": 3, "strengths": [], "gaps": ["Non-JSON"], "revise_prompt": guide_draft}
    write_json(os.path.join(outdir, "guide_eval1.json"), eval1)
    guide_final = eval1.get("revise_prompt", guide_draft)

    if int(eval1.get("score", 3)) < 5:
        p_eval2 = eval_guide_prompt(guide_final)
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

def run_first_cycle(client, model: str, df: pd.DataFrame, guide_final: str, outdir: str, batch_size: int = 25) -> List[Dict[str,Any]]:
    ensure_dir(outdir)
    prompts_dir = ensure_dir(os.path.join(outdir, "prompts_used"))
    rows = df[["participant_id","segment_id","text_ko"]].to_dict(orient="records")
    all_recs: List[Dict[str,Any]] = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i+batch_size]
        p_fc = first_cycle_prompt(guide_final, batch)
        write_text(os.path.join(prompts_dir, f"first_cycle_batch_{i//batch_size+1:02d}.txt"), p_fc)
        out_text = chat_once(client, model, p_fc)
        # Parse JSON lines / objects
        recs = parse_json_objects_from_text(out_text)
        recs = [coerce_first_cycle_types(r) for r in recs]
        all_recs.extend(recs)
        time.sleep(0.4)
    save_jsonl(os.path.join(outdir, "first_cycle.jsonl"), all_recs)
    return all_recs

def run_codebook(client, model: str, first_cycle_records: List[Dict[str,Any]], outdir: str):
    ensure_dir(outdir)
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

def run_pattern_axial_thematic(client, model: str, guide_final: str, first_cycle_records: List[Dict[str,Any]], outdir: str):
    ensure_dir(outdir)
    # Pattern
    p1 = pattern_prompt(guide_final, first_cycle_records)
    write_text(os.path.join(outdir, "prompts_used", "pattern_prompt.txt"), p1)
    patt_md = chat_once(client, model, p1)
    write_text(os.path.join(outdir, "pattern_coding.md"), patt_md)
    # Axial
    p2 = axial_prompt(first_cycle_records)
    write_text(os.path.join(outdir, "prompts_used", "axial_prompt.txt"), p2)
    axial_md = chat_once(client, model, p2)
    write_text(os.path.join(outdir, "axial_coding.md"), axial_md)
    # Thematic
    p3 = thematic_prompt(first_cycle_records)
    write_text(os.path.join(outdir, "prompts_used", "thematic_prompt.txt"), p3)
    them_md = chat_once(client, model, p3)
    write_text(os.path.join(outdir, "thematic_findings.md"), them_md)

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

# ---------- Main ----------

def main():
    ag = argparse.ArgumentParser()
    ag.add_argument("--hamlet", default="text_hamlet.csv")
    ag.add_argument("--venice", default="text_venice.csv")
    ag.add_argument("--model", default="gpt-3.5-turbo")
    ag.add_argument("--batch", type=int, default=25)
    args = ag.parse_args()

    # Ensure sample data exists if missing (dry run convenience)
    ensure_sample(args.hamlet, SAMPLE_HAMLET)
    ensure_sample(args.venice, SAMPLE_VENICE)

    out_root = ensure_dir(os.path.join("outputs", now_ts()))
    prompts_root = ensure_dir(os.path.join(out_root, "prompts_used"))
    print("[OUT]", out_root)

    # Load & longify
    df_h = read_and_longify(args.hamlet)
    df_v = read_and_longify(args.venice)
    df_h.to_csv(os.path.join(out_root, "hamlet_long.csv"), index=False, encoding="utf-8")
    df_v.to_csv(os.path.join(out_root, "venice_long.csv"), index=False, encoding="utf-8")
    df_all = pd.concat([df_h, df_v], ignore_index=True)

    # Create OpenAI client
    client = make_client()

    # Stage 1 (A2C) — Build + refine guide
    print("[Stage 1] Building Analytical Guide Prompt ...")
    guide_final = run_stage1(client, args.model, df_all, out_root)
    print("[Stage 1] Done.")

    # Stage 2 — First-cycle coding (Hamlet & Venice)
    print("[Stage 2] Hamlet first-cycle coding ...")
    first_h = run_first_cycle(client, args.model, df_h, guide_final, os.path.join(out_root, "hamlet"), batch_size=args.batch)
    print(f"[Stage 2] Hamlet coded records: {len(first_h)}")

    print("[Stage 2] Venice first-cycle coding ...")
    first_v = run_first_cycle(client, args.model, df_v, guide_final, os.path.join(out_root, "venice"), batch_size=args.batch)
    print(f"[Stage 2] Venice coded records: {len(first_v)}")

    # Codebook (combined)
    print("[Codebook] Creating compact codebook ...")
    run_codebook(client, args.model, first_h + first_v, out_root)

    # Second-cycle: Pattern / Axial / Thematic
    print("[Stage 4] Pattern, Axial, Thematic ...")
    run_pattern_axial_thematic(client, args.model, guide_final, first_h + first_v, out_root)

    # Summaries
    print("[Summary] Writing quick summaries ...")
    summary = quick_summaries(first_h + first_v)
    write_json(os.path.join(out_root, "summary.json"), summary)

    print("[DONE] See:", out_root)

if __name__ == "__main__":
    main()
