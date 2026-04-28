"""
Word Twist - 고등 영어 어휘 문맥 문제 출제기
Gemini / GPT / Claude 통합, 한 지문에서 10+ 문제 생성
"""
import streamlit as st
import json
import os
import concurrent.futures

import google.generativeai as genai

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    import anthropic
except Exception:
    anthropic = None

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = ['https://www.googleapis.com/auth/drive.file']
SAVE_FILE = "saved_questions.json"
POS_ROTATION = ["verb", "adjective", "adverb", "conjunction"]
POS_KOR = {"verb": "동사", "adjective": "형용사", "adverb": "부사", "conjunction": "접속사"}

GEMINI_MODELS = ["gemini-flash-latest", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash-latest", "gemini-pro-latest"]
OPENAI_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
CLAUDE_MODELS = ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001", "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"]


def load_all_questions():
    if not os.path.exists(SAVE_FILE):
        return []
    try:
        with open(SAVE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_question(data):
    fd = load_all_questions()
    fd.append(data)
    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump(fd, f, ensure_ascii=False, indent=4)


def questions_for_text(text):
    t = text.strip()
    return [i for i in load_all_questions() if i.get("original_text", "").strip() == t]


def previous_targets(text):
    return [i.get("original_word", "") for i in questions_for_text(text) if i.get("original_word")]


def previous_pos(text):
    return [i.get("answer_pos", "") for i in questions_for_text(text) if i.get("answer_pos")]


def pick_focus_pos(text):
    history = previous_pos(text)
    if not history:
        return POS_ROTATION[0]
    counts = {p: history.count(p) for p in POS_ROTATION}
    mn = min(counts.values())
    cands = [p for p in POS_ROTATION if counts[p] == mn]
    for p in POS_ROTATION:
        if p in cands and history[-1] != p:
            return p
    return cands[0]


def call_llm(provider, model, prompt, api_keys):
    timeout_sec = 30  # 개별 API 호출 제한 시간 추가

    if provider == "gemini":
        api_key = api_keys.get("gemini", "").strip()
        if not api_key:
            raise ValueError("Gemini API 키가 입력되지 않았습니다.")
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(model).generate_content(
            prompt, 
            request_options={"timeout": timeout_sec}
        ).text
        
    if provider == "openai":
        if OpenAI is None:
            raise RuntimeError("pip install openai 필요")
        api_key = api_keys.get("openai", "").strip()
        if not api_key:
            raise ValueError("OpenAI API 키가 입력되지 않았습니다.")
        c = OpenAI(api_key=api_key)
        r = c.chat.completions.create(
            model=model, 
            messages=[{"role": "user", "content": prompt}], 
            temperature=0.9,
            timeout=timeout_sec
        )
        return r.choices[0].message.content
        
    if provider == "anthropic":
        if anthropic is None:
            raise RuntimeError("pip install anthropic 필요")
        api_key = api_keys.get("anthropic", "").strip()
        if not api_key:
            raise ValueError("Anthropic API 키가 입력되지 않았습니다.")
        c = anthropic.Anthropic(api_key=api_key)
        r = c.messages.create(
            model=model, 
            max_tokens=4096, 
            messages=[{"role": "user", "content": prompt}],
            timeout=timeout_sec
        )
        return r.content[0].text
        
    raise ValueError("unknown provider: " + str(provider))


def build_prompt(text, prev_targets, focus_pos):
    avoid_part = ""
    if prev_targets:
        avoid_part = "\n[중복 절대 금지] 아래 단어들은 이전 문제에서 정답으로 이미 사용됨. 이번 정답 타겟으로 절대 다시 쓰지 마라: " + ", ".join(prev_targets)

    pos_kor = POS_KOR[focus_pos]

    template = """너는 대한민국 고등학교 상위권~수능 수준의 영어 어휘 문제 전문가다.
다음 영어 지문으로 '문맥상 낱말의 쓰임이 적절하지 않은 것은?' 문제를 만든다.

[난이도]
대한민국 고2~고3 상위권 / 수능·모의고사 수준. 단순 어휘가 아닌 문맥·논리·뉘앙스로 판별 가능해야 한다.

[보기 선정 규칙]
1. 5개 보기는 반드시 형용사/동사/접속사/부사 중에서만 (명사·전치사·관사 등 금지).
2. 5개 품사가 가능한 한 다양하게 섞이도록 선택한다.
3. 5개 모두 문맥 흐름을 결정짓는 핵심 단어여야 한다.
4. 모든 보기 단어 앞에 (1)~(5) 번호 + HTML <u>단어</u> 태그.

[정답 변형 규칙]
5. 이번 문제의 정답 타겟 품사는 반드시 __POS_KOR__ (__POS_EN__) 다.
6. 정답 단어는 원문을 반의어/정반대 의미 단어로 교체한다.
7. 변형 후 문장은 문법은 자연스럽지만 문맥상 명백히 어긋나야 한다.
8. 나머지 4개 보기는 원문 그대로 유지.

[전체 흐름]
9. 변형된 1개를 제외하고 지문의 내용·논리·문장 순서·구조는 원문과 100% 동일.
__AVOID__

[출력]
마크다운 코드블록 없이 순수 JSON 한 덩어리만:
{
    "question_text": "(<u>(1) word</u> 형태 5개 포함된 영어 지문 전문)",
    "answer": 1,
    "original_word": "변형 전 원래 단어",
    "modified_word": "변형 후 들어간 단어",
    "answer_pos": "__POS_EN__",
    "explanation": "왜 부적절한지, 어떤 단어가 와야 하는지 한국어 2~3문장"
}

원문 텍스트:
\"\"\"
__TEXT__
\"\"\"
"""
    return (template
            .replace("__POS_KOR__", pos_kor)
            .replace("__POS_EN__", focus_pos)
            .replace("__AVOID__", avoid_part)
            .replace("__TEXT__", text))


def generate_one_raw(text, prev_targets, focus_pos, provider, model, api_keys):
    prompt = build_prompt(text, prev_targets, focus_pos)
    raw = call_llm(provider, model, prompt, api_keys)
    cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
    s = cleaned.find("{")
    e = cleaned.rfind("}")
    if s != -1 and e != -1:
        cleaned = cleaned[s:e + 1]
    return json.loads(cleaned)


def get_drive_service():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if os.path.exists('credentials.json'):
                flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
            else:
                st.error("credentials.json 파일이 필요합니다.")
                return None
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('drive', 'v3', credentials=creds)


def upload_to_drive(path):
    s = get_drive_service()
    if not s:
        return None
    media = MediaFileUpload(path, mimetype='application/json')
    f = s.files().create(body={'name': os.path.basename(path)}, media_body=media, fields='id').execute()
    return f.get('id')


st.set_page_config(page_title="Word Twist", page_icon="🌀", layout="wide")

CSS = """
<style>
.stApp {
    background:
      radial-gradient(1200px 600px at 10% -10%, rgba(168,85,247,0.18), transparent 60%),
      radial-gradient(1000px 500px at 110% 10%, rgba(236,72,153,0.15), transparent 60%),
      radial-gradient(900px 600px at 50% 120%, rgba(6,182,212,0.12), transparent 60%),
      linear-gradient(180deg, #0a0e1a 0%, #0b1020 100%);
    color: #e5e7eb;
}
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
h1, h2, h3, h4 { color: #e5e7eb !important; letter-spacing: -0.01em; }
.hero-title { font-size: 2.6rem; font-weight: 800; line-height: 1.1; background: linear-gradient(90deg, #a855f7, #ec4899, #06b6d4); -webkit-background-clip: text; background-clip: text; color: transparent; margin: 0.2rem 0 0.4rem 0; }
.hero-sub { color: #9ca3af; font-size: 0.95rem; margin-bottom: 1.4rem; }
.chip { display: inline-block; padding: 4px 10px; border-radius: 999px; font-size: 0.75rem; font-weight: 600; margin-right: 6px; margin-bottom: 4px; border: 1px solid rgba(255,255,255,0.08); background: rgba(255,255,255,0.03); color: #9ca3af; }
.chip-accent { background: linear-gradient(90deg, rgba(168,85,247,0.3), rgba(236,72,153,0.3)); color: #fff; border-color: transparent; }
.q-card { background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02)); border: 1px solid rgba(255,255,255,0.08); border-radius: 18px; padding: 1.4rem 1.6rem; line-height: 1.85; font-size: 1.02rem; box-shadow: 0 10px 40px rgba(0,0,0,0.3); }
.q-card u { text-decoration: none; border-bottom: 2px solid #a855f7; padding: 0 2px; color: #fff; font-weight: 600; }
.stTextArea textarea { background: rgba(255,255,255,0.03) !important; color: #e5e7eb !important; border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 12px !important; }
.stButton > button { background: linear-gradient(90deg, #a855f7, #ec4899); color: white; border: none; border-radius: 12px; padding: 0.55rem 1.1rem; font-weight: 600; box-shadow: 0 8px 22px rgba(168,85,247,0.25); transition: transform 0.15s ease, box-shadow 0.15s ease; }
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 12px 28px rgba(236,72,153,0.35); }
section[data-testid="stSidebar"] { background: rgba(10,14,26,0.7); border-right: 1px solid rgba(255,255,255,0.08); }
.metric-row { display: flex; gap: 14px; flex-wrap: wrap; }
.metric { flex: 1; min-width: 140px; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 0.9rem 1rem; }
.metric .label { color: #9ca3af; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; }
.metric .value { font-size: 1.5rem; font-weight: 700; margin-top: 0.2rem; }
.divider { height: 1px; background: rgba(255,255,255,0.08); margin: 1.4rem 0; }
.small-dim { color: #9ca3af; font-size: 0.85rem; }

.loading-overlay { position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: radial-gradient(800px 500px at 50% 50%, rgba(168,85,247,0.25), rgba(10,14,26,0.92) 70%); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); z-index: 99999; display: flex; align-items: center; justify-content: center; animation: fadeIn 0.25s ease; }
@keyframes fadeIn { from {opacity:0} to {opacity:1} }
.loading-box { text-align: center; padding: 38px 56px; border-radius: 22px; background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02)); border: 1px solid rgba(168,85,247,0.35); box-shadow: 0 30px 80px rgba(0,0,0,0.5), 0 0 60px rgba(168,85,247,0.25); }
.spinner { width: 84px; height: 84px; border-radius: 50%; margin: 0 auto 24px; background: conic-gradient(from 0deg, #a855f7, #ec4899, #06b6d4, #a855f7); -webkit-mask: radial-gradient(circle 32px at center, transparent 98%, #000 100%); mask: radial-gradient(circle 32px at center, transparent 98%, #000 100%); animation: spin 1.2s linear infinite; filter: drop-shadow(0 0 18px rgba(168,85,247,0.5)); }
@keyframes spin { to { transform: rotate(360deg); } }
.loading-title { color: #fff; font-size: 1.35rem; font-weight: 800; letter-spacing: -0.01em; margin-bottom: 6px; background: linear-gradient(90deg, #c084fc, #ec4899); -webkit-background-clip: text; background-clip: text; color: transparent; }
.loading-sub { color: #cbd5e1; font-size: 0.95rem; display: flex; align-items: center; gap: 6px; justify-content: center; }
.loading-dots::after { content: ''; display: inline-block; width: 1em; text-align: left; animation: dots 1.4s steps(4, end) infinite; }
@keyframes dots { 0%{content:''} 25%{content:'.'} 50%{content:'..'} 75%{content:'...'} 100%{content:''} }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🔌 모델 / API 키")
    provider = st.radio("Provider", ["gemini", "openai", "anthropic"],
        format_func=lambda x: {"gemini": "Gemini", "openai": "GPT", "anthropic": "Claude"}[x], horizontal=True)
    if provider == "gemini":
        model = st.selectbox("모델", GEMINI_MODELS, index=0)
        gemini_key = st.text_input("Gemini API Key", type="password", value=os.environ.get("GEMINI_API_KEY", ""))
        st.session_state["gemini_api_key"] = gemini_key
    elif provider == "openai":
        model = st.selectbox("모델", OPENAI_MODELS, index=0)
        openai_key = st.text_input("OpenAI API Key", type="password", value=os.environ.get("OPENAI_API_KEY", ""))
        st.session_state["openai_api_key"] = openai_key
    else:
        model = st.selectbox("모델", CLAUDE_MODELS, index=0)
        anthropic_key = st.text_input("Anthropic API Key", type="password", value=os.environ.get("ANTHROPIC_API_KEY", ""))
        st.session_state["anthropic_api_key"] = anthropic_key

    st.markdown("---")
    st.markdown("### 🎯 출제 옵션")
    auto_pos = st.checkbox("정답 품사 자동 로테이션", value=True)
    manual_pos = "verb"
    if not auto_pos:
        manual_pos = st.selectbox("정답 품사 고정", POS_ROTATION, format_func=lambda x: POS_KOR[x])
    batch_n = st.number_input("한 번에 만들 문제 수", 1, 10, 1)
    parallel_mode = st.checkbox("병렬 생성 (배치 시 빠름)", value=True,
        help="여러 개를 한 번에 만들 때 동시에 호출해서 5~10배 빠름. 단 같은 단어가 정답으로 겹칠 수 있어요(자동 제외).")

st.markdown("<div class='hero-title'>Word Twist</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-sub'>한 지문 · 무한히 비틀기 — Gemini · GPT · Claude 통합 어휘 문제 생성기</div>", unsafe_allow_html=True)

if "current_text" not in st.session_state:
    st.session_state.current_text = ""
if "last_results" not in st.session_state:
    st.session_state.last_results = []

user_input = st.text_area("영어 지문", height=220, value=st.session_state.current_text,
    placeholder="여기에 영어 지문을 붙여 넣으세요...")

if user_input.strip() != st.session_state.current_text.strip():
    st.session_state.current_text = user_input
    st.session_state.last_results = []

if user_input.strip():
    existing = questions_for_text(user_input)
    used_targets = [q.get("original_word", "?") for q in existing]
    next_pos = pick_focus_pos(user_input) if auto_pos else manual_pos
    pos_label = POS_KOR.get(next_pos, "?")

    metric_html = (
        "<div class='metric-row'>"
        "<div class='metric'><div class='label'>이 지문 누적</div><div class='value'>" + str(len(existing)) + "개</div></div>"
        "<div class='metric'><div class='label'>다음 정답 품사</div><div class='value'>" + pos_label + "</div></div>"
        "<div class='metric'><div class='label'>회피 단어 수</div><div class='value'>" + str(len(used_targets)) + "</div></div>"
        "</div>"
    )
    st.markdown(metric_html, unsafe_allow_html=True)

    if used_targets:
        st.markdown("<div class='small-dim' style='margin-top:0.6rem'>이전 정답 단어:</div>", unsafe_allow_html=True)
        chips = "".join("<span class='chip'>" + w + "</span>" for w in used_targets)
        st.markdown(chips, unsafe_allow_html=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    btn_one = st.button("🌀 새 문제 만들기")
with c2:
    btn_batch = st.button("⚡ " + str(int(batch_n)) + "개 한 번에")


def make_one(text):
    prev_t = previous_targets(text)
    focus = pick_focus_pos(text) if auto_pos else manual_pos
    api_keys = {
        "gemini": st.session_state.get("gemini_api_key", ""),
        "openai": st.session_state.get("openai_api_key", ""),
        "anthropic": st.session_state.get("anthropic_api_key", ""),
    }
    last_err = None
    for attempt in range(4):
        try:
            result = generate_one_raw(text, prev_t, focus, provider, model, api_keys)
        except Exception as e:
            last_err = e
            continue
        ow = (result.get("original_word") or "").strip().lower()
        if not ow:
            last_err = "original_word 누락"
            continue
        if ow in {t.lower() for t in prev_t}:
            last_err = "중복 타겟: " + ow
            continue
        result["original_text"] = text.strip()
        result["_provider"] = provider
        result["_model"] = model
        save_question(result)
        st.session_state.last_results.append(result)
        return result
    raise RuntimeError("생성 실패: " + str(last_err))


def render_loading(placeholder, current, total, provider_name, mode_label=""):
    sub = provider_name.upper() + " · " + str(current) + " / " + str(total)
    if mode_label:
        sub = provider_name.upper() + " · " + mode_label
    hint = "<div style='margin-top:14px; color:#94a3b8; font-size:0.78rem'>한 문제당 보통 8~15초 걸려요</div>"
    overlay_html = (
        "<div class='loading-overlay'><div class='loading-box'>"
        "<div class='spinner'></div>"
        "<div class='loading-title'>문제를 만들고 있어요</div>"
        "<div class='loading-sub'>" + sub + " <span class='loading-dots'></span></div>"
        + hint +
        "</div></div>"
    )
    placeholder.markdown(overlay_html, unsafe_allow_html=True)


def _make_one_threadsafe(text, prev_targets_snapshot, focus, provider_name, model_name, api_keys):
    return generate_one_raw(text, prev_targets_snapshot, focus, provider_name, model_name, api_keys)


if btn_one or btn_batch:
    if not user_input.strip():
        st.warning("먼저 영어 지문을 입력해주세요.")
    else:
        n = int(batch_n) if btn_batch else 1
        loader = st.empty()
        ok = 0
        errors = []

        api_keys_snapshot = {
            "gemini": st.session_state.get("gemini_api_key", ""),
            "openai": st.session_state.get("openai_api_key", ""),
            "anthropic": st.session_state.get("anthropic_api_key", ""),
        }

        if n > 1 and parallel_mode:
            render_loading(loader, 0, n, provider, mode_label=str(n) + "개 동시 생성 중")
            prev_snapshot = previous_targets(user_input)
            focus_default = pick_focus_pos(user_input) if auto_pos else manual_pos

            tasks_focus = []
            for i in range(n):
                if auto_pos:
                    tasks_focus.append(POS_ROTATION[(POS_ROTATION.index(focus_default) + i) % len(POS_ROTATION)])
                else:
                    tasks_focus.append(manual_pos)

            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(n, 8)) as ex:
                futures = {
                    ex.submit(_make_one_threadsafe, user_input, prev_snapshot, tasks_focus[i],
                              provider, model, api_keys_snapshot): i for i in range(n)
                }
                try:
                    # 병렬 처리 전체 블록에도 타임아웃(120초) 적용
                    for f in concurrent.futures.as_completed(futures, timeout=120):
                        try:
                            results.append(f.result())
                        except Exception as e:
                            errors.append(str(e))
                except concurrent.futures.TimeoutError:
                    errors.append("병렬 생성 중 시간 초과 발생")

            seen = {t.lower() for t in prev_snapshot}
            for r in results:
                ow = (r.get("original_word") or "").strip().lower()
                if not ow:
                    errors.append("original_word 누락")
                    continue
                if ow in seen:
                    errors.append("중복 정답으로 제외: " + ow)
                    continue
                seen.add(ow)
                r["original_text"] = user_input.strip()
                r["_provider"] = provider
                r["_model"] = model
                save_question(r)
                st.session_state.last_results.append(r)
                ok += 1
        else:
            for i in range(n):
                render_loading(loader, i + 1, n, provider)
                try:
                    make_one(user_input)
                    ok += 1
                except Exception as e:
                    errors.append(str(e))

        loader.empty()
        if ok:
            st.success(str(ok) + "개 문제 생성 완료 (saved_questions.json 저장)")
        if errors:
            with st.expander("⚠️ 실패 로그"):
                for e in errors:
                    st.write("- " + e)


def render_question_card(idx, r):
    prov = str(r.get("_provider", "?")).upper()
    pos = str(r.get("answer_pos", "?"))
    ans = str(r.get("answer", "?"))
    chips = (
        "<span class='chip chip-accent'>" + prov + "</span>"
        "<span class='chip'>" + pos + "</span>"
        "<span class='chip'>정답 " + ans + "번</span>"
    )
    st.markdown("<div style='margin-top:1rem'><b>문제 " + str(idx) + "</b> &nbsp; " + chips + "</div>", unsafe_allow_html=True)
    qtext = r.get("question_text", "")
    st.markdown("<div class='q-card'>" + qtext + "</div>", unsafe_allow_html=True)
    with st.expander("정답 · 해설"):
        st.markdown("**정답:** " + ans + "번")
        st.markdown("**원래 단어:** `" + str(r.get("original_word", "?")) + "` → **변형:** `" + str(r.get("modified_word", "?")) + "`")
        st.markdown("**품사:** " + pos)
        st.markdown("**해설:** " + str(r.get("explanation", "")))


if st.session_state.last_results:
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("### 🧾 이번 세션에서 만든 문제")
    for idx, r in enumerate(st.session_state.last_results, start=1):
        render_question_card(idx, r)

if user_input.strip():
    saved = questions_for_text(user_input)
    if saved:
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        with st.expander("📚 이 지문 누적 문제 (" + str(len(saved)) + "개)"):
            for idx, q in enumerate(saved, start=1):
                meta = (
                    "**#" + str(idx) + "** &nbsp; "
                    "정답 " + str(q.get("answer", "?")) + "번 · "
                    "`" + str(q.get("original_word", "?")) + "` → "
                    "`" + str(q.get("modified_word", "?")) + "` · "
                    "품사 " + str(q.get("answer_pos", "?")) + " · "
                    + str(q.get("_provider", "?")) + "/" + str(q.get("_model", "?"))
                )
                st.markdown(meta)
                st.markdown("<div class='q-card' style='margin-bottom:0.6rem'>" + q.get("question_text", "") + "</div>", unsafe_allow_html=True)
                st.caption(q.get("explanation", ""))

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
cdrv1, cdrv2 = st.columns([1, 3])
with cdrv1:
    drv = st.button("☁️ 드라이브 백업")
with cdrv2:
    st.markdown("<div class='small-dim' style='padding-top:0.5rem'>saved_questions.json 을 구글 드라이브로 업로드합니다.</div>", unsafe_allow_html=True)
if drv:
    if os.path.exists(SAVE_FILE):
        try:
            fid = upload_to_drive(SAVE_FILE)
            if fid:
                st.success("업로드 성공 (ID: " + fid + ")")
        except Exception as e:
            st.error("업로드 실패: " + str(e))
    else:
        st.warning("저장된 문제가 없습니다.")
