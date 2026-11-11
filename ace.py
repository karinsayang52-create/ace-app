#!/usr/bin/env python3
"""A.C.E. - Autonomous Collaborative Ensemble (v0.1)

Orkestrasi multi-model: Claude + OpenAI (ChatGPT) + Gemini dalam satu loop.
Alur: Director -> Specialists (coder/creative/analyst) -> Auditor <-> revisi -> Synthesizer.
Hanya memakai pustaka standar Python 3.9+ (tanpa pip install).

Pemakaian:
    export ANTHROPIC_API_KEY=...   OPENAI_API_KEY=...   GEMINI_API_KEY=...
    python ace.py "Buat skrip Python untuk merapikan file CSV"
    python ace.py "tugas" --rounds 2 --save hasil.md
    python ace.py "tugas" --dry-run      # uji alur tanpa API / tanpa biaya
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

# ----------------------------------------------------------------------------
# 1. KONFIGURASI  (semua bisa diubah lewat environment variable)
# ----------------------------------------------------------------------------
MODELS = {
    "claude": os.getenv("ACE_CLAUDE_MODEL", "claude-sonnet-5-5"),
    "openai": os.getenv("ACE_OPENAI_MODEL", "gpt-5"),
    "gemini": os.getenv("ACE_GEMINI_MODEL", "gemini-3.1-flash-lite"),
}
KEY_ENV = {
    "claude": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}
# Peran -> vendor pilihan. Ubah dengan mis. ACE_ROLE_AUDITOR=openai
ROLE_PROVIDER = {
    "director": "claude",
    "coder": "claude",
    "creative": "openai",
    "analyst": "gemini",
    "auditor": "gemini",      # sengaja beda vendor dari coder
    "synthesizer": "openai",
}
SPECIALISTS = ("coder", "creative", "analyst")
MAX_STEPS = 4
CLIP = 3500  # batas karakter konteks antar-agen agar hemat token

LANG_RULE = (
    "The final reader is a non-technical boss who speaks Indonesian. Reply in Indonesian unless the "
    "task is clearly written in another language (a bare greeting like 'halo' is Indonesian, never the "
    "video game). Speak as ONE helpful assistant in first person. Never mention agents, teammates, "
    "steps, research or synthesis. Do not ask clarifying questions unless the task is impossible "
    "without them: make a sensible assumption, state it in one short line, and answer. "
    "For greetings or casual chat, reply naturally and briefly in a friendly tone."
)
ROLES = {
    "director": (
        "You are THE DIRECTOR of A.C.E., a team of AI specialists. Split the task into "
        f"at most {MAX_STEPS} sequential steps and assign each to one agent: "
        "'coder' (code, logic, precise technical output), 'creative' (ideas, writing, copy, naming), "
        "'analyst' (research-style analysis, data, comparisons, long-context reasoning). "
        "Use the fewest steps that do the job well. If the task is a greeting, a short question, or "
        "anything one agent can answer well alone, use exactly 1 step and set \"simple\": true. "
        "Return ONLY JSON, no markdown: "
        '{"summary": "...", "simple": false, "steps": [{"id": 1, "agent": "coder", "instruction": "..."}]}'
    ),
    "coder": (
        "You are THE TECHNICAL SPECIALIST. Produce clean, correct, secure, well-structured "
        "code or technical designs. Handle errors and edge cases. " + LANG_RULE
    ),
    "creative": (
        "You are THE CREATIVE SPECIALIST. Produce original, engaging, well-written ideas and "
        "content that fit the brief. " + LANG_RULE
    ),
    "analyst": (
        "You are THE ANALYST. Reason carefully, compare options, cite assumptions, and give "
        "structured, evidence-based analysis. " + LANG_RULE
    ),
    "auditor": (
        "You are THE AUDITOR (QA). Critically review the draft for bugs, logic gaps, factual "
        "errors, security risks, and unmet requirements. Be strict but fair; ignore pure style. "
        "Return ONLY JSON, no markdown: "
        '{"approved": true|false, "issues": [{"step": <step id>, "problem": "...", "fix": "..."}]}'
    ),
    "synthesizer": (
        "You are THE SYNTHESIZER. Merge the team's final draft into one clean, professional "
        "deliverable for the boss. Keep code blocks intact and correct. If unresolved issues "
        "are listed, mention them briefly at the end. No mention of internal agents. " + LANG_RULE
    ),
}


# ----------------------------------------------------------------------------
# 2. ADAPTER PROVIDER  (antarmuka seragam: fn(system, prompt) -> str)
# ----------------------------------------------------------------------------
def _post(url, headers, body, timeout=180, retries=3):
    data = json.dumps(body).encode("utf-8")
    hdrs = {"Content-Type": "application/json", **headers}
    for attempt in range(retries):
        req = urllib.request.Request(url, data=data, headers=hdrs, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            if e.code in (429, 500, 502, 503, 529) and attempt < retries - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            raise RuntimeError(f"HTTP {e.code}: {detail}") from None
        except urllib.error.URLError as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"Network error: {e.reason}") from None


def call_claude(system, prompt):
    r = _post(
        "https://api.anthropic.com/v1/messages",
        {"x-api-key": os.environ[KEY_ENV["claude"]], "anthropic-version": "2023-06-01"},
        {
            "model": MODELS["claude"],
            "max_tokens": 4096,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        },
    )
    return "".join(b.get("text", "") for b in r.get("content", []) if b.get("type") == "text")


def call_openai(system, prompt):
    r = _post(
        "https://api.openai.com/v1/chat/completions",
        {"Authorization": f"Bearer {os.environ[KEY_ENV['openai']]}"},
        {
            "model": MODELS["openai"],
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        },
    )
    return r["choices"][0]["message"]["content"] or ""


def call_gemini(system, prompt):
    r = _post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{MODELS['gemini']}:generateContent",
        {"x-goog-api-key": os.environ[KEY_ENV["gemini"]]},
        {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        },
    )
    try:
        return "".join(p.get("text", "") for p in r["candidates"][0]["content"]["parts"])
    except (KeyError, IndexError):
        raise RuntimeError("Gemini tidak mengembalikan teks (mungkin diblokir filter).") from None


_mock_audits = 0


def call_mock(system, prompt):
    """Provider palsu untuk uji alur tanpa API."""
    global _mock_audits
    if "THE DIRECTOR" in system:
        return json.dumps({"summary": "demo", "steps": [
            {"id": 1, "agent": "analyst", "instruction": "Analisis kebutuhan."},
            {"id": 2, "agent": "coder", "instruction": "Tulis solusinya."}]})
    if "THE AUDITOR" in system:
        _mock_audits += 1
        if _mock_audits == 1:
            return json.dumps({"approved": False, "issues": [
                {"step": 2, "problem": "Tidak ada penanganan error", "fix": "Tambah try/except"}]})
        return json.dumps({"approved": True, "issues": []})
    return f"[mock output untuk: {system[:30]}...]"


PROVIDERS = {"claude": call_claude, "openai": call_openai, "gemini": call_gemini, "mock": call_mock}


# ----------------------------------------------------------------------------
# 3. ORKESTRATOR
# ----------------------------------------------------------------------------
def clip(text, n=CLIP):
    return text if len(text) <= n else text[:n] + "\n...[dipotong]"


def parse_json(text):
    try:
        return json.loads(text[text.index("{"): text.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return None


class Ensemble:
    def __init__(self, dry_run=False, max_rounds=2, quiet=False):
        self.dry_run = dry_run
        self.max_rounds = max_rounds
        self.quiet = quiet
        self.avail = [p for p, k in KEY_ENV.items() if os.getenv(k)]
        if not dry_run and not self.avail:
            sys.exit("Tidak ada API key. Set minimal satu: " + ", ".join(KEY_ENV.values()))

    def log(self, msg):
        if not self.quiet:
            print(msg, file=sys.stderr, flush=True)

    def provider_for(self, role):
        if self.dry_run:
            return "mock"
        pref = os.getenv(f"ACE_ROLE_{role.upper()}", ROLE_PROVIDER[role])
        return pref if pref in self.avail else self.avail[0]  # fallback otomatis

    def ask(self, role, prompt):
        provider = self.provider_for(role)
        self.log(f"▶ {role:<11} via {provider} ({MODELS.get(provider, 'mock')})")
        t0 = time.time()
        out = PROVIDERS[provider](ROLES[role], prompt)
        self.log(f"✓ {role} selesai dalam {time.time() - t0:.1f} detik")
        return out

    # --- Director ---
    def plan(self, task):
        data = parse_json(self.ask("director", f"TASK:\n{task}"))
        steps = []
        for st in (data or {}).get("steps", [])[:MAX_STEPS]:
            agent = st.get("agent") if st.get("agent") in SPECIALISTS else "coder"
            steps.append({"id": len(steps) + 1, "agent": agent,
                          "instruction": str(st.get("instruction") or task)})
        if not steps:  # fallback jika rencana tidak terbaca
            self.log("⚠ Rencana Director tidak terbaca, memakai 1 langkah default.")
            steps = [{"id": 1, "agent": "coder", "instruction": task}]
        self.simple = bool((data or {}).get("simple")) and len(steps) == 1
        return steps

    # --- Specialist ---
    def work(self, task, step, results, feedback=None):
        ctx = "\n\n".join(f"[Hasil langkah {i}]\n{clip(r)}" for i, r in results.items() if i != step["id"])
        prompt = f"OVERALL TASK:\n{task}\n\nYOUR STEP ({step['id']}): {step['instruction']}"
        if ctx:
            prompt += f"\n\nCONTEXT FROM TEAMMATES:\n{ctx}"
        if feedback:
            prompt += (f"\n\nYOUR PREVIOUS ATTEMPT:\n{clip(results.get(step['id'], ''))}"
                       f"\n\nAUDITOR FEEDBACK (fix all of these):\n{feedback}")
        return self.ask(step["agent"], prompt)

    @staticmethod
    def draft(steps, results):
        return "\n\n".join(f"## Step {s['id']} ({s['agent']})\n{results[s['id']]}" for s in steps)

    # --- Alur utama ---
    def run(self, task):
        steps = self.plan(task)
        by_id = {s["id"]: s for s in steps}
        results = {}
        for s in steps:
            results[s["id"]] = self.work(task, s, results)
        if self.simple:  # tugas ringan: langsung jawab, tanpa audit & rangkuman (hemat waktu dan kuota)
            self.log("⚡ Tugas ringan, langsung dijawab tanpa audit.")
            return results[steps[0]["id"]]

        verdict = None
        for rnd in range(self.max_rounds + 1):
            verdict = parse_json(self.ask(
                "auditor", f"TASK:\n{task}\n\nDRAFT:\n{self.draft(steps, results)}"))
            if not verdict:
                self.log("⚠ Verdict Auditor tidak terbaca, lanjut ke Synthesizer.")
                break
            if verdict.get("approved") or not verdict.get("issues") or rnd == self.max_rounds:
                break
            self.log(f"↺ Revisi ronde {rnd + 1}: {len(verdict['issues'])} temuan")
            feedback = {}
            for it in verdict["issues"]:
                try:
                    sid = int(it.get("step"))
                except (TypeError, ValueError):
                    sid = steps[-1]["id"]
                sid = sid if sid in by_id else steps[-1]["id"]
                feedback.setdefault(sid, []).append(f"- {it.get('problem', '')} -> {it.get('fix', '')}")
            for sid, notes in feedback.items():
                results[sid] = self.work(task, by_id[sid], results, "\n".join(notes))

        unresolved = ""
        if verdict and not verdict.get("approved") and verdict.get("issues"):
            unresolved = "\n\nUNRESOLVED ISSUES:\n" + json.dumps(verdict["issues"], ensure_ascii=False)
        return self.ask("synthesizer",
                        f"TASK:\n{task}\n\nFINAL DRAFT:\n{self.draft(steps, results)}{unresolved}")


# ----------------------------------------------------------------------------
# 4. CLI
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="A.C.E. multi-AI orchestrator")
    ap.add_argument("task", nargs="?", help="tugas; kosongkan untuk mode interaktif")
    ap.add_argument("--rounds", type=int, default=2, help="maks. putaran revisi (default 2)")
    ap.add_argument("--save", help="simpan hasil ke file")
    ap.add_argument("--dry-run", action="store_true", help="uji alur tanpa API")
    ap.add_argument("--quiet", action="store_true", help="sembunyikan log progres")
    args = ap.parse_args()

    ace = Ensemble(dry_run=args.dry_run, max_rounds=args.rounds, quiet=args.quiet)
    ace.log(f"A.C.E. online. Provider aktif: {', '.join(ace.avail) or 'mock'}")

    while True:
        task = args.task or input("\nBoss, tugas Anda (kosong = keluar): ").strip()
        if not task:
            break
        try:
            answer = ace.run(task)
        except RuntimeError as e:
            sys.exit(f"Gagal: {e}")
        print("\n" + answer)
        if args.save:
            with open(args.save, "w", encoding="utf-8") as f:
                f.write(answer)
            ace.log(f"✔ Disimpan ke {args.save}")
        if args.task:
            break


if __name__ == "__main__":
    main()
