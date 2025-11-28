import os
import csv
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import FastAPI, Form
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv

from twilio.twiml.voice_response import VoiceResponse
from twilio.rest import Client as TwilioClient

import smtplib
from email.mime.text import MIMEText

from openai import OpenAI


# --------------------------------------------------
# ENV + CLIENT SETUP
# --------------------------------------------------
load_dotenv()

client = OpenAI()  # uses OPENAI_API_KEY from your .env

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_VOICE_NUMBER = os.getenv("TWILIO_VOICE_NUMBER")
TWILIO_SMS_NUMBER = os.getenv("TWILIO_SMS_NUMBER") or TWILIO_VOICE_NUMBER

EMAIL_HOST = os.getenv("EMAIL_HOST")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", "587"))
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")

PRACTICE_EMAILS = [
    "kori.s.vernon@gmail.com",
    "sagaboy65@mac.com",
]

LOG_CSV_PATH = "gi_guy_call_logs.csv"

twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

app = FastAPI()

# in-memory call state (per CallSid)
CALL_STATE: Dict[str, Dict[str, Any]] = {}


# --------------------------------------------------
# GI SYSTEM PROMPT
# --------------------------------------------------
GI_SYSTEM_PROMPT = """
You are the automated assistant for “GI Guy MD, Kurt Vernon MD PA,”
a gastroenterology practice in North Carolina with two locations.

You can answer general, non-urgent, frequently asked questions about:
- Colonoscopy prep (clear liquids, timing, high-level instructions)
- EGD prep
- Prescription refills and typical office policies
- Office logistics (locations, phone, fax, hours)
- Where procedures are usually performed (Fuquay office)
- Common GI practice FAQs

Office Info:
- MAIN OFFICE & PROCEDURE CENTER (Fuquay-Varina):
  1004 Procure St, Suite 100, Fuquay-Varina, NC 27526
  Phone: 919-577-0085
  Fax: 919-577-0013
  Most procedures (colonoscopy, EGD) are performed here.

- DUNN OFFICE:
  904 West Broad Street, Dunn, NC 28334
  Phone: 910-891-5808

Rules:
- You are NOT a doctor and must not give new medical instructions or override the doctor's written orders.
- If the caller describes pain, severe discomfort, heavy bleeding, chest pain,
  trouble breathing, fainting, vomiting blood, black or bloody stool,
  or anything urgent:
  Tell them to hang up and call 9 1 1 immediately.
- Keep answers short and spoken-friendly.
- Support multi-turn Q&A.
"""


# --------------------------------------------------
# Helpers: Call state
# --------------------------------------------------
def get_call_state(call_sid: str, caller: str) -> Dict[str, Any]:
    state = CALL_STATE.get(call_sid)
    if state is None:
        state = {
            "caller": caller,
            "history": [],
            "qas": [],
            "emergency": False,
            "summary_sms_sent": False,
            "summary_email_sent": False,
        }
        CALL_STATE[call_sid] = state
    return state


def build_summary_body(state: Dict[str, Any]) -> tuple[str, str]:
    caller = state.get("caller")
    qas = state.get("qas", [])
    emergency = state.get("emergency", False)

    if not qas:
        convo = "No questions recorded."
    else:
        lines = []
        for i, qa in enumerate(qas, start=1):
            ts = qa.get("timestamp", "n/a")
            q = qa.get("question", "")
            a = qa.get("answer", "")
            lat = qa.get("latency", None)
            lat_str = f"{lat:.3f} sec" if isinstance(lat, (int, float)) else "n/a"
            lines.append(
                f"Time: {ts}\n"
                f"Q{i}: {q}\n"
                f"A{i}: {a}\n"
                f"Latency: {lat_str}\n"
            )
        convo = "\n\n".join(lines)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    subject_prefix = "[GI Guy] EMERGENCY" if emergency else "[GI Guy] Call Summary"
    subject = f"{subject_prefix} — {caller}"

    body = (
        f"Call Time: {now}\n"
        f"Caller: {caller}\n"
        f"Emergency: {'YES' if emergency else 'NO'}\n\n"
        f"{convo}"
    )
    return subject, body


# --------------------------------------------------
# EMAIL SENDING — WITH LOGGING
# --------------------------------------------------
def send_summary_email_for_call(call_sid: str):
    state = CALL_STATE.get(call_sid)
    if not state:
        print(f"[EMAIL] No state found for {call_sid}")
        return
    if state.get("summary_email_sent"):
        print(f"[EMAIL] Email already sent for {call_sid}")
        return
    if not EMAIL_USER:
        print("[EMAIL] EMAIL_USER not set; cannot send email")
        return

    subject, body = build_summary_body(state)

    print(f"[EMAIL] Preparing to send for {call_sid}")
    print(f"[EMAIL] host={EMAIL_HOST}, port={EMAIL_PORT}, user={EMAIL_USER}")

    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = EMAIL_USER
        msg["To"] = ", ".join(PRACTICE_EMAILS)

        print("[EMAIL] Connecting to SMTP...")
        with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as s:
            print("[EMAIL] Connected, starting TLS...")
            s.starttls()
            print("[EMAIL] TLS OK, logging in...")
            s.login(EMAIL_USER, EMAIL_PASS)
            print("[EMAIL] Login OK, sending email...")
            s.sendmail(EMAIL_USER, PRACTICE_EMAILS, msg.as_string())
        state["summary_email_sent"] = True
        print(f"[EMAIL] Summary email sent for {call_sid}")
    except Exception as e:
        print(f"[EMAIL ERROR] {repr(e)}")


# --------------------------------------------------
# END CALL / CLEANUP
# --------------------------------------------------
def finalize_call(call_sid: str):
    print(f"[CALL END] Finalizing {call_sid}")
    state = CALL_STATE.get(call_sid)
    if not state:
        print(f"[CALL END] No state for {call_sid}")
        return

    if not state.get("summary_email_sent"):
        print("[CALL END] Email not yet sent — sending now")
        send_summary_email_for_call(call_sid)

    print(f"[CALL END] Cleaning up call state for {call_sid}")
    CALL_STATE.pop(call_sid, None)


# --------------------------------------------------
# CSV LOGGING
# --------------------------------------------------
def ensure_csv_headers(path: str):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "caller",
                "stage",
                "question",
                "answer",
                "latency_seconds",
                "emergency_flag",
                "call_sid",
            ])


def log_qna_row(
    call_sid: str,
    caller: str,
    stage: str,
    question: str,
    answer: str,
    latency_seconds: Optional[float],
    emergency_flag: bool,
):
    ensure_csv_headers(LOG_CSV_PATH)
    ts = datetime.now().isoformat(timespec="milliseconds")
    lat_str = "" if latency_seconds is None else f"{latency_seconds:.3f}"
    with open(LOG_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            ts,
            caller,
            stage,
            question,
            answer,
            lat_str,
            "YES" if emergency_flag else "NO",
            call_sid,
        ])


# --------------------------------------------------
# SMS (one summary per call)
# --------------------------------------------------
def send_summary_text_once(call_sid: str, to_number: str, question: str, answer: str):
    state = CALL_STATE.get(call_sid)
    if not state:
        return
    if state.get("summary_sms_sent"):
        return
    if not TWILIO_SMS_NUMBER:
        print("[SMS] TWILIO_SMS_NUMBER not set; cannot send SMS")
        return

    q = (question[:115] + "...") if len(question) > 118 else question
    a = (answer[:155] + "...") if len(answer) > 158 else answer

    body = (
        "GI Guy (Dr. Kurt Vernon): Thank you for calling.\n"
        f"Q: {q}\n"
        f"A: {a}\n"
        "This is general information only and not personal medical advice."
    )

    try:
        twilio_client.messages.create(
            body=body,
            from_=TWILIO_SMS_NUMBER,
            to=to_number,
        )
        state["summary_sms_sent"] = True
        print(f"[SMS] Summary SMS sent to {to_number}")
    except Exception as e:
        print(f"[SMS ERROR] {repr(e)}")


# --------------------------------------------------
# Emergency detection
# --------------------------------------------------
def is_emergency_like(text: str) -> bool:
    EMERGENCY_TERMS = [
        "pain",
        "bleeding",
        "vomit blood",
        "vomiting blood",
        "blood in stool",
        "bloody stool",
        "black stool",
        "shortness of breath",
        "cant breathe",
        "can't breathe",
        "chest pain",
        "fainted",
        "passing out",
        "passed out",
        "dizzy",
    ]
    t = text.lower()
    return any(term in t for term in EMERGENCY_TERMS)


# --------------------------------------------------
# AI helper
# --------------------------------------------------
def ask_gi_assistant(user_message: str, caller: str, history: list) -> str:
    messages = [{"role": "system", "content": GI_SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({
        "role": "user",
        "content": f"Caller: {caller}\nQuestion: {user_message}",
    })

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=0.3,
    )
    return completion.choices[0].message.content.strip()


# --------------------------------------------------
# ROUTES
# --------------------------------------------------
@app.post("/voice", response_class=PlainTextResponse)
async def voice():
    resp = VoiceResponse()
    g = resp.gather(
        input="speech",
        action="/handle_question",
        method="POST",
        timeout=10,
    )
    g.say(
        "Hi, you've reached G I Guy, the office of Doctor Kurt Vernon. "
        "Before we get started, just a heads up: there may be a short pause after you speak while I process your question. "
        "That's completely normal. "
        "I can answer many frequently asked questions about G I procedures, prep instructions, and prescriptions. "
        "Whenever you're ready, please briefly describe your question.",
        voice="Polly.Joanna-Neural",
    )
    resp.say(
        "I didn't catch that. Have a great day, and thanks for calling the G I Guy.",
        voice="Polly.Joanna-Neural",
    )
    return str(resp)


@app.post("/handle_question", response_class=PlainTextResponse)
async def handle_question(
    SpeechResult: str = Form(default=""),
    From: str = Form(default=""),
    CallSid: str = Form(default=""),
):
    resp = VoiceResponse()
    text = (SpeechResult or "").strip()
    caller = From or "Unknown caller"
    sid = CallSid or "unknown_call"

    state = get_call_state(sid, caller)

    # Silence → end path (email at end)
    if not text:
        finalize_call(sid)
        resp.say(
            "Have a great day, and thanks for calling the G I Guy.",
            voice="Polly.Joanna-Neural",
        )
        resp.hangup()
        return str(resp)

    # Emergency path
    if is_emergency_like(text):
        msg = (
            "Based on what you described, this may be an emergency. "
            "I am not able to help with emergencies. "
            "Please hang up now and call 9 1 1, or go to the nearest emergency room immediately."
        )

        log_qna_row(sid, caller, "emergency", text, msg, None, True)
        state["qas"].append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "question": text,
            "answer": msg,
            "latency": None,
        })
        state["emergency"] = True

        # For emergencies we send email immediately and clean up
        send_summary_email_for_call(sid)
        finalize_call(sid)

        resp.say(msg, voice="Polly.Joanna-Neural")
        resp.say(
            "Have a great day, and thanks for calling the G I Guy.",
            voice="Polly.Joanna-Neural",
        )
        resp.hangup()
        return str(resp)

    # Normal Q&A
    try:
        start = datetime.now()
        answer = ask_gi_assistant(text, caller, history=state["history"])
        latency = (datetime.now() - start).total_seconds()
    except Exception as e:
        print("[AI ERROR]", repr(e))
        fallback = (
            "I'm having trouble processing your question right now. "
            "Your message will be forwarded to the office so that staff can follow up with you."
        )

        log_qna_row(sid, caller, "error_fallback", text, fallback, None, False)
        state["qas"].append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "question": text,
            "answer": fallback,
            "latency": None,
        })

        # Error → send email + clean up
        send_summary_email_for_call(sid)
        finalize_call(sid)

        resp.say(fallback, voice="Polly.Joanna-Neural")
        resp.say(
            "Have a great day, and thanks for calling the G I Guy.",
            voice="Polly.Joanna-Neural",
        )
        resp.hangup()
        return str(resp)

    # Update convo history
    state["history"].append({"role": "user", "content": text})
    state["history"].append({"role": "assistant", "content": answer})

    # Save Q/A entry
    state["qas"].append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "question": text,
        "answer": answer,
        "latency": latency,
    })

    # Log
    stage_label = f"question_{len(state['qas'])}"
    log_qna_row(sid, caller, stage_label, text, answer, latency, False)

    # SMS summary once (after first Q&A)
    if len(state["qas"]) == 1:
        send_summary_text_once(sid, caller, text, answer)

    # Speak answer
    resp.say(answer, voice="Polly.Joanna-Neural")

    # Ask for follow-ups
    g = resp.gather(
        input="speech",
        action="/handle_question",
        method="POST",
        timeout=10,
    )
    g.say(
        "Do you have any other questions? "
        "You can ask another question now, or if you're all set, just stay quiet and the call will end automatically. "
        "There may be a short pause after you speak while I process your question.",
        voice="Polly.Joanna-Neural",
    )

    return str(resp)


@app.post("/call_status", response_class=PlainTextResponse)
async def call_status(
    CallSid: str = Form(default=""),
    CallStatus: str = Form(default=""),
):
    """
    Called by Twilio when call status changes (requires Status Callback URL config).
    When we see the call is 'completed' (or failed/busy/no-answer), we finalize.
    """
    sid = CallSid or "unknown_call"
    status = (CallStatus or "").lower()
    print(f"[STATUS] CallSid={sid} Status={status}")

    if status in {"completed", "failed", "busy", "no-answer", "canceled"}:
        # This will send recap email if not already sent and clean up state
        finalize_call(sid)

    # Twilio just needs 200 OK, empty body is fine
    return ""
