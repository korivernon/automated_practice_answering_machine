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

# in-memory call state (per CallSid); fine for demo / single process
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
  or anything that sounds urgent or serious:
  *Immediately* tell them this may be an emergency and to hang up and call 9 1 1
  or go to the nearest emergency room. Do not try to manage the emergency.
- Keep answers short, clear, and spoken-friendly.
- You may answer multiple questions in the same call; keep track of context.
- If the question is specific to their personal case (exact timing, medication changes,
  can they safely proceed, complex histories), say that office staff will need to call them back.
"""


# --------------------------------------------------
# Helpers: Call state
# --------------------------------------------------
def get_call_state(call_sid: str, caller: str) -> Dict[str, Any]:
    state = CALL_STATE.get(call_sid)
    if state is None:
        state = {
            "caller": caller,
            "history": [],  # list of {"role": "user"/"assistant", "content": "..."}
            "qas": [],      # list of {"question": q, "answer": a, "latency": float}
            "emergency": False,
            "summary_sms_sent": False,
        }
        CALL_STATE[call_sid] = state
    return state


def finalize_call(call_sid: str):
    """Build a single summary email from the whole call and clear state."""
    state = CALL_STATE.get(call_sid)
    if not state:
        return

    caller = state.get("caller", "Unknown caller")
    qas = state.get("qas", [])
    emergency = state.get("emergency", False)

    # Build conversation summary
    if not qas:
        convo = "No questions were captured in this call.\n"
    else:
        lines = []
        for i, qa in enumerate(qas, start=1):
            q = qa.get("question", "")
            a = qa.get("answer", "")
            lat = qa.get("latency", None)
            lat_str = f"{lat:.3f} sec" if isinstance(lat, (int, float)) else "n/a"
            lines.append(
                f"Q{i}: {q}\n"
                f"A{i}: {a}\n"
                f"Latency: {lat_str}\n"
            )
        convo = "\n\n".join(lines)

    # Email body
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    subject_prefix = "[GI Guy] EMERGENCY FLAG" if emergency else "[GI Guy] Call summary"
    subject = f"{subject_prefix} — {caller}"

    body = (
        f"Time: {ts}\n"
        f"Caller: {caller}\n"
        f"Emergency flagged: {'YES' if emergency else 'NO'}\n\n"
        f"Conversation summary:\n\n"
        f"{convo}\n"
    )

    # Send email
    if EMAIL_USER:
        try:
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = EMAIL_USER
            msg["To"] = ", ".join(PRACTICE_EMAILS)

            with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as s:
                s.starttls()
                s.login(EMAIL_USER, EMAIL_PASS)
                s.sendmail(EMAIL_USER, PRACTICE_EMAILS, msg.as_string())
        except Exception as e:
            print("EMAIL SUMMARY ERROR:", repr(e))

    # Clear state
    del CALL_STATE[call_sid]


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
    ts = datetime.now().isoformat(timespec="seconds")
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
# TEXT: Single summary SMS per call
# --------------------------------------------------
def send_summary_text_once(call_sid: str, to_number: str, question: str, answer: str):
    state = CALL_STATE.get(call_sid)
    if not state:
        return

    if state.get("summary_sms_sent"):
        return  # only once per call

    if not TWILIO_SMS_NUMBER:
        print("SMS SUMMARY ERROR: TWILIO_SMS_NUMBER not set or SMS not configured.")
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
        print(f"SMS SUMMARY SENT to {to_number}")
    except Exception as e:
        print("SMS SUMMARY ERROR:", repr(e))


# --------------------------------------------------
# Emergency detection
# --------------------------------------------------
def is_emergency_like(text: str) -> bool:
    t = text.lower()
    keywords = [
        "severe pain",
        "bad pain",
        "really bad pain",
        "chest pain",
        "stomach pain",
        "abdominal pain",
        "hurts a lot",
        "bleeding",
        "vomiting blood",
        "throwing up blood",
        "black stool",
        "bloody stool",
        "shortness of breath",
        "can’t breathe",
        "cant breathe",
        "trouble breathing",
        "passed out",
        "fainted",
        "dizzy",
    ]
    # also treat generic "pain" as emergency-ish for this practice
    if "pain" in t:
        return True
    return any(k in t for k in keywords)


# --------------------------------------------------
# AI Helper (with conversation history)
# --------------------------------------------------
def ask_gi_assistant(
    user_message: str,
    caller: Optional[str],
    history: Optional[list] = None,
) -> str:
    history = history or []
    messages = [{"role": "system", "content": GI_SYSTEM_PROMPT}]

    # include prior turns for context
    messages.extend(history)

    messages.append({
        "role": "user",
        "content": f"Caller: {caller or 'unknown'}\nQuestion: {user_message}",
    })

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=0.3,
    )
    reply = completion.choices[0].message.content.strip()
    return reply


# --------------------------------------------------
# FASTAPI ROUTES
# --------------------------------------------------
@app.post("/voice", response_class=PlainTextResponse)
async def inbound_call():
    resp = VoiceResponse()

    g = resp.gather(
        input="speech",
        action="/handle_question",
        method="POST",
        timeout=10,  # 10 seconds of silence = done
    )
    g.say(
        "Hi, you've reached G I Guy, the office of Doctor Kurt Vernon. "
        "I can answer many frequently asked questions about GI procedures, prep, and prescriptions. "
        "There may be a short pause after you speak while I process your question. "
        "If at any point you'd rather speak directly with the office, you can hang up and call back during normal office hours. "
        "Please briefly describe your question. You can start speaking now.",
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
    call_sid = CallSid or "unknown_call"

    state = get_call_state(call_sid, caller)

    # If no speech (timeout) -> end call, send one summary email, hang up
    if not text:
        finalize_call(call_sid)
        resp.say(
            "Have a great day, and thanks for calling the G I Guy.",
            voice="Polly.Joanna-Neural",
        )
        resp.hangup()
        return str(resp)

    # Check for emergency
    if is_emergency_like(text):
        state["emergency"] = True
        emergency_msg = (
            "Based on what you described, this may be an emergency. "
            "I am not able to help with emergencies. "
            "Please hang up now and call 9 1 1, or go to the nearest emergency room immediately."
        )

        # log
        log_qna_row(
            call_sid=call_sid,
            caller=caller,
            stage="emergency",
            question=text,
            answer=emergency_msg,
            latency_seconds=None,
            emergency_flag=True,
        )
        # store in state for email summary
        state["qas"].append({
            "question": text,
            "answer": emergency_msg,
            "latency": None,
        })

        # finalize & end call
        finalize_call(call_sid)

        resp.say(emergency_msg, voice="Polly.Joanna-Neural")
        resp.say(
            "Have a great day, and thanks for calling the G I Guy.",
            voice="Polly.Joanna-Neural",
        )
        resp.hangup()
        return str(resp)

    # Normal, non-emergency Q&A
    start = datetime.now()
    answer = ask_gi_assistant(text, caller=caller, history=state["history"])
    latency = (datetime.now() - start).total_seconds()

    # Update conversation history for LLM
    state["history"].append({"role": "user", "content": text})
    state["history"].append({"role": "assistant", "content": answer})

    # Append to Q/A list for summary
    state["qas"].append({
        "question": text,
        "answer": answer,
        "latency": latency,
    })

    # Log this turn to CSV
    stage_label = f"question_{len(state['qas'])}"
    log_qna_row(
        call_sid=call_sid,
        caller=caller,
        stage=stage_label,
        question=text,
        answer=answer,
        latency_seconds=latency,
        emergency_flag=False,
    )

    # Send SMS summary once per call (based on first Q/A)
    if len(state["qas"]) == 1:
        send_summary_text_once(call_sid, caller, text, answer)

    # Speak answer
    resp.say(answer, voice="Polly.Joanna-Neural")

    # Ask if they have more questions; silence = end
    g = resp.gather(
        input="speech",
        action="/handle_question",
        method="POST",
        timeout=10,  # 10 seconds silence -> we end and summarize
    )
    g.say(
        "Hi, you've reached G I Guy, the office of Doctor Kurt Vernon. "
        "Before we get started, heads up — there may be a short pause after you speak while I process your question. "
        "That's totally normal. "
        "I can answer many frequently asked questions about G I procedures, prep instructions, and prescriptions. "
        "You will be sent a summary of your response, and a response will also be provided to office staff. If you'd prefer to speak directly with the office, you can hang up anytime and call back during normal hours. "
        "Whenever you're ready, please briefly describe your question.",
        voice="Polly.Joanna-Neural",
    )

    return str(resp)
