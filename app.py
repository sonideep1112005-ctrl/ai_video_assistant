import os
import time
import uuid
import threading
import traceback

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from utils.audio_processing import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB uploads

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# job_id -> {"stage", "progress", "status", "result", "error", "rag_chain"}
JOBS = {}


def with_retry(fn, *args, tries=5, **kwargs):
    """Retry a call when Groq returns 429 (rate limit), waiting longer each time."""
    for i in range(tries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            msg = str(e).lower()
            if "429" in msg or "rate limit" in msg or "rate_limit" in msg:
                wait = 10 * (i + 1)  # 10s, 20s, 30s, 40s, 50s
                print(f"Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            raise
    return fn(*args, **kwargs)


def run_pipeline(source: str, language: str = "english", on_stage=lambda s, p: None) -> dict:
    """Same pipeline as before, with progress callbacks for the UI."""
    on_stage("Downloading & splitting audio", 8)
    chunks = process_input(source)
    print(f"Audio chunks created: {len(chunks) if chunks else 0}")

    if not chunks:
        raise ValueError(
            "No audio could be extracted. Check the link/file and that ffmpeg is installed."
        )

    on_stage("Transcribing speech", 30)
    transcript = transcribe_all(chunks, language=language)
    print(f"Transcript length: {len(transcript) if transcript else 0}")

    if not transcript or not transcript.strip():
        raise ValueError(
            "Transcription came back empty. Check the audio and the transcriber settings."
        )

    on_stage("Writing a title", 52)
    title = with_retry(generate_title, transcript)
    time.sleep(2)

    on_stage("Summarizing", 64)
    summary = with_retry(summarize, transcript)
    time.sleep(2)

    on_stage("Finding action items", 76)
    action_items = with_retry(extract_action_items, transcript)
    time.sleep(2)

    on_stage("Spotting key decisions", 86)
    decisions = with_retry(extract_key_decisions, transcript)
    time.sleep(2)
    questions = with_retry(extract_questions, transcript)
    time.sleep(2)

    on_stage("Building chat memory", 94)
    rag_chain = with_retry(build_rag_chain, transcript)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


def _worker(job_id, source, language):
    job = JOBS[job_id]

    def on_stage(stage, progress):
        job["stage"], job["progress"] = stage, progress

    try:
        result = run_pipeline(source, language, on_stage)
        job["rag_chain"] = result.pop("rag_chain")
        job["result"] = result
        job["progress"], job["stage"], job["status"] = 100, "Done", "done"
    except Exception as e:  # noqa
        traceback.print_exc()
        job["status"], job["error"] = "error", str(e)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/process", methods=["POST"])
def process():
    language = (request.form.get("language") or "english").strip()
    source = (request.form.get("url") or "").strip()
    file = request.files.get("file")

    if file and file.filename:
        path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{secure_filename(file.filename)}")
        file.save(path)
        source = path
    if not source:
        return jsonify({"error": "Paste a YouTube link or choose a file."}), 400

    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"status": "running", "stage": "Starting", "progress": 2}
    threading.Thread(target=_worker, args=(job_id, source, language), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job"}), 404
    out = {k: job.get(k) for k in ("status", "stage", "progress", "error", "result")}
    return jsonify(out)


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    job = JOBS.get(data.get("job_id"))
    question = (data.get("question") or "").strip()
    if not job or "rag_chain" not in job:
        return jsonify({"error": "Process a video first."}), 400
    if not question:
        return jsonify({"error": "Type a question."}), 400
    try:
        return jsonify({"answer": with_retry(ask_question, job["rag_chain"], question)})
    except Exception as e:  # noqa
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)