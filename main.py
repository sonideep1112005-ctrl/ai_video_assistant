import time
from dotenv import load_dotenv
from utils.audio_processing import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question


load_dotenv()


def with_retry(fn, *args, tries=5, **kwargs):
    """Retry when Groq returns 429 (rate limit), waiting longer each time."""
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


def run_pipeline(source: str, language: str = "english") -> dict:
    print("starting ai video assistant")

    chunks = process_input(source)
    print(f"Audio chunks created: {len(chunks) if chunks else 0}")

    if not chunks:
        raise ValueError(
            "No audio could be extracted. Check the link/file and that ffmpeg is installed."
        )

    transcript = transcribe_all(chunks, language=language)
    print(f"Transcript length: {len(transcript) if transcript else 0}")

    if not transcript or not transcript.strip():
        raise ValueError(
            "Transcription came back empty. Check the audio and the transcriber settings."
        )

    print(f"raw transcription (first 300 characters ) {transcript[:300]}")

    title = with_retry(generate_title, transcript)
    time.sleep(2)

    summary = with_retry(summarize, transcript)
    time.sleep(2)

    action_item = with_retry(extract_action_items, transcript)
    time.sleep(2)

    decisions = with_retry(extract_key_decisions, transcript)
    time.sleep(2)

    questions = with_retry(extract_questions, transcript)
    time.sleep(2)

    rag_chain = with_retry(build_rag_chain, transcript)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_item,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


if __name__ == "__main__":
    # CLI entry point
    source = input("Enter YouTube URL or local file path: ").strip()
    language = input("Language (english/hinglish): ").strip() or "english"

    try:
        result = run_pipeline(source, language)
    except ValueError as e:
        print(f"\n❌ {e}")
        raise SystemExit(1)

    print("\n" + "=" * 60)
    print(f"📌 Title: {result['title']}")
    print(f"\n📋 Summary:\n{result['summary']}")
    print(f"\n✅ Action Items:\n{result['action_items']}")
    print(f"\n🔑 Key Decisions:\n{result['key_decisions']}")
    print(f"\n❓ Open Questions:\n{result['open_questions']}")
    print("=" * 60)

    print("\n💬 Chat with your meeting (type 'exit' to quit)\n")
    rag_chain = result["rag_chain"]
    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break
        if not question:
            continue
        answer = ask_question(rag_chain, question)
        print(f"\n🤖 Assistant: {answer}\n")