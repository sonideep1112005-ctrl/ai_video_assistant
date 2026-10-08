# 🎬 AI Video Assistant

Turn any video or audio into **notes you can chat with**. Paste a YouTube link or upload a file, and the app transcribes the speech, summarizes it, pulls out action items and decisions, and lets you ask questions about the content using RAG (Retrieval-Augmented Generation).


## ✨ Features

- 🔗 **YouTube link or file upload** (up to 500 MB)
- 🗣️ **Speech-to-text transcription** with language selection
- 🏷️ **Auto-generated title** for the video
- 📝 **Summary** of the whole content
- ✅ **Action items** extracted from the speech
- 🧭 **Key decisions** and ❓ **open questions**
- 💬 **Chat with your video**: ask questions and get answers grounded in the transcript (RAG)
- 📊 **Live progress bar** showing each processing stage
- ♻️ **Automatic retry** when the LLM API rate-limits a request

---

## 🧠 How it works

```
YouTube link / file
        │
        ▼
 Download + split audio (ffmpeg)
        │
        ▼
 Transcribe speech  ──►  Full transcript
        │
        ├──► Title, summary, action items, decisions, questions (LLM)
        │
        └──► Split into chunks ─► Embeddings ─► Vector store ─► RAG chat
```

The heavy work runs in a background thread, and the browser polls `/api/status/<job_id>` to show progress.

---

## 🗂️ Project structure

```
ai_video_assistant/
├── app.py                  # Flask app, API routes, processing pipeline
├── main.py                 # Command-line entry point
├── requirements.txt
├── .env.example            # Template for your API keys
├── core/
│   ├── transcriber.py      # Speech-to-text
│   ├── summarizer.py       # Title + summary
│   ├── extractor.py        # Action items, decisions, questions
│   ├── rag_engine.py       # RAG chain and question answering
│   └── vector_store.py     # Embeddings + vector database
├── utils/
│   └── audio_processing.py # Download, convert and split audio
├── templates/              # HTML (index.html)
└── static/                 # CSS / JS
```

---

## 🚀 Getting started

### 1. Prerequisites

- **Python 3.10+**
- **ffmpeg** installed and available in your PATH
  - Windows: `winget install ffmpeg` (or download from ffmpeg.org and add it to PATH)
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`
- A **Groq API key** from [console.groq.com](https://console.groq.com)

### 2. Clone the repo

```bash
git clone https://github.com/<your-username>/ai_video_assistant.git
cd ai_video_assistant
```

### 3. Create a virtual environment and install dependencies

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 4. Add your API key

Copy `.env.example` to `.env` and put in your real key:

```
GROQ_API_KEY=your_key_here
```

### 5. Run the app

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## 🖱️ Usage

1. Paste a YouTube link **or** choose a video/audio file.
2. Pick the spoken language.
3. Click process and watch the progress bar.
4. Read the title, summary, action items, decisions and open questions.
5. Use the chat box to ask anything about the video, for example: *"What was decided about the deadline?"*

---

## 🛠️ Tech stack

- **Backend:** Python, Flask
- **LLM + speech:** Groq API
- **RAG:** LangChain + vector database
- **Audio:** ffmpeg, yt-dlp
- **Frontend:** HTML, CSS, JavaScript

---

## ⚠️ Known limitations

- Jobs are stored in memory, so they are lost when the server restarts.
- Long videos take longer because of API rate limits (the app waits and retries automatically).
- YouTube downloads can occasionally fail depending on network or YouTube changes. Uploading a file always works.
- Run the app with a single worker process, since job state lives in memory.

---

## 🔮 Future improvements

- Save past videos and chats to a database
- Timestamps in the transcript and in chat answers
- Export summary as PDF
- Multi-video chat

---

## 👤 Author

**Deep Soni**
AI + Robotics + 3D | Creator of TRAMY | NIT Goa



If you found this useful, give the repo a ⭐
