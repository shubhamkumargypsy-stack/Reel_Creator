# Audio → Emotional Story Video

Turn an audio file into an Instagram-ready animated video with captions — using only
free, no-subscription tools. Everything runs **locally in your browser**; the only thing
that needs a (free) API key is stock footage.

## Status: full pipeline ✅

**Audio in → transcript (word timings) → emotion per sentence → matching stock
footage → animated captioned video you can download.**

Everything runs on your device: no server, no upload. The AI models (Whisper +
emotion classifiers) download once from the Hugging Face CDN and cache in your
browser. Stock photos come from Pexels (free key). There is no LLM and no paid service.

### Pipeline
1. **Analyze** — Whisper transcribes with word timestamps; a text model (28-emotion
   go_emotions) and an acoustic model (wav2vec2 voice tone) score each sentence.
2. **Find footage** — each sentence becomes a keyword query (nouns + emotion mood word)
   and pulls matching vertical photos from Pexels. Click any storyboard card to swap it.
3. **Make video** — a canvas animates each photo (Ken Burns pan/zoom, intensity scaled
   by the emotion), tints it by emotion, and burns in karaoke captions synced to the
   word timings. Record it to WebM, optionally convert to MP4. Optional background music
   is mixed low under the narration. Output is capped at 3 minutes.


## Run it

In-browser models need to be served over `http://` (not opened as a `file://` path).
From this folder:

```bash
# Python (any 3.x)
python -m http.server 8000
```

Then open <http://localhost:8000> in **Chrome or Edge** (best Whisper support).

1. Choose or drag in an audio **or video** file (mp3, wav, m4a, mp4, mov, …).
   For video files, only the audio track is decoded — the video is ignored entirely.
2. (Optional) tick **Try WebGPU** for a big speedup on Chrome/Edge.
3. Click **1 · Analyze** — you get the transcript + per-sentence emotions.
4. Paste your free **Pexels API key** (get one at <https://www.pexels.com/api/>) and
   click **2 · Find footage**. A storyboard appears; click any card to swap its photo.
5. Click **3 · Make video**, then **▶ Play preview** to check it, and
   **● Render & download** to save a WebM. Hit **Convert last render to MP4** for an
   Instagram-friendly file. (Optional: pick a background-music file before rendering.)

The **first** run downloads the models (tens of MB) and is slow. Later runs are fast
because the models are cached.

## Notes / known trade-offs
- `whisper-tiny.en` is the default (fastest). Switch to `whisper-base.en` for better
  accuracy at the cost of speed. Use `whisper-tiny` (no `.en`) for non-English audio.
- Emotion is shown two ways per sentence:
  - **Voice (tone)** — an acoustic model (`wav2vec2` SER) that hears *how* it's said
    (angry/happy/sad/calm/…). Toggle it off with the "Analyze voice tone" checkbox if slow.
  - **Words (text)** — `SamLowe/roberta-base-go_emotions-onnx` (28 emotions) on the transcript.
    If it can't load, the app falls back to a positive/negative sentiment model and says so.
- Text emotion depends on transcription quality, so if accuracy is poor, switch the
  dropdown to `whisper-base.en` — fewer transcription errors means better text emotion.
- Accuracy depends on audio quality; noisy recordings degrade both transcription and emotion.

## Tech
- [transformers.js](https://huggingface.co/docs/transformers.js) — runs ONNX models in the browser
- Whisper (speech-to-text) + RoBERTa go_emotions (text emotion) + wav2vec2 (voice emotion) — all open-weight, permissively licensed
- [Pexels API](https://www.pexels.com/api/) — free stock photos (the only thing needing a key)
- [ffmpeg.wasm](https://ffmpegwasm.netlify.app/) — audio extraction fallback + WebM→MP4, in-browser
- Canvas 2D + MediaRecorder — the animation and video capture

## Notes on the video step
- Recording happens in **real time** (a 2-minute video takes ~2 minutes to record) because
  it captures the live canvas + audio. Keep the tab visible while it records.
- Browsers record **WebM**; Instagram prefers **MP4**, so use the convert button. The MP4
  conversion runs ffmpeg.wasm locally and can take a little while on long clips.
- Stock photos are loaded with CORS so the canvas can export them. Pexels supports this;
  if a specific image ever fails to load, that scene falls back to an emotion color card.
