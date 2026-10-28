// Audio -> emotion -> stock footage -> animated captioned video.
// Everything runs in the browser. Models come from the Hugging Face CDN (cached
// after first use); stock photos come from Pexels (free key). Nothing is uploaded.

import {
  pipeline,
  env,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3";

env.allowLocalModels = false;

// ---- Models ------------------------------------------------------------
const EMOTION_MODELS = [
  "SamLowe/roberta-base-go_emotions-onnx",
  "Xenova/distilbert-base-uncased-finetuned-sst-2-english", // sentiment fallback
];
const SER_MODEL = "onnx-community/wav2vec2-base-Speech_Emotion_Recognition-ONNX";

const SR = 16000;
const MAX_SECONDS = 180; // hard cap: 3 minutes

// Emotion -> [tint color, arousal 0..1 for motion intensity].
const EMOTION_INFO = {
  joy: ["#ffd166", 0.8], amusement: ["#ffd166", 0.8], excitement: ["#ff9f1c", 1.0],
  love: ["#ff6b9d", 0.6], admiration: ["#f4a261", 0.6], approval: ["#90be6d", 0.4],
  gratitude: ["#90be6d", 0.4], optimism: ["#ffd166", 0.6], caring: ["#ff8fa3", 0.4],
  pride: ["#f4a261", 0.7], relief: ["#90be6d", 0.3], desire: ["#ff6b9d", 0.7],
  sadness: ["#5b8def", 0.2], grief: ["#4361ee", 0.2], disappointment: ["#6c8ec9", 0.3],
  remorse: ["#6c8ec9", 0.3], anger: ["#ef476f", 1.0], annoyance: ["#f77f00", 0.7],
  disgust: ["#8ac926", 0.6], disapproval: ["#e07a5f", 0.5], fear: ["#9b5de5", 0.9],
  nervousness: ["#9b5de5", 0.7], embarrassment: ["#c77dff", 0.5], surprise: ["#4ad6b0", 0.9],
  realization: ["#4ad6b0", 0.5], curiosity: ["#4cc9f0", 0.6], confusion: ["#90a8c3", 0.5],
  neutral: ["#9aa0c7", 0.4], positive: ["#90be6d", 0.6], negative: ["#ef476f", 0.6],
  // acoustic SER labels
  angry: ["#ef476f", 1.0], happy: ["#ffd166", 0.9], sad: ["#5b8def", 0.2],
  calm: ["#4ad6b0", 0.3], fearful: ["#9b5de5", 0.9], surprised: ["#4ad6b0", 0.9],
  disgusted: ["#8ac926", 0.6], ps: ["#4ad6b0", 0.9],
};
const emoColor = (l) => (EMOTION_INFO[l?.toLowerCase()] || EMOTION_INFO.neutral)[0];
const emoArousal = (l) => (EMOTION_INFO[l?.toLowerCase()] || EMOTION_INFO.neutral)[1];

// Visual mood word appended to the stock-photo query, by emotion.
const MOOD_WORD = {
  sadness: "lonely", grief: "somber", sad: "melancholy", fear: "dark", fearful: "ominous",
  anger: "dramatic", angry: "intense", joy: "bright", happy: "sunny", excitement: "vibrant",
  love: "warm", calm: "serene", neutral: "", surprise: "surreal", surprised: "surreal",
};

const STOPWORDS = new Set("a an the and or but if then this that these those i you he she it we they me him her us them my your his its our their am is are was were be been being have has had do does did will would shall should can could may might must not no nor so than too very just only also as at by for from in into of off on onto out over to up with about after again against all any because before below between both down during each few more most other some such own same how what when where which who whom why yours".split(/\s+/));

// ---- DOM ---------------------------------------------------------------
const el = (id) => document.getElementById(id);
const audioInput = el("audioInput");
const musicInput = el("musicInput");
const userPhotosInput = el("userPhotosInput");
const fileLabel = el("fileLabel");
const drop = el("drop");
const analyzeBtn = el("analyzeBtn");
const footageBtn = el("footageBtn");
const renderBtn = el("renderBtn");
const statusEl = el("status");
const progressEl = el("progress");
const player = el("player");
const resultsPanel = el("resultsPanel");
const tbody = document.querySelector("#resultsTable tbody");
const transcriptEl = el("transcript");
const storyboardPanel = el("storyboardPanel");
const storyboardEl = el("storyboard");
const videoPanel = el("videoPanel");
const stage = el("stage");
const ctx = stage.getContext("2d");
const playBtn = el("playBtn");
const recordBtn = el("recordBtn");
const mp4Btn = el("mp4Btn");
const downloadLink = el("downloadLink");
const apiKeyInput = el("apiKey");
const providerSel = el("stockProvider");

// ---- State -------------------------------------------------------------
let selectedFile = null;
let musicFile = null;
let userPhotos = [];     // [{img, url, name}] — user-uploaded photos to place by content
let audioSamples = null; // Float32Array @16k (for SER slicing)
let scenes = [];         // [{text,start,end,words,voice,text,emotion,query,photos,idx,img}]
let asr = null, emo = null, ser = null, emoLabelMode = "emotion";
let clip = null;         // CLIP image↔text matcher for placing user photos
let ffmpeg = null, ffmpegFetchFile = null;
let audioGraph = null;   // {ctx, dest} for recording
let musicBuffer = null;
let lastWebmBlob = null;
let denoisedUrl = null;  // object URL of the cleaned narration, when denoise is on
let originalUrl = null;  // object URL of the original file (to revert when denoise is off)
let baseVideoEl = null;  // the person's video, used as the base layer in talking mode

// ---- Small helpers -----------------------------------------------------
function setStatus(msg, isError = false) {
  statusEl.textContent = msg;
  statusEl.classList.toggle("error", isError);
}
function showProgress(f) {
  if (f == null) { progressEl.hidden = true; return; }
  progressEl.hidden = false;
  progressEl.value = Math.round(f * 100);
}
function fmtTime(s) {
  if (s == null || Number.isNaN(s)) return "–";
  const m = Math.floor(s / 60);
  return `${m}:${(s % 60).toFixed(1).padStart(4, "0")}`;
}
function escapeHtml(s) {
  return (s || "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

// ---- Audio decode (built-in -> ffmpeg fallback) ------------------------
async function decodeTo16kMono(file) {
  try {
    return await decodeViaWebAudio(file);
  } catch (err) {
    console.warn("Built-in decode failed; using FFmpeg.", err);
    setStatus("Browser couldn't decode this file — extracting audio with FFmpeg (first time downloads ~30 MB)…");
    return await decodeViaFFmpeg(file);
  }
}
async function decodeViaWebAudio(file) {
  const arrayBuf = await file.arrayBuffer();
  const AC = window.AudioContext || window.webkitAudioContext;
  const ac = new AC();
  let decoded;
  try { decoded = await ac.decodeAudioData(arrayBuf); } finally { ac.close(); }
  if (decoded.sampleRate === SR && decoded.numberOfChannels === 1) return decoded.getChannelData(0);
  const offline = new OfflineAudioContext(1, Math.ceil(decoded.duration * SR), SR);
  const src = offline.createBufferSource();
  src.buffer = decoded; src.connect(offline.destination); src.start();
  return (await offline.startRendering()).getChannelData(0);
}
async function loadFFmpeg() {
  if (ffmpeg) return ffmpeg;
  const { FFmpeg } = await import("https://cdn.jsdelivr.net/npm/@ffmpeg/ffmpeg@0.12.10/dist/esm/index.js");
  const { toBlobURL, fetchFile } = await import("https://cdn.jsdelivr.net/npm/@ffmpeg/util@0.12.1/dist/esm/index.js");
  ffmpegFetchFile = fetchFile;
  const base = "https://cdn.jsdelivr.net/npm/@ffmpeg/core@0.12.6/dist/esm";
  const ff = new FFmpeg();
  ff.on("progress", ({ progress }) => { if (progress > 0 && progress <= 1) showProgress(progress); });
  await ff.load({
    coreURL: await toBlobURL(`${base}/ffmpeg-core.js`, "text/javascript"),
    wasmURL: await toBlobURL(`${base}/ffmpeg-core.wasm`, "application/wasm"),
  });
  ffmpeg = ff;
  return ffmpeg;
}
async function decodeViaFFmpeg(file) {
  const ff = await loadFFmpeg();
  const inName = "in_" + (file.name || "media").replace(/[^\w.-]/g, "_");
  await ff.writeFile(inName, await ffmpegFetchFile(file));
  setStatus("Extracting audio with FFmpeg…");
  await ff.exec(["-i", inName, "-vn", "-ac", "1", "-ar", String(SR), "-f", "f32le", "out.pcm"]);
  const data = await ff.readFile("out.pcm");
  await ff.deleteFile(inName).catch(() => {});
  await ff.deleteFile("out.pcm").catch(() => {});
  const aligned = data.slice();
  showProgress(null);
  return new Float32Array(aligned.buffer, 0, Math.floor(aligned.byteLength / 4));
}

// Denoise with FFmpeg: high-pass (kill rumble) + FFT denoiser. Returns a clean WAV Blob,
// used both for the models (better transcription/emotion) and as the video's narration.
async function denoiseToWavBlob(file) {
  const ff = await loadFFmpeg();
  const inName = "dn_" + (file.name || "media").replace(/[^\w.-]/g, "_");
  await ff.writeFile(inName, await ffmpegFetchFile(file));
  const run = (af) => ff.exec(["-i", inName, "-vn", "-af", af, "-ac", "1", "-ar", "44100", "clean.wav"]);
  try {
    await run("highpass=f=80,afftdn=nf=-25,lowpass=f=12000");   // FFT denoise
  } catch (e) {
    console.warn("afftdn unavailable, using band-pass only", e);
    await run("highpass=f=80,lowpass=f=10000");                 // fallback
  }
  const data = await ff.readFile("clean.wav");
  await ff.deleteFile(inName).catch(() => {});
  await ff.deleteFile("clean.wav").catch(() => {});
  showProgress(null);
  return new Blob([data.slice().buffer], { type: "audio/wav" });
}

// ---- Sentence building (keeps word timings for karaoke) ----------------
function buildSentences(chunks) {
  const out = [];
  let cur = null;
  const push = () => {
    if (cur && cur.text.trim()) out.push({ text: cur.text.trim(), start: cur.start, end: cur.end, words: cur.words });
    cur = null;
  };
  for (const c of chunks) {
    const word = c.text ?? "";
    const [s, e] = c.timestamp ?? [null, null];
    if (!cur) cur = { text: "", start: s, end: e, words: [] };
    if (cur.start == null) cur.start = s;
    if (e != null) cur.end = e;
    cur.text += word;
    cur.words.push({ text: word.trim(), start: s, end: e });
    if (/[.!?…]["'”’)\]]?\s*$/.test(word)) push();
  }
  push();
  if (out.length === 0 && chunks.length) {
    out.push({
      text: chunks.map((c) => c.text).join("").trim(),
      start: chunks[0].timestamp?.[0] ?? 0,
      end: chunks[chunks.length - 1].timestamp?.[1] ?? null,
      words: chunks.map((c) => ({ text: (c.text || "").trim(), start: c.timestamp?.[0], end: c.timestamp?.[1] })),
    });
  }
  return out;
}

// ---- Model loaders -----------------------------------------------------
function makeProgressCallback(name) {
  return (d) => {
    if (d.status === "progress" && d.total) {
      setStatus(`Downloading ${name}: ${d.file} (${Math.round((d.loaded / d.total) * 100)}%)`);
      showProgress(d.loaded / d.total);
    } else if (d.status === "ready" || d.status === "done") showProgress(null);
  };
}
async function getASR(model, device) {
  if (asr && asr._model === model && asr._device === device) return asr;
  setStatus(`Loading transcription model (${model})…`);
  const opts = { progress_callback: makeProgressCallback("transcriber") };
  if (device) opts.device = device;
  const p = await pipeline("automatic-speech-recognition", model, opts);
  p._model = model; p._device = device; asr = p; return asr;
}
async function getEmotion(device) {
  if (emo) return emo;
  let lastErr;
  for (const model of EMOTION_MODELS) {
    try {
      setStatus(`Loading emotion model (${model})…`);
      const opts = { progress_callback: makeProgressCallback("emotion") };
      if (device) opts.device = device;
      const p = await pipeline("text-classification", model, opts);
      p._model = model; emoLabelMode = model.includes("sst-2") ? "sentiment" : "emotion"; emo = p; return emo;
    } catch (err) { console.warn(`Emotion model failed: ${model}`, err); lastErr = err; }
  }
  throw lastErr || new Error("No emotion model loaded.");
}
async function getSER(device) {
  if (ser) return ser;
  setStatus("Loading voice-tone model…");
  const opts = { progress_callback: makeProgressCallback("voice tone") };
  if (device) opts.device = device;
  ser = await pipeline("audio-classification", SER_MODEL, opts);
  return ser;
}
function sliceAudio(audio, start, end) {
  const n = audio.length;
  const a = Math.max(0, Math.floor((start ?? 0) * SR));
  let b = Math.min(n, Math.ceil((end ?? n / SR) * SR));
  if (b <= a) b = Math.min(n, a + Math.floor(0.4 * SR));
  const slice = audio.subarray(a, b);
  const minLen = Math.floor(0.4 * SR);
  if (slice.length < minLen) { const p = new Float32Array(minLen); p.set(slice); return p; }
  return slice;
}

// ---- Emotion fusion + query --------------------------------------------
function fuseEmotion(voice, text) {
  const v = voice && voice[0], t = text && text[0];
  if (v && v.score >= 0.5) return v.label; // confident tone wins
  if (t) return t.label;
  if (v) return v.label;
  return "neutral";
}
function extractQuery(text, emotion) {
  // Keep original case so we can spot proper nouns (names/places) as likely subjects.
  const tokens = text.split(/\s+/).map((w) => w.replace(/[^A-Za-z0-9']/g, "")).filter(Boolean);
  const best = new Map();
  tokens.forEach((tok, i) => {
    const low = tok.toLowerCase();
    if (low.length < 3 || STOPWORDS.has(low)) return;
    let score = low.length;                         // longer words tend to be more concrete
    if (i > 0 && /^[A-Z]/.test(tok)) score += 8;    // mid-sentence capital = proper noun
    if (score > (best.get(low) || 0)) best.set(low, score);
  });
  const ranked = [...best.entries()].sort((a, b) => b[1] - a[1]).map((e) => e[0]);
  let terms = ranked.slice(0, 3);
  if (!terms.length) terms = tokens.slice(0, 2).map((t) => t.toLowerCase());
  // Only lean on the emotion mood word when the sentence gives us little to go on.
  const mood = terms.length < 2 ? MOOD_WORD[emotion?.toLowerCase()] || "" : "";
  return {
    query: (terms.slice(0, 2).join(" ") + (mood ? " " + mood : "")).trim(),
    keywords: ranked.length ? ranked : terms,
  };
}
function scorePhoto(photo, keywords) {
  const hay = ((photo.alt || "") + "").toLowerCase();
  let s = 0;
  for (const k of keywords) if (hay.includes(k)) s += 2;
  return s;
}

// ---- CLIP: place the user's own photos onto best-matching scenes --------
async function getCLIP(device) {
  if (clip) return clip;
  setStatus("Loading image matcher (CLIP)…");
  const opts = { progress_callback: makeProgressCallback("image matcher") };
  if (device) opts.device = device;
  clip = await pipeline("zero-shot-image-classification", "Xenova/clip-vit-base-patch32", opts);
  return clip;
}
// Assign each uploaded photo to the scene whose sentence it matches best (1:1, greedy).
// asOverlay=true → store as an overlay above the talking video; otherwise replace the scene's media.
async function placeUserPhotos(device, asOverlay) {
  if (!userPhotos.length || !scenes.length) return 0;
  let model;
  try { model = await getCLIP(device); }
  catch (err) { if (device) { try { model = await getCLIP(undefined); } catch {} } }
  if (!model) { console.warn("CLIP unavailable"); return 0; }

  const texts = scenes.map((s) => (s.text || "").slice(0, 120) || "scene");
  const pairs = [];
  for (let i = 0; i < userPhotos.length; i++) {
    setStatus(`Matching your photo ${i + 1}/${userPhotos.length} to the story…`);
    try {
      const out = await model(userPhotos[i].url, texts);
      const m = new Map(out.map((o) => [o.label, o.score]));
      for (let j = 0; j < scenes.length; j++) pairs.push({ i, j, score: m.get(texts[j]) ?? 0 });
    } catch (err) { console.warn("CLIP match failed", i, err); }
  }
  pairs.sort((a, b) => b.score - a.score);
  const usedImg = new Set(), usedScene = new Set();
  let placed = 0;
  for (const p of pairs) {
    if (usedImg.has(p.i) || usedScene.has(p.j)) continue;
    usedImg.add(p.i); usedScene.add(p.j);
    const sc = scenes[p.j], up = userPhotos[p.i];
    if (asOverlay) {
      sc.overlayImg = up.img; sc.overlayAssigned = true;
    } else {
      disposeSceneMedia(sc);
      sc.userAssigned = true; sc.kind = "image"; sc.img = up.img;
      sc.photos = [{ thumb: up.url, full: up.url, user: true }]; sc.clips = []; sc.idx = 0;
    }
    placed++;
  }
  return placed;
}

// ---- Pexels / Pixabay --------------------------------------------------
async function pexelsSearch(query, key) {
  const url = `https://api.pexels.com/v1/search?query=${encodeURIComponent(query)}&per_page=12&orientation=portrait`;
  const res = await fetch(url, { headers: { Authorization: key } });
  if (!res.ok) throw new Error(`Pexels ${res.status}`);
  const data = await res.json();
  return (data.photos || []).map((p) => ({
    thumb: p.src.tiny || p.src.small,
    full: p.src.portrait || p.src.large2x || p.src.large,
    alt: p.alt, by: p.photographer,
  }));
}
async function pixabaySearch(query, key) {
  const url = `https://pixabay.com/api/?key=${encodeURIComponent(key)}&q=${encodeURIComponent(query)}&image_type=photo&orientation=vertical&per_page=12&safesearch=true`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Pixabay ${res.status}`);
  const data = await res.json();
  // Pixabay terms: use webformatURL (CORS-friendly), not largeImageURL hotlinks.
  return (data.hits || []).map((h) => ({
    thumb: h.previewURL || h.webformatURL,
    full: h.webformatURL || h.largeImageURL,
    alt: h.tags, by: h.user,
  }));
}
function stockSearch(query, key, provider) {
  return provider === "pixabay" ? pixabaySearch(query, key) : pexelsSearch(query, key);
}
function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("image load failed: " + url));
    img.src = url;
  });
}

// ---- Stock VIDEO (same free keys as photos) ----------------------------
function pickPexelsFile(files) {
  const mp4 = (files || []).filter((f) => f.file_type === "video/mp4" && f.link);
  mp4.sort((a, b) => (a.width || 0) - (b.width || 0));
  const pick = mp4.find((f) => (f.width || 0) >= 540) || mp4[mp4.length - 1];
  return pick ? pick.link : null;
}
async function pexelsVideoSearch(query, key) {
  const url = `https://api.pexels.com/videos/search?query=${encodeURIComponent(query)}&per_page=10&orientation=portrait&size=medium`;
  const res = await fetch(url, { headers: { Authorization: key } });
  if (!res.ok) throw new Error(`Pexels ${res.status}`);
  const data = await res.json();
  return (data.videos || [])
    .map((v) => ({ thumb: v.image, file: pickPexelsFile(v.video_files), alt: (v.tags || []).join(" "), by: v.user?.name }))
    .filter((v) => v.file);
}
async function pixabayVideoSearch(query, key) {
  const url = `https://pixabay.com/api/videos/?key=${encodeURIComponent(key)}&q=${encodeURIComponent(query)}&per_page=10`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Pixabay ${res.status}`);
  const data = await res.json();
  return (data.hits || [])
    .map((h) => ({
      thumb: `https://i.vimeocdn.com/video/${h.picture_id}_295x166.jpg`,
      file: h.videos?.medium?.url || h.videos?.small?.url || h.videos?.tiny?.url,
      alt: h.tags, by: h.user,
    }))
    .filter((v) => v.file);
}
function stockVideoSearch(query, key, provider) {
  return provider === "pixabay" ? pixabayVideoSearch(query, key) : pexelsVideoSearch(query, key);
}
// Load a clip into a muted, looping, detached <video>. crossOrigin so canvas export isn't tainted;
// if the CDN doesn't allow CORS the element errors (we then fall back to a photo).
function loadVideo(url) {
  return new Promise((resolve, reject) => {
    const v = document.createElement("video");
    v.crossOrigin = "anonymous";
    v.muted = true; v.loop = true; v.playsInline = true; v.preload = "auto";
    v.style.cssText = "position:absolute;left:-9999px;width:1px;height:1px;opacity:0;pointer-events:none";
    const to = setTimeout(() => { cleanup(); reject(new Error("video load timeout")); }, 20000);
    function cleanup() { clearTimeout(to); v.removeEventListener("canplay", ok); v.removeEventListener("error", er); }
    const ok = () => { cleanup(); resolve(v); };
    const er = () => { cleanup(); try { v.remove(); } catch {} reject(new Error("video load failed")); };
    v.addEventListener("canplay", ok);
    v.addEventListener("error", er);
    document.body.appendChild(v);
    v.src = url;
    v.load();
  });
}
function disposeSceneMedia(sc) {
  if (sc.videoEl) { try { sc.videoEl.pause(); sc.videoEl.removeAttribute("src"); sc.videoEl.load(); sc.videoEl.remove(); } catch {} sc.videoEl = null; }
}

// ---- Analyze (transcribe + emotion + build scenes) ---------------------
async function analyze() {
  if (!selectedFile) return;
  setBusy(true);
  resultsPanel.hidden = true; storyboardPanel.hidden = true; videoPanel.hidden = true;
  tbody.innerHTML = "";
  const model = el("asrModel").value;
  const device = el("useWebGPU").checked ? "webgpu" : undefined;
  const wantVoice = el("analyzeVoice").checked;

  try {
    setStatus("Decoding audio…");
    let srcForModels = selectedFile;
    if (el("denoise").checked) {
      try {
        setStatus("Reducing background noise with FFmpeg… (first time downloads ~30 MB)");
        const cleanBlob = await denoiseToWavBlob(selectedFile);
        if (denoisedUrl) URL.revokeObjectURL(denoisedUrl);
        denoisedUrl = URL.createObjectURL(cleanBlob);
        player.src = denoisedUrl;          // video + preview use the cleaned narration
        srcForModels = cleanBlob;          // transcription + emotion use it too
      } catch (err) {
        console.warn("Denoise failed; using original audio.", err);
        setStatus("Denoise failed — continuing with the original audio…");
      }
    } else if (originalUrl) {
      player.src = originalUrl;            // denoise off → ensure original audio is used
    }
    audioSamples = await decodeTo16kMono(srcForModels);

    let transcriber;
    try { transcriber = await getASR(model, device); }
    catch (err) {
      if (device) { setStatus("WebGPU unavailable — using CPU…"); transcriber = await getASR(model, undefined); }
      else throw err;
    }

    setStatus("Transcribing… (slow on first run)");
    showProgress(null);
    const result = await transcriber(audioSamples, { return_timestamps: "word", chunk_length_s: 30, stride_length_s: 5 });
    const sentences = buildSentences(result.chunks || []);
    transcriptEl.textContent = (result.text || "").trim() || "(no speech detected)";

    const emotion = await getEmotion(device).catch((err) => { if (device) return getEmotion(undefined); throw err; });
    let voiceModel = null;
    if (wantVoice) {
      try { voiceModel = await getSER(device); }
      catch (err) { if (device) { try { voiceModel = await getSER(undefined); } catch {} } if (!voiceModel) console.warn("No voice model", err); }
    }

    setStatus(`Scoring emotion for ${sentences.length} sentence(s)…`);
    scenes = [];
    for (let i = 0; i < sentences.length; i++) {
      const s = sentences[i];
      let textScores = [], voiceScores = [];
      try { const o = await emotion(s.text, { top_k: 3 }); textScores = Array.isArray(o) ? o : [o]; }
      catch (err) { console.warn("text emotion fail", i, err); }
      if (voiceModel) {
        try { const o = await voiceModel(sliceAudio(audioSamples, s.start, s.end), { top_k: 2 }); voiceScores = Array.isArray(o) ? o : [o]; }
        catch (err) { console.warn("voice emotion fail", i, err); }
      }
      const emotionLabel = fuseEmotion(voiceScores, textScores);
      const eq = extractQuery(s.text, emotionLabel);
      scenes.push({
        ...s, voiceScores, textScores, emotion: emotionLabel,
        query: eq.query, keywords: eq.keywords, photos: [], idx: 0, img: null,
      });
      renderRow(i + 1, s, voiceScores, textScores);
    }

    resultsPanel.hidden = false;
    renderStoryboard();
    storyboardPanel.hidden = false;
    setStatus(emoLabelMode === "sentiment"
      ? "Analyzed. (Fine-grained emotion model unavailable — words show sentiment.) Now click “Find footage”."
      : "Analyzed. Now click “Find footage” (needs your free Pexels key).");
    showProgress(null);
  } catch (err) {
    console.error(err); setStatus(`Error: ${err.message || err}`, true); showProgress(null);
  } finally { setBusy(false); }
}

function renderRow(index, sentence, voiceScores, textScores) {
  const tr = document.createElement("tr");
  const tdIdx = document.createElement("td"); tdIdx.className = "idx"; tdIdx.textContent = index;
  const tdTime = document.createElement("td"); tdTime.className = "time";
  tdTime.textContent = `${fmtTime(sentence.start)} – ${fmtTime(sentence.end)}`;
  const tdText = document.createElement("td"); tdText.textContent = sentence.text;
  const tdVoice = document.createElement("td"); fillChips(tdVoice, voiceScores);
  const tdWords = document.createElement("td"); fillChips(tdWords, textScores);
  tr.append(tdIdx, tdTime, tdText, tdVoice, tdWords);
  tbody.appendChild(tr);
}
function fillChips(td, scores) {
  if (scores && scores[0]) { td.appendChild(chip(scores[0])); scores.slice(1, 2).forEach((s) => td.appendChild(chip(s, true))); }
  else td.textContent = "–";
}
function chip(item, secondary = false) {
  const c = document.createElement("span");
  c.className = "chip" + (secondary ? " secondary" : "");
  const dot = document.createElement("span");
  Object.assign(dot.style, { width: "9px", height: "9px", borderRadius: "50%", background: emoColor(item.label) });
  c.appendChild(dot); c.appendChild(document.createTextNode(item.label));
  const sc = document.createElement("span"); sc.className = "score"; sc.textContent = `${Math.round(item.score * 100)}%`;
  c.appendChild(sc); return c;
}

// ---- Storyboard + footage ----------------------------------------------
function sceneItems(sc) { return sc.kind === "video" ? sc.clips : sc.photos; }
function renderStoryboard() {
  storyboardEl.innerHTML = "";
  scenes.forEach((sc, i) => {
    const items = sceneItems(sc) || [];
    const card = document.createElement("div");
    card.className = "scene-card";
    card.title = "Click to swap";
    if (sc.overlayImg) {
      const im = document.createElement("img"); im.src = sc.overlayImg.src; card.appendChild(im);
    } else if (items[sc.idx] && items[sc.idx].thumb) {
      const im = document.createElement("img"); im.src = items[sc.idx].thumb; card.appendChild(im);
    } else {
      const ph = document.createElement("div"); ph.className = "ph"; ph.textContent = sc.query || "(no query)"; card.appendChild(ph);
    }
    const tag = sc.overlayImg ? "⭐ overlay" : sc.userAssigned ? "⭐ yours" : sc.kind === "video" ? "🎬" : "🖼";
    if (sc.overlayImg || sc.userAssigned) {
      const sw = document.createElement("div"); sw.className = "swap"; sw.textContent = tag; card.appendChild(sw);
    } else if (items.length > 1) {
      const sw = document.createElement("div"); sw.className = "swap"; sw.textContent = `${tag} ${sc.idx + 1}/${items.length}`; card.appendChild(sw);
    }
    const cap = document.createElement("div"); cap.className = "cap";
    cap.innerHTML = `<div>${escapeHtml(sc.text.slice(0, 60))}${sc.text.length > 60 ? "…" : ""}</div><div class="q">🔎 ${escapeHtml(sc.query)} · ${escapeHtml(sc.emotion)}</div>`;
    card.appendChild(cap);
    card.addEventListener("click", () => swapMedia(i));
    storyboardEl.appendChild(card);
  });
}
async function swapMedia(i) {
  const sc = scenes[i];
  const items = sceneItems(sc) || [];
  if (items.length <= 1) return;
  sc.idx = (sc.idx + 1) % items.length;
  try {
    if (sc.kind === "video") {
      disposeSceneMedia(sc);
      sc.posterImg = await loadImage(items[sc.idx].thumb).catch(() => sc.posterImg);
      sc.videoEl = await loadVideo(items[sc.idx].file);
      if (activeVid === i) activeVid = -1; // force re-sync on next loop
    } else {
      sc.img = await loadImage(items[sc.idx].full);
    }
  } catch (e) { console.warn("swap failed", e); }
  renderStoryboard();
  if (!videoPanel.hidden) renderFrame(player.currentTime || 0);
}

async function findFootage() {
  if (!scenes.length) { setStatus("Analyze an audio/video file first.", true); return; }
  if (el("reelMode").value === "talking") { return prepareTalking(); }
  const provider = providerSel.value;
  const ftype = el("footageType").value;   // "video" | "photo"
  const key = apiKeyInput.value.trim();
  const site = provider === "pixabay" ? "Pixabay (pixabay.com/api/docs)" : "Pexels (pexels.com/api)";
  if (!key) { setStatus(`Enter your free ${site} API key first.`, true); return; }
  if (!scenes.length) { setStatus("Analyze an audio file first.", true); return; }
  localStorage.setItem("stockProvider", provider);
  localStorage.setItem("stockKey_" + provider, key);
  localStorage.setItem("footageType", ftype);
  setBusy(true);
  let keyError = null, videoFellBack = 0;
  try {
    // Fresh start: clear any previous media + user-photo placements.
    const device = el("useWebGPU").checked ? "webgpu" : undefined;
    for (const sc of scenes) {
      disposeSceneMedia(sc);
      sc.userAssigned = false; sc.kind = null; sc.img = null; sc.posterImg = null; sc.photos = []; sc.clips = []; sc.idx = 0;
    }

    // 1) Place the user's own photos onto the scenes they match best (CLIP).
    let placedUser = 0;
    if (userPhotos.length) {
      setStatus("Matching your photos to the story (CLIP)…");
      placedUser = await placeUserPhotos(device);
    }

    // 2) Fill the remaining scenes with stock footage.
    for (let i = 0; i < scenes.length; i++) {
      const sc = scenes[i];
      if (sc.userAssigned) continue;   // keep the user's photo here
      setStatus(`Finding ${ftype === "video" ? "clips" : "photos"} ${i + 1}/${scenes.length}: “${sc.query}”…`);
      showProgress((i + 1) / scenes.length);
      try {
        if (ftype === "video") {
          let clips = await stockVideoSearch(sc.query, key, provider);
          if (!clips.length) clips = await stockVideoSearch(MOOD_WORD[sc.emotion] || sc.emotion || "abstract", key, provider);
          clips = clips.map((c) => ({ ...c, _s: scorePhoto(c, sc.keywords || []) })).sort((a, b) => b._s - a._s);
          sc.clips = clips;
          if (clips.length) {
            sc.posterImg = await loadImage(clips[0].thumb).catch(() => null);
            try { sc.videoEl = await loadVideo(clips[0].file); sc.kind = "video"; }
            catch (ve) { videoFellBack++; await fetchPhotoForScene(sc, key, provider); }
          }
        } else {
          await fetchPhotoForScene(sc, key, provider);
        }
      } catch (err) {
        console.warn("footage fail", i, err);
        const s = String(err);
        if (/\b(400|401|403|429)\b/.test(s)) { keyError = s; break; }
      }
    }
    if (keyError) {
      const why = keyError.includes("429") ? "rate limit hit (429) — wait a minute and retry" : "key rejected — check it's the right provider's key";
      setStatus(`${provider === "pixabay" ? "Pixabay" : "Pexels"} ${why}.`, true);
      showProgress(null); return;
    }
    renderStoryboard();
    videoPanel.hidden = false;
    setupStageSize();
    renderFrame(0);
    const missing = scenes.filter((s) => !s.img && !s.videoEl).length;
    let msg = "Footage ready. Preview or render the video below.";
    if (placedUser) msg = `Placed ${placedUser} of your photo(s) into best-matching scenes; rest filled with stock. Preview or render below.`;
    if (videoFellBack) msg += ` (${videoFellBack} clip(s) unavailable via browser — used photos; try Pixabay for video.)`;
    if (missing) msg = `Footage ready (${missing} scene(s) had no match — color card). Preview or render below.`;
    setStatus(msg);
    showProgress(null);
  } finally { setBusy(false); }
}
async function fetchPhotoForScene(sc, key, provider) {
  let photos = await stockSearch(sc.query, key, provider);
  if (!photos.length) photos = await stockSearch(MOOD_WORD[sc.emotion] || sc.emotion || "abstract", key, provider);
  photos = photos.map((p) => ({ ...p, _s: scorePhoto(p, sc.keywords || []) })).sort((a, b) => b._s - a._s);
  sc.photos = photos; sc.idx = 0; sc.kind = "image";
  if (photos.length) sc.img = await loadImage(photos[0].full);
}

// Load the person's uploaded video as the base layer (muted; its audio comes via the player graph).
async function ensureBaseVideo() {
  const url = originalUrl;
  if (!url) throw new Error("no video loaded");
  if (!baseVideoEl) {
    baseVideoEl = document.createElement("video");
    baseVideoEl.muted = true; baseVideoEl.playsInline = true; baseVideoEl.preload = "auto";
    baseVideoEl.style.cssText = "position:absolute;left:-9999px;width:1px;height:1px;opacity:0";
    document.body.appendChild(baseVideoEl);
  }
  if (baseVideoEl._url !== url) {
    baseVideoEl._url = url;
    await new Promise((res, rej) => {
      const ok = () => { cl(); res(); };
      const er = () => { cl(); rej(new Error("base video load failed")); };
      const to = setTimeout(er, 20000);
      function cl() { clearTimeout(to); baseVideoEl.removeEventListener("canplay", ok); baseVideoEl.removeEventListener("error", er); }
      baseVideoEl.addEventListener("canplay", ok);
      baseVideoEl.addEventListener("error", er);
      baseVideoEl.src = url; baseVideoEl.load();
    });
  }
  return baseVideoEl;
}

// Talking mode: person's video = base; user photos overlay above them, timed to matching sentences.
async function prepareTalking() {
  setBusy(true);
  try {
    const device = el("useWebGPU").checked ? "webgpu" : undefined;
    for (const sc of scenes) { sc.overlayImg = null; sc.overlayAssigned = false; }

    setStatus("Loading your video…");
    try { await ensureBaseVideo(); }
    catch (err) { setStatus("Couldn't load the video. Talking mode needs a video file as the main upload.", true); return; }
    if (!baseVideoEl.videoWidth) {
      setStatus("That file has no video track. Talking mode needs a video of the person.", true); return;
    }

    let placed = 0;
    if (userPhotos.length) {
      setStatus("Matching your images to what's being said (CLIP)…");
      placed = await placeUserPhotos(device, true);
    }

    renderStoryboard();
    storyboardPanel.hidden = false;
    videoPanel.hidden = false;
    setupStageSize();
    renderFrame(0);
    setStatus(placed
      ? `Ready — ${placed} image(s) will pop up above you when you mention them. Preview or render below.`
      : (userPhotos.length ? "Ready, but no images matched. Preview or render below."
        : "Ready. Add images above to have them appear; or just preview/render the talking video."));
    showProgress(null);
  } finally { setBusy(false); }
}

// ---- Canvas rendering --------------------------------------------------
function setupStageSize() {
  const [w, h] = { "9:16": [1080, 1920], "1:1": [1080, 1080], "16:9": [1920, 1080] }[el("aspect").value] || [1080, 1920];
  stage.width = w; stage.height = h;
}
function sceneAt(t) {
  let found = scenes[0];
  for (const s of scenes) { if (s.start != null && s.start <= t) found = s; else if (s.start > t) break; }
  return found;
}
function hexToRgba(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}
function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }
function easeOutBack(x) { return 1 + 2.70158 * Math.pow(x - 1, 3) + 1.70158 * Math.pow(x - 1, 2); }
function contrastColor(hex) {
  const n = parseInt(hex.slice(1), 16);
  const lum = (0.299 * ((n >> 16) & 255) + 0.587 * ((n >> 8) & 255) + 0.114 * (n & 255)) / 255;
  return lum > 0.6 ? "#000000" : "#ffffff";
}
function roundRectPath(c, x, y, w, h, r) {
  r = Math.min(r, h / 2, w / 2);
  c.beginPath();
  c.moveTo(x + r, y);
  c.arcTo(x + w, y, x + w, y + h, r);
  c.arcTo(x + w, y + h, x, y + h, r);
  c.arcTo(x, y + h, x, y, r);
  c.arcTo(x, y, x + w, y, r);
  c.closePath();
}

// Per-emotion split-tone grade: [shadow tint, highlight tint].
const GRADE = {
  joy: ["#241400", "#ffeeb8"], happy: ["#241400", "#ffeeb8"], amusement: ["#241400", "#ffeeb8"],
  excitement: ["#241000", "#ffd9a0"], optimism: ["#241400", "#ffeeb8"], pride: ["#241000", "#ffd9a0"],
  love: ["#2a0016", "#ffc6dd"], desire: ["#2a0016", "#ffc6dd"], caring: ["#2a0014", "#ffd0e0"],
  gratitude: ["#0e2010", "#c8f0cc"], approval: ["#0e2010", "#c8f0cc"], relief: ["#06201c", "#bfeee0"], calm: ["#06201c", "#bfeee0"],
  sadness: ["#0b1a3a", "#a8c4ff"], sad: ["#0b1a3a", "#a8c4ff"], grief: ["#0a1430", "#90b0ff"],
  disappointment: ["#101c33", "#b4c8f0"], remorse: ["#101c33", "#b4c8f0"], nervousness: ["#140a2a", "#c9b0ff"],
  anger: ["#2a0008", "#ffb0b0"], angry: ["#2a0008", "#ffb0b0"], annoyance: ["#280a00", "#ffc2a0"], disapproval: ["#200a04", "#f0bba0"],
  fear: ["#140a2a", "#c9b0ff"], fearful: ["#140a2a", "#c9b0ff"],
  disgust: ["#101c00", "#d6f0a0"], disgusted: ["#101c00", "#d6f0a0"],
  surprise: ["#06201c", "#bfeee0"], surprised: ["#06201c", "#bfeee0"], curiosity: ["#061c24", "#b0e4f0"], realization: ["#061c24", "#b0e4f0"],
  neutral: ["#0c1020", "#dfe6ff"],
};
const grade = (e) => GRADE[e?.toLowerCase()] || GRADE.neutral;

const CAP_FONT = '"Arial Black", "Montserrat", Impact, system-ui, sans-serif';

// Draw one scene's visuals (video frame or photo + motion + grade + vignette) to any ctx.
function drawSceneVisual(c, cw, ch, sc, t, extraZoom) {
  const dur = Math.max(0.1, (sc.end ?? t + 2) - (sc.start ?? t));
  const p = clamp((t - (sc.start ?? t)) / dur, 0, 1);
  const energy = emoArousal(sc.emotion);
  const isVideo = sc.kind === "video" && sc.videoEl && sc.videoEl.readyState >= 2;
  // Video already moves, so use a gentler Ken Burns than for stills.
  const amp = (isVideo ? 0.02 : 0.04) + (isVideo ? 0.04 : 0.11) * energy;
  const zoom = (isVideo ? 1.03 : 1.05) + amp * p + (extraZoom || 0);
  const dir = scenes.indexOf(sc) % 2 === 0 ? 1 : -1;
  const freq = 0.5 + energy * 1.1;
  const driftX = Math.sin(t * freq) * cw * 0.012 * (0.6 + energy);
  const driftY = Math.cos(t * freq * 0.75) * ch * 0.008 * (0.6 + energy);
  const dx = dir * (cw * 0.03 + cw * 0.03 * energy) * (p - 0.5) + driftX;
  const dy = (ch * 0.03) * (p - 0.5) + driftY;

  let src = null, mw = 0, mh = 0;
  if (isVideo) { src = sc.videoEl; mw = sc.videoEl.videoWidth; mh = sc.videoEl.videoHeight; }
  else if (sc.img) { src = sc.img; mw = sc.img.naturalWidth; mh = sc.img.naturalHeight; }
  else if (sc.posterImg) { src = sc.posterImg; mw = sc.posterImg.naturalWidth; mh = sc.posterImg.naturalHeight; }

  if (src && mw && mh) {
    const scale = Math.max(cw / mw, ch / mh) * zoom;
    const w = mw * scale, h = mh * scale;
    try { c.drawImage(src, (cw - w) / 2 + dx, (ch - h) / 2 + dy, w, h); }
    catch { c.fillStyle = emoColor(sc.emotion); c.fillRect(0, 0, cw, ch); }
  } else {
    c.fillStyle = emoColor(sc.emotion); c.fillRect(0, 0, cw, ch);
  }

  // Split-tone grade: shadows (multiply), highlights (screen), emotion wash (soft-light).
  const g = grade(sc.emotion);
  c.globalCompositeOperation = "multiply"; c.globalAlpha = 0.30; c.fillStyle = g[0]; c.fillRect(0, 0, cw, ch);
  c.globalCompositeOperation = "screen"; c.globalAlpha = 0.18; c.fillStyle = g[1]; c.fillRect(0, 0, cw, ch);
  c.globalCompositeOperation = "soft-light"; c.globalAlpha = 0.22; c.fillStyle = emoColor(sc.emotion); c.fillRect(0, 0, cw, ch);
  c.globalCompositeOperation = "source-over"; c.globalAlpha = 1;

  const vg = c.createRadialGradient(cw / 2, ch / 2, ch * 0.3, cw / 2, ch / 2, ch * 0.78);
  vg.addColorStop(0, "rgba(0,0,0,0)"); vg.addColorStop(1, "rgba(0,0,0,0.55)");
  c.fillStyle = vg; c.fillRect(0, 0, cw, ch);
}

// Offscreen canvas for crossfades.
let offCanvas = null, offCtx = null;
function getOff(cw, ch) {
  if (!offCanvas) { offCanvas = document.createElement("canvas"); offCtx = offCanvas.getContext("2d"); }
  if (offCanvas.width !== cw || offCanvas.height !== ch) { offCanvas.width = cw; offCanvas.height = ch; }
  return offCtx;
}
function currentIndex(t) {
  let idx = 0;
  for (let i = 0; i < scenes.length; i++) { if (scenes[i].start != null && scenes[i].start <= t) idx = i; else if (scenes[i].start > t) break; }
  return idx;
}
function transitionType(sc) {
  const e = (sc.emotion || "").toLowerCase();
  if (["anger", "angry", "fear", "fearful", "nervousness"].includes(e)) return "punch";
  if (["surprise", "surprised", "excitement", "joy", "happy", "amusement"].includes(e)) return "dipwhite";
  if (["sadness", "sad", "grief", "remorse", "disappointment", "disgust", "disgusted"].includes(e)) return "dipblack";
  return "crossfade";
}
function renderFrame(t) {
  const cw = stage.width, ch = stage.height;
  ctx.fillStyle = "#000"; ctx.fillRect(0, 0, cw, ch);
  if (el("reelMode").value === "talking") return drawTalkingFrame(t, cw, ch);
  if (!scenes.length) return;
  const i = currentIndex(t);
  const sc = scenes[i];
  const into = t - (sc.start ?? 0);
  const type = transitionType(sc);
  const DUR = type === "punch" ? 0.22 : 0.3;

  if (i > 0 && into < DUR) {
    const k = clamp(into / DUR, 0, 1);
    if (type === "punch") {
      drawSceneVisual(ctx, cw, ch, sc, t, 0.12 * (1 - k));         // zoom-punch, hard cut
    } else if (type === "crossfade") {
      drawSceneVisual(ctx, cw, ch, sc, t, 0);
      const oc = getOff(cw, ch);
      oc.clearRect(0, 0, cw, ch);
      drawSceneVisual(oc, cw, ch, scenes[i - 1], t, 0);
      ctx.save(); ctx.globalAlpha = 1 - k; ctx.drawImage(offCanvas, 0, 0); ctx.restore();
    } else {
      // dip: flash from white (upbeat) or black (somber) revealing the new scene
      drawSceneVisual(ctx, cw, ch, sc, t, 0.04 * (1 - k));
      ctx.save(); ctx.globalAlpha = (1 - k) * 0.9;
      ctx.fillStyle = type === "dipwhite" ? "#ffffff" : "#000000";
      ctx.fillRect(0, 0, cw, ch); ctx.restore();
    }
  } else {
    drawSceneVisual(ctx, cw, ch, sc, t, 0);
  }
  drawCaptions(sc, t);
}

// Talking mode: draw the person's video full-frame, image overlays above them, small captions.
function drawTalkingFrame(t, cw, ch) {
  if (baseVideoEl && baseVideoEl.readyState >= 2) {
    const mw = baseVideoEl.videoWidth, mh = baseVideoEl.videoHeight;
    const scale = Math.max(cw / mw, ch / mh);
    const w = mw * scale, h = mh * scale;
    try { ctx.drawImage(baseVideoEl, (cw - w) / 2, (ch - h) / 2, w, h); }
    catch { ctx.fillStyle = "#111"; ctx.fillRect(0, 0, cw, ch); }
  } else { ctx.fillStyle = "#111"; ctx.fillRect(0, 0, cw, ch); }

  if (!scenes.length) return;
  const sc = scenes[currentIndex(t)];
  if (sc && sc.overlayImg) drawOverlay(sc, t, cw, ch);
  if (sc) drawCaptions(sc, t);
}
// Draw one user image as a framed card in the upper area, popping in/out over its sentence.
function drawOverlay(sc, t, cw, ch) {
  const img = sc.overlayImg;
  if (!img || !img.naturalWidth) return;
  const start = sc.start ?? 0, end = Math.max((sc.end ?? start + 3), start + 1.2);
  if (t < start - 0.05 || t > end + 0.25) return;
  const inK = clamp((t - start) / 0.28, 0, 1);
  const outK = clamp((end + 0.25 - t) / 0.28, 0, 1);
  const pop = easeOutBack(inK);
  const fade = clamp(Math.min(inK, outK), 0, 1);

  const boxW = cw * 0.82, boxH = ch * 0.40;
  const s = Math.min(boxW / img.naturalWidth, boxH / img.naturalHeight);
  const w = img.naturalWidth * s, h = img.naturalHeight * s;
  const cx = cw / 2, cy = ch * 0.08 + h / 2;
  const pad = cw * 0.012, r = cw * 0.03;

  ctx.save();
  ctx.globalAlpha = fade;
  ctx.translate(cx, cy);
  ctx.scale(pop, pop);
  ctx.shadowColor = "rgba(0,0,0,0.5)"; ctx.shadowBlur = cw * 0.03; ctx.shadowOffsetY = cw * 0.012;
  ctx.fillStyle = "#ffffff";
  roundRectPath(ctx, -w / 2 - pad, -h / 2 - pad, w + 2 * pad, h + 2 * pad, r);
  ctx.fill();
  ctx.shadowColor = "transparent"; ctx.shadowBlur = 0; ctx.shadowOffsetY = 0;
  ctx.save();
  roundRectPath(ctx, -w / 2, -h / 2, w, h, r * 0.7);
  ctx.clip();
  ctx.drawImage(img, -w / 2, -h / 2, w, h);
  ctx.restore();
  ctx.restore();
}

// Group a scene's words into 1-3 word chunks (cached), closing gaps.
function buildCapGroups(sc) {
  const words = (sc.words && sc.words.length ? sc.words : [{ text: sc.text, start: sc.start, end: sc.end }]).filter((w) => w.text);
  const groups = [];
  let g = null;
  for (const w of words) {
    if (!g) g = { words: [], start: w.start, end: w.end };
    const chars = g.words.reduce((a, x) => a + x.text.length, 0) + w.text.length;
    if (g.words.length >= 3 || (g.words.length && chars > 15)) { groups.push(g); g = { words: [], start: w.start, end: w.end }; }
    g.words.push(w);
    if (w.start != null && (g.start == null || w.start < g.start)) g.start = w.start;
    if (w.end != null) g.end = w.end;
  }
  if (g && g.words.length) groups.push(g);
  for (let i = 0; i < groups.length; i++) {
    if (groups[i].end == null) groups[i].end = (groups[i].start ?? 0) + 1;
    const next = groups[i + 1];
    if (next && next.start != null) groups[i].end = Math.max(groups[i].end, next.start);
    if (groups[i].end - (groups[i].start ?? 0) < 0.5) groups[i].end = (groups[i].start ?? 0) + 0.5;
  }
  if (groups[0]) groups[0].start = Math.min(groups[0].start ?? (sc.start ?? 0), sc.start ?? (groups[0].start ?? 0));
  return groups;
}
function activeGroup(sc, t) {
  if (!sc._capGroups) sc._capGroups = buildCapGroups(sc);
  const gs = sc._capGroups;
  if (!gs.length) return null;
  let g = gs[0];
  for (const x of gs) { if (x.start != null && x.start <= t) g = x; else if (x.start > t) break; }
  return g;
}
function drawCaptions(sc, t) {
  const cw = stage.width, ch = stage.height;
  const capMode = el("captionMode").value;
  if (capMode === "off") return;
  const small = capMode === "small";
  const group = activeGroup(sc, t);
  if (!group || !group.words.length) return;
  const words = group.words.map((w) => ({ ...w, disp: (w.text || "").toUpperCase() }));

  // Fit-to-width: size the whole line to maxWidth, clamped.
  const S0 = 100;
  ctx.font = `900 ${S0}px ${CAP_FONT}`;
  const spaceW0 = ctx.measureText(" ").width;
  const line0 = words.reduce((a, w) => a + ctx.measureText(w.disp).width, 0) + spaceW0 * (words.length - 1);
  const maxWidth = cw * 0.80;
  const fontSize = small
    ? clamp((S0 * maxWidth) / Math.max(1, line0), cw * 0.03, cw * 0.052)
    : clamp((S0 * maxWidth) / Math.max(1, line0), cw * 0.085, cw * 0.14);

  ctx.font = `900 ${fontSize}px ${CAP_FONT}`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  const spaceW = ctx.measureText(" ").width;
  const wds = words.map((w) => ctx.measureText(w.disp).width);
  const totalW = wds.reduce((a, b) => a + b, 0) + spaceW * (words.length - 1);

  const is916 = Math.abs(cw / ch - 9 / 16) < 0.05;
  const cy = small ? ch * 0.90 : clamp(is916 ? ch * 0.72 : ch * 0.80, ch * 0.15, ch * 0.85);

  // Style varies by emotion energy: punchy "box" for high arousal, soft karaoke "fill" for calm.
  const energy = emoArousal(sc.emotion);
  const style = small ? "fill" : (energy >= 0.5 ? "box" : "fill");
  const popD = energy >= 0.5 ? 0.14 : 0.2;
  const emoC = emoColor(sc.emotion);

  const drawBase = (disp) => {
    ctx.shadowColor = "rgba(0,0,0,0.55)"; ctx.shadowBlur = fontSize * 0.14; ctx.shadowOffsetY = fontSize * 0.06;
    ctx.lineWidth = fontSize * 0.16; ctx.lineJoin = "round"; ctx.strokeStyle = "rgba(0,0,0,0.92)";
    ctx.strokeText(disp, 0, 0);
    ctx.shadowColor = "transparent"; ctx.shadowBlur = 0; ctx.shadowOffsetY = 0;
    ctx.fillStyle = "#ffffff"; ctx.fillText(disp, 0, 0);
  };

  let x = cw / 2 - totalW / 2;
  for (let i = 0; i < words.length; i++) {
    const w = words[i], ww = wds[i];
    const cx = x + ww / 2;
    x += ww + spaceW;
    const isActive = w.start != null && w.end != null && t >= w.start - 0.05 && t <= w.end + 0.05;
    const spoken = w.end != null && t > w.end;

    if (style === "box") {
      // Words pop in as spoken; the active word sits in a filled box.
      const appearT = w.start ?? group.start ?? t;
      const age = t - appearT;
      if (age < 0) continue;
      const s = easeOutBack(clamp(age / popD, 0, 1));
      ctx.save();
      ctx.globalAlpha = clamp(age / 0.08, 0, 1);
      ctx.translate(cx, cy);
      ctx.scale(s, s);
      if (isActive) {
        const padX = fontSize * 0.22, padY = fontSize * 0.14, r = fontSize * 0.22;
        ctx.fillStyle = emoC;
        roundRectPath(ctx, -ww / 2 - padX, -fontSize * 0.55 - padY, ww + 2 * padX, fontSize + 2 * padY, r);
        ctx.fill();
        ctx.fillStyle = contrastColor(emoC);
        ctx.fillText(w.disp, 0, 0);
      } else {
        drawBase(w.disp);
      }
      ctx.restore();
    } else {
      // Whole group visible; the active word fills left-to-right in the emotion color.
      const upcoming = w.start != null && t < w.start;
      ctx.save();
      ctx.globalAlpha = upcoming ? 0.45 : 1;
      ctx.translate(cx, cy);
      drawBase(w.disp);
      if (spoken) {
        ctx.fillStyle = emoC; ctx.fillText(w.disp, 0, 0);
      } else if (isActive) {
        const frac = clamp((t - w.start) / Math.max(0.08, (w.end - w.start)), 0, 1);
        ctx.save();
        ctx.beginPath(); ctx.rect(-ww / 2, -fontSize, ww * frac, fontSize * 2); ctx.clip();
        ctx.fillStyle = emoC; ctx.fillText(w.disp, 0, 0);
        ctx.restore();
      }
      ctx.restore();
    }
  }
}

// ---- Playback + record -------------------------------------------------
let rafId = null;
let activeVid = -1;
function resetVideos() {
  for (const s of scenes) if (s.videoEl) { try { s.videoEl.pause(); s.videoEl.currentTime = 0; } catch {} }
  if (baseVideoEl) { try { baseVideoEl.pause(); baseVideoEl.currentTime = 0; } catch {} }
  activeVid = -1;
}
function syncVideo(i) {
  if (activeVid === i) return;
  if (scenes[activeVid] && scenes[activeVid].videoEl) { try { scenes[activeVid].videoEl.pause(); } catch {} }
  activeVid = i;
  const el2 = scenes[i] && scenes[i].videoEl;
  if (el2) { try { el2.currentTime = 0; } catch {} const pr = el2.play(); if (pr && pr.catch) pr.catch(() => {}); }
}
function renderEnd() { return Math.min(player.duration || MAX_SECONDS, MAX_SECONDS); }
function startLoop(onDone) {
  const end = renderEnd();
  const talking = el("reelMode").value === "talking";
  const loop = () => {
    const t = player.currentTime;
    if (talking) {
      if (baseVideoEl && Math.abs(baseVideoEl.currentTime - t) > 0.25) { try { baseVideoEl.currentTime = t; } catch {} }
    } else {
      syncVideo(currentIndex(t));
    }
    renderFrame(t);
    if (player.paused || player.ended || t >= end) {
      if (t >= end && !player.paused) player.pause();
      cancelAnimationFrame(rafId); rafId = null;
      if (scenes[activeVid] && scenes[activeVid].videoEl) { try { scenes[activeVid].videoEl.pause(); } catch {} }
      if (baseVideoEl) { try { baseVideoEl.pause(); } catch {} }
      onDone && onDone(); return;
    }
    rafId = requestAnimationFrame(loop);
  };
  rafId = requestAnimationFrame(loop);
}
function ensureAudioGraph() {
  if (audioGraph) return audioGraph;
  const AC = window.AudioContext || window.webkitAudioContext;
  const actx = new AC();
  const srcNode = actx.createMediaElementSource(player);
  const dest = actx.createMediaStreamDestination();
  srcNode.connect(dest); srcNode.connect(actx.destination);
  audioGraph = { ctx: actx, dest };
  return audioGraph;
}
async function getMusicBuffer(actx) {
  if (!musicFile) return null;
  if (musicBuffer) return musicBuffer;
  musicBuffer = await actx.decodeAudioData(await musicFile.arrayBuffer());
  return musicBuffer;
}
// Merge all word [start,end] spans (with pre-roll/release) into speech intervals for ducking.
function speechIntervals() {
  const raw = [];
  for (const sc of scenes)
    for (const w of sc.words || [])
      if (w.start != null && w.end != null) raw.push([Math.max(0, w.start - 0.12), w.end + 0.35]);
  raw.sort((a, b) => a[0] - b[0]);
  const merged = [];
  for (const iv of raw) {
    const last = merged[merged.length - 1];
    if (last && iv[0] <= last[1] + 0.3) last[1] = Math.max(last[1], iv[1]);
    else merged.push([iv[0], iv[1]]);
  }
  return merged;
}

async function previewPlay() {
  if (!scenes.length) return;
  setupStageSize();
  resetVideos();
  if (el("reelMode").value === "talking" && baseVideoEl) { try { baseVideoEl.currentTime = 0; await baseVideoEl.play(); } catch {} }
  const g = ensureAudioGraph();
  if (g.ctx.state === "suspended") await g.ctx.resume();
  player.currentTime = 0;
  await player.play();
  startLoop();
}

function pickMime() {
  for (const m of ["video/webm;codecs=vp9,opus", "video/webm;codecs=vp8,opus", "video/webm"])
    if (window.MediaRecorder && MediaRecorder.isTypeSupported(m)) return m;
  return "";
}

async function recordVideo() {
  if (!scenes.length) return;
  setupStageSize();
  resetVideos();
  renderFrame(0);
  const g = ensureAudioGraph();
  if (g.ctx.state === "suspended") await g.ctx.resume();

  // Optional background music, mixed low and ducked under the narration.
  let musicSrc = null, musicGain = null;
  const mbuf = await getMusicBuffer(g.ctx).catch(() => null);
  if (mbuf) {
    musicSrc = g.ctx.createBufferSource(); musicSrc.buffer = mbuf; musicSrc.loop = true;
    musicGain = g.ctx.createGain(); musicGain.gain.value = 0.16;
    musicSrc.connect(musicGain); musicGain.connect(g.dest); musicGain.connect(g.ctx.destination);
  }

  const canvasStream = stage.captureStream(30);
  const mixed = new MediaStream([...canvasStream.getVideoTracks(), ...g.dest.stream.getAudioTracks()]);
  const mime = pickMime();
  const rec = new MediaRecorder(mixed, mime ? { mimeType: mime } : undefined);
  const chunks = [];
  rec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };

  const done = new Promise((resolve) => {
    rec.onstop = () => {
      lastWebmBlob = new Blob(chunks, { type: "video/webm" });
      const url = URL.createObjectURL(lastWebmBlob);
      downloadLink.href = url; downloadLink.download = "story.webm";
      downloadLink.textContent = "⬇ Download story.webm"; downloadLink.hidden = false;
      resolve();
    };
  });

  setBusy(true);
  setStatus("Recording in real time… (plays through once; keep this tab visible)");
  recordBtn.textContent = "● Recording…";
  player.currentTime = 0;
  if (el("reelMode").value === "talking" && baseVideoEl) { try { baseVideoEl.currentTime = 0; await baseVideoEl.play(); } catch {} }
  if (musicSrc) {
    const base = g.ctx.currentTime + 0.12;
    try { musicSrc.start(); } catch {}
    // Duck music down during speech, swell back up in the gaps (audio-thread, no frame cost).
    const DUCK = 0.05, SWELL = 0.16;
    for (const [s, e] of speechIntervals()) {
      musicGain.gain.setTargetAtTime(DUCK, base + s, 0.08);
      musicGain.gain.setTargetAtTime(SWELL, base + e, 0.25);
    }
  }
  await player.play();
  rec.start(200);
  startLoop(() => { try { rec.stop(); } catch {} if (musicSrc) try { musicSrc.stop(); } catch {} });
  await done;
  recordBtn.textContent = "● Render & download (WebM)";
  setStatus("Done. WebM downloaded. Use “Convert last render to MP4” for an Instagram-friendly file.");
  setBusy(false);
}

async function convertToMp4() {
  if (!lastWebmBlob) return;
  setBusy(true);
  try {
    setStatus("Loading FFmpeg…"); const ff = await loadFFmpeg();
    await ff.writeFile("in.webm", await ffmpegFetchFile(lastWebmBlob));
    setStatus("Converting to MP4 (H.264 / AAC)…");
    await ff.exec(["-i", "in.webm", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "23", "-c:a", "aac", "-b:a", "160k", "out.mp4"]);
    const data = await ff.readFile("out.mp4");
    await ff.deleteFile("in.webm").catch(() => {});
    await ff.deleteFile("out.mp4").catch(() => {});
    const blob = new Blob([data.slice().buffer], { type: "video/mp4" });
    const url = URL.createObjectURL(blob);
    downloadLink.href = url; downloadLink.download = "story.mp4";
    downloadLink.textContent = "⬇ Download story.mp4"; downloadLink.hidden = false;
    setStatus("MP4 ready — downloaded below.");
    showProgress(null);
  } catch (err) { console.error(err); setStatus(`MP4 convert failed: ${err.message || err}`, true); }
  finally { setBusy(false); }
}

// ---- UI wiring ---------------------------------------------------------
function setBusy(b) {
  analyzeBtn.disabled = b || !selectedFile;
  footageBtn.disabled = b || scenes.length === 0;
  renderBtn.disabled = b || scenes.length === 0;
  playBtn.disabled = b;
  recordBtn.disabled = b;
  mp4Btn.disabled = b || !lastWebmBlob;
}
function onFile(file) {
  if (!file) return;
  selectedFile = file;
  fileLabel.textContent = file.name;
  analyzeBtn.disabled = false;
  if (denoisedUrl) { URL.revokeObjectURL(denoisedUrl); denoisedUrl = null; }
  originalUrl = URL.createObjectURL(file);
  player.hidden = false; player.src = originalUrl;
  setStatus("");
}

audioInput.addEventListener("change", (e) => onFile(e.target.files[0]));
musicInput.addEventListener("change", (e) => { musicFile = e.target.files[0] || null; musicBuffer = null; });
userPhotosInput.addEventListener("change", async (e) => {
  userPhotos.forEach((p) => URL.revokeObjectURL(p.url));
  userPhotos = [];
  for (const f of [...e.target.files]) {
    const url = URL.createObjectURL(f);
    try { userPhotos.push({ img: await loadImage(url), url, name: f.name }); }
    catch { URL.revokeObjectURL(url); }
  }
  setStatus(userPhotos.length
    ? `${userPhotos.length} photo(s) ready — they'll be auto-placed into the best-matching scenes when you click “Find footage”.`
    : "");
});
analyzeBtn.addEventListener("click", analyze);
footageBtn.addEventListener("click", findFootage);
renderBtn.addEventListener("click", () => { videoPanel.hidden = false; setupStageSize(); renderFrame(0); videoPanel.scrollIntoView({ behavior: "smooth" }); });
playBtn.addEventListener("click", previewPlay);
recordBtn.addEventListener("click", recordVideo);
mp4Btn.addEventListener("click", convertToMp4);
el("aspect").addEventListener("change", () => { if (!videoPanel.hidden) { setupStageSize(); renderFrame(player.currentTime || 0); } });
el("reelMode").addEventListener("change", () => {
  const talking = el("reelMode").value === "talking";
  footageBtn.textContent = talking ? "2 · Place images" : "2 · Find footage";
  el("captionMode").value = talking ? "small" : "big";
  if (!videoPanel.hidden) renderFrame(player.currentTime || 0);
});
el("captionMode").addEventListener("change", () => { if (!videoPanel.hidden) renderFrame(player.currentTime || 0); });

// Restore saved provider + its key, and swap the key field when provider changes.
const savedProvider = localStorage.getItem("stockProvider");
if (savedProvider) providerSel.value = savedProvider;
const savedFootage = localStorage.getItem("footageType");
if (savedFootage) el("footageType").value = savedFootage;
function loadKeyForProvider() {
  apiKeyInput.value = localStorage.getItem("stockKey_" + providerSel.value) || "";
}
loadKeyForProvider();
providerSel.addEventListener("change", loadKeyForProvider);

["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("dragover"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("dragover"); }));
drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) { audioInput.files = e.dataTransfer.files; onFile(f); } });
