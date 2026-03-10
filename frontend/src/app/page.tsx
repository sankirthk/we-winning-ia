"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  FileText,
  Loader2,
  Mic,
  Play,
  Square,
  UploadCloud,
  Volume2,
} from "lucide-react";
import { PCMPlayer } from "../lib/pcm-player";
import { MicStreamer } from "../lib/mic-stream";

type JobState = "idle" | "uploading" | "processing" | "done" | "error";

function InviteGate({ onVerified }: { onVerified: (token: string | null) => void }) {
  const [input, setInput] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError("");

    try {
      const API_BASE_AUTH = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8080/api";
      const res = await fetch(`${API_BASE_AUTH}/auth`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code: input }),
      });
      const data = await res.json();
      if (data.valid) {
        localStorage.setItem("nrtfm_verified", "1");
        if (data.token) localStorage.setItem("nrtfm_token", data.token);
        onVerified(data.token ?? null);
      } else {
        setError(data.error ?? "Invalid invite code.");
      }
    } catch {
      setError("Something went wrong. Try again.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-black text-white flex items-center justify-center">
      <div className="w-full max-w-sm rounded-3xl border border-zinc-800 bg-zinc-950 p-8 shadow-2xl">
        <h1 className="text-2xl font-semibold tracking-tight mb-1">NeverRTFM</h1>
        <p className="text-sm text-zinc-400 mb-8">Enter your invite code to continue.</p>
        <form onSubmit={submit} className="space-y-4">
          <input
            type="text"
            value={input}
            onChange={(e) => { setInput(e.target.value); setError(""); }}
            placeholder="Invite code"
            autoFocus
            disabled={loading}
            className="w-full rounded-2xl border border-zinc-700 bg-zinc-900 px-4 py-3 text-sm outline-none placeholder:text-zinc-500 focus:border-zinc-500 disabled:opacity-50"
          />
          {error && <p className="text-xs text-red-400">{error}</p>}
          <button
            type="submit"
            disabled={loading || !input}
            className="inline-flex w-full items-center justify-center gap-2 rounded-2xl bg-white px-4 py-3 text-sm font-medium text-black hover:opacity-90 transition disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading && <Loader2 className="h-4 w-4 animate-spin" />}
            Continue
          </button>
        </form>
      </div>
    </main>
  );
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8080/api";
const MAX_PDF_PAGES = 20;

function wsUrlForJob(jobId: string, token: string | null) {
  const url = new URL(API_BASE);
  const wsProtocol = url.protocol === "https:" ? "wss:" : "ws:";
  const base = `${wsProtocol}//${url.host}/api/live/${jobId}`;
  return token ? `${base}?token=${encodeURIComponent(token)}` : base;
}

export default function Home() {
  const [verified, setVerified] = useState<boolean | null>(null); // null = hydrating
  const [token, setToken] = useState<string | null>(null);

  useEffect(() => {
    const ok = localStorage.getItem("nrtfm_verified") === "1";
    setVerified(ok);
    if (ok) setToken(localStorage.getItem("nrtfm_token"));
  }, []);

  if (verified === null) return null;
  if (!verified) return (
    <InviteGate onVerified={(t) => { setVerified(true); setToken(t ?? ""); }} />
  );
  return <App token={token} />;
}

function App({ token }: { token: string | null }) {
  const authHeader: Record<string, string> = token ? { Authorization: `Bearer ${token}` } : {};
  const [file, setFile] = useState<File | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  const [tone, setTone] = useState<"formal" | "explanatory" | "casual">("explanatory");
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobState, setJobState] = useState<JobState>("idle");
  const [statusText, setStatusText] = useState("Upload a PDF to begin.");
  const [videoUrl, setVideoUrl] = useState<string | null>(null);

  const [liveConnected, setLiveConnected] = useState(false);
  const [isMicActive, setIsMicActive] = useState(false);
  const [isUserSpeaking, setIsUserSpeaking] = useState(false);
  const [isAgentSpeaking, setIsAgentSpeaking] = useState(false);

  const [question, setQuestion] = useState("");

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const pollRef = useRef<number | null>(null);
  const micRef = useRef<MicStreamer | null>(null);
  const resumeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Ref mirrors for use inside stale closures (ws.onmessage)
  const isUserSpeakingRef = useRef(false);

  const player = useMemo(() => new PCMPlayer(24000), []);

  async function handleUpload() {
    if (!file) return;

    setJobState("uploading");
    setStatusText("Uploading PDF...");
    setVideoUrl(null);
    setJobId(null);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("tone", tone);

    try {
      const res = await fetch(`${API_BASE}/generate`, {
        method: "POST",
        headers: authHeader,
        body: formData,
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail ?? `Upload failed: ${res.status}`);
      }

      const data = await res.json();
      setJobId(data.job_id);
      setJobState("processing");
      setStatusText("Processing report and generating video...");
    } catch (err) {
      console.error(err);
      setJobState("error");
      setStatusText(err instanceof Error ? err.message : "Upload failed.");
    }
  }

  useEffect(() => {
    if (!jobId || jobState !== "processing") return;

    const poll = async () => {
      try {
        const res = await fetch(`${API_BASE}/status/${jobId}`, { headers: authHeader });
        if (!res.ok) return;

        const data = await res.json();

        if (data.status === "done") {
          setJobState("done");
          setStatusText("Generation complete.");
          if (data.video_url) {
            setVideoUrl(data.video_url);
          } else {
            setVideoUrl("https://www.w3schools.com/html/mov_bbb.mp4");
          }

          if (pollRef.current) {
            window.clearInterval(pollRef.current);
            pollRef.current = null;
          }
        } else if (data.status === "error") {
          setJobState("error");
          setStatusText(data.error || "Generation failed.");

          if (pollRef.current) {
            window.clearInterval(pollRef.current);
            pollRef.current = null;
          }
        } else {
          setStatusText(`Pipeline running: ${data.step ?? "processing"}...`);
        }
      } catch (err) {
        console.error(err);
      }
    };

    void poll();
    pollRef.current = window.setInterval(poll, 10000);

    return () => {
      if (pollRef.current) {
        window.clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [jobId, jobState]);

  async function connectLive() {
    if (!jobId || wsRef.current) return;

    const ws = new WebSocket(wsUrlForJob(jobId, token));
    wsRef.current = ws;

    ws.onopen = () => {
      setLiveConnected(true);
      setStatusText("Live agent connected.");
    };

    ws.onclose = () => {
      setLiveConnected(false);
      setIsMicActive(false);
      setIsUserSpeaking(false);
      isUserSpeakingRef.current = false;
      setIsAgentSpeaking(false);
      wsRef.current = null;
    };

    ws.onerror = (err) => {
      console.error("WebSocket error", err);
    };

    ws.onmessage = async (event) => {
      const msg = JSON.parse(event.data);

      if (msg.type === "pause_video") {
        videoRef.current?.pause();
      }

      if (msg.type === "resume_video") {
        // Cancel any pending resume
        if (resumeTimerRef.current) clearTimeout(resumeTimerRef.current);

        // Poll until PCMPlayer drains, then resume with a short grace period
        const waitForDrain = () => {
          const remaining = player.remainingMs();
          if (remaining > 50) {
            resumeTimerRef.current = setTimeout(waitForDrain, Math.min(remaining / 2, 300));
          } else {
            resumeTimerRef.current = setTimeout(() => {
              setIsAgentSpeaking(false);
              void videoRef.current?.play().catch(() => {});
            }, 400);
          }
        };
        waitForDrain();
      }

      if (msg.type === "audio") {
        // Cancel any pending resume — more audio is still arriving
        if (resumeTimerRef.current) {
          clearTimeout(resumeTimerRef.current);
          resumeTimerRef.current = null;
        }
        if (!isUserSpeakingRef.current) {
          setIsAgentSpeaking(true);
          void player.playChunk(msg.data_b64);
        }
      }

      if (msg.type === "error") {
        console.error("Live agent error:", msg.message);
        setStatusText(`Live error: ${msg.message}`);
      }
    };
  }

  async function sendQuestion() {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN || !question.trim()) return;

    ws.send(JSON.stringify({ type: "text", text: question }));
    setQuestion("");
  }

  async function startMic() {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (micRef.current) return;

    const mic = new MicStreamer({
      onPcmChunk: (chunk) => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(chunk);
        }
      },
      onSpeechStart: () => {
        isUserSpeakingRef.current = true;
        setIsUserSpeaking(true);
        setIsAgentSpeaking(false);
        if (resumeTimerRef.current) clearTimeout(resumeTimerRef.current);
        player.stop();
        if (videoRef.current) {
          videoRef.current.pause();
        }
      },
      outputSampleRate: 16000,
    });

    await mic.start();
    micRef.current = mic;
    setIsMicActive(true);
    setStatusText("Microphone live. Start speaking.");
  }

  async function stopMic() {
    if (micRef.current) {
      await micRef.current.stop();
      micRef.current = null;
    }

    isUserSpeakingRef.current = false;
    setIsMicActive(false);
    setIsUserSpeaking(false);

    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "end_turn" }));
    }
  }

  useEffect(() => {
    return () => {
      if (pollRef.current) window.clearInterval(pollRef.current);
      if (wsRef.current) wsRef.current.close();
      if (resumeTimerRef.current) clearTimeout(resumeTimerRef.current);
      void player.close();
      void micRef.current?.stop();
    };
  }, [player]);

  return (
    <main className="min-h-screen bg-black text-white">
      <div className="mx-auto max-w-7xl px-6 py-8">
        <div className="mb-10 flex items-center justify-between">
          <div>
            <h1 className="text-4xl font-semibold tracking-tight">NeverRTFM</h1>
            <p className="mt-2 text-sm text-zinc-400">
              Upload any report. Get a short video. Ask questions live.
            </p>
          </div>
          <div className="rounded-full border border-zinc-800 bg-zinc-900 px-4 py-2 text-sm text-zinc-300">
            {statusText}
          </div>
        </div>

        <div className="grid gap-8 lg:grid-cols-[460px_minmax(0,1fr)]">
          <section className="flex flex-col rounded-3xl border border-zinc-800 bg-zinc-950 p-6 shadow-2xl">
            <div className="mb-5 flex items-center gap-3">
              <div className="rounded-2xl bg-zinc-900 p-3">
                <UploadCloud className="h-5 w-5" />
              </div>
              <div>
                <h2 className="text-xl font-medium">Upload report</h2>
                <p className="text-sm text-zinc-400">PDF only · max {MAX_PDF_PAGES} pages</p>
              </div>
            </div>

            <label className="flex cursor-pointer flex-col items-center justify-center rounded-3xl border border-dashed border-zinc-700 bg-zinc-900/60 px-6 py-10 text-center hover:border-zinc-500">
              <FileText className="mb-3 h-8 w-8 text-zinc-300" />
              <span className="font-medium">Choose a PDF</span>
              <span className="mt-1 text-sm text-zinc-400">
                Drag and drop or browse
              </span>
              <input
                type="file"
                accept="application/pdf"
                className="hidden"
                onChange={async (e) => {
                  setFileError(null);
                  const selected = e.target.files?.[0] ?? null;
                  if (selected) {
                    const buf = await selected.arrayBuffer();
                    const text = new TextDecoder("latin1").decode(buf);
                    const pages = (text.match(/\/Type\s*\/Page[^s]/g) || []).length;
                    if (pages > MAX_PDF_PAGES) {
                      setFileError(`This PDF has ${pages} pages. The limit is ${MAX_PDF_PAGES}.`);
                      setFile(null);
                      e.target.value = "";
                      return;
                    }
                  }
                  setFile(selected);
                }}
              />
            </label>

            {fileError && (
              <p className="mt-3 text-sm text-red-400">{fileError}</p>
            )}

            {file && !fileError && (
              <div className="mt-4 rounded-2xl border border-zinc-800 bg-zinc-900 p-4 text-sm text-zinc-300">
                Selected: <span className="font-medium">{file.name}</span>
              </div>
            )}

            <div className="mt-4">
              <label className="mb-1.5 block text-xs font-medium uppercase tracking-wide text-zinc-400">
                Video tone
              </label>
              <div className="grid grid-cols-3 gap-2">
                {(["formal", "explanatory", "casual"] as const).map((t) => (
                  <button
                    key={t}
                    type="button"
                    onClick={() => setTone(t)}
                    className={`rounded-xl border py-2 text-sm font-medium capitalize transition ${
                      tone === t
                        ? "border-white bg-white text-black"
                        : "border-zinc-700 bg-zinc-900 text-zinc-300 hover:border-zinc-500"
                    }`}
                  >
                    {t === "explanatory" ? "Explain" : t === "casual" ? "Casual" : "Formal"}
                  </button>
                ))}
              </div>
              <p className="mt-1.5 text-xs text-zinc-500">
                {tone === "formal" && "Precise, professional language — executive briefing style."}
                {tone === "explanatory" && "Clear, educational — explains the why behind every number."}
                {tone === "casual" && "Conversational and direct — like a smart friend explaining it."}
              </p>
            </div>

            <button
              onClick={handleUpload}
              disabled={!file || jobState === "uploading" || jobState === "processing"}
              className="mt-5 inline-flex w-full items-center justify-center gap-2 rounded-2xl bg-white px-4 py-3 font-medium text-black transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {(jobState === "uploading" || jobState === "processing") && (
                <Loader2 className="h-4 w-4 animate-spin" />
              )}
              Generate video
            </button>

            <div className="mt-auto pt-8">
              <h3 className="mb-3 text-sm font-semibold uppercase tracking-wide text-zinc-400">
                Live Q&A
              </h3>

              <div className="space-y-3">
                <textarea
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  rows={3}
                  className="w-full rounded-2xl border border-zinc-800 bg-zinc-900 p-3 text-sm outline-none ring-0"
                  placeholder="Ask a question"
                />

                <button
                  onClick={connectLive}
                  disabled={!jobId || liveConnected === true}
                  className="inline-flex w-full items-center justify-center gap-2 rounded-2xl border border-zinc-700 bg-zinc-900 px-4 py-3 text-sm font-medium hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  <Volume2 className="h-4 w-4" />
                  {liveConnected ? "Live connected" : "Connect live agent"}
                </button>

                <button
                  onClick={sendQuestion}
                  disabled={!liveConnected || !question.trim()}
                  className="inline-flex w-full items-center justify-center gap-2 rounded-2xl border border-zinc-700 bg-zinc-900 px-4 py-3 text-sm font-medium hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  <Play className="h-4 w-4" />
                  Send question
                </button>

                {!isMicActive ? (
                  <button
                    onClick={startMic}
                    disabled={!liveConnected}
                    className="inline-flex w-full items-center justify-center gap-2 rounded-2xl bg-emerald-400 px-4 py-3 text-sm font-semibold text-black hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    <Mic className="h-4 w-4" />
                    Start talking
                  </button>
                ) : (
                  <button
                    onClick={stopMic}
                    className="inline-flex w-full items-center justify-center gap-2 rounded-2xl bg-red-400 px-4 py-3 text-sm font-semibold text-black hover:opacity-90"
                  >
                    <Square className="h-4 w-4" />
                    Stop mic / end turn
                  </button>
                )}
              </div>

              <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
                <div className="rounded-2xl border border-zinc-800 bg-zinc-900 p-3">
                  User speaking:{" "}
                  <span className={isUserSpeaking ? "text-emerald-400" : "text-zinc-400"}>
                    {isUserSpeaking ? "yes" : "no"}
                  </span>
                </div>
                <div className="rounded-2xl border border-zinc-800 bg-zinc-900 p-3">
                  Agent speaking:{" "}
                  <span className={isAgentSpeaking ? "text-sky-400" : "text-zinc-400"}>
                    {isAgentSpeaking ? "yes" : "no"}
                  </span>
                </div>
              </div>
            </div>
          </section>

          <section className="flex flex-col justify-center rounded-3xl border border-zinc-800 bg-zinc-950 p-6 shadow-2xl lg:p-10">
            <div className="mb-6 flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-medium">Generated Video</h2>
                <p className="mt-1 text-sm text-zinc-400">
                  Agent audio plays here. Speak to interrupt.
                </p>
              </div>
              <div className="rounded-full border border-zinc-800 bg-zinc-900 px-4 py-1.5 text-xs text-zinc-400">
                {jobState}
              </div>
            </div>

            <div className="overflow-hidden rounded-2xl border border-zinc-800 bg-black shadow-lg">
              {videoUrl ? (
                <video
                  ref={videoRef}
                  src={videoUrl}
                  controls
                  className="aspect-video w-full bg-black"
                />
              ) : (
                <div className="flex aspect-video items-center justify-center text-sm text-zinc-500">
                  Video will appear here after generation.
                </div>
              )}
            </div>
          </section>
        </div>
      </div>
    </main>
  );
}
