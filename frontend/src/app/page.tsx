"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { UploadCloud, FileText, Mic, Play, Pause, Loader2 } from "lucide-react";

// HTTP calls go through Next.js rewrite proxy (/api/* → backend).
// WebSocket must use the backend directly (Next.js doesn't proxy WS).
const WS_BASE =
  process.env.NEXT_PUBLIC_WS_URL ??
  (typeof window !== "undefined"
    ? `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.hostname}:8080`
    : "ws://localhost:8080");

const STEP_LABELS: Record<string, string> = {
  queued:       "Queued — waiting to start...",
  parsing:      "Reading your PDF...",
  scripting:    "Writing narration script...",
  tts:          "Generating voiceover...",
  video_script: "Writing video directions...",
  veo:          "Rendering clips with Veo (this takes a while)...",
  stitching:    "Stitching clips together...",
  complete:     "Wrapping up...",
};

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<"idle" | "processing" | "done" | "error">("idle");
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const isRecordingRef = useRef(false);
  const [pipelineStep, setPipelineStep] = useState<string>("");
  const [errorMsg, setErrorMsg] = useState<string>("");

  const videoRef = useRef<HTMLVideoElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  // Recording (mic → Gemini)
  const audioCtxRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  // Playback (Gemini audio → speaker)
  const playbackCtxRef = useRef<AudioContext | null>(null);
  const nextPlayTimeRef = useRef<number>(0);
  // Debounce flag: only send one interrupt per Gemini turn
  const interruptSentRef = useRef(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };

  // ── Upload ──────────────────────────────────────────────────────────────────
  const handleUpload = async () => {
    if (!file) return;
    setIsUploading(true);
    setJobStatus("processing");
    setPipelineStep("queued");
    setErrorMsg("");

    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch("/api/generate", { method: "POST", body: formData });
      if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
      const { job_id } = await res.json();
      setIsUploading(false); // upload done, pipeline now running
      setJobId(job_id);
    } catch (err) {
      console.error("[upload]", err);
      setErrorMsg(err instanceof Error ? err.message : "Upload failed.");
      setJobStatus("error");
      setIsUploading(false);
    }
  };

  // ── Status polling ───────────────────────────────────────────────────────────
  useEffect(() => {
    if (!jobId) return;

    const interval = setInterval(async () => {
      try {
        const res = await fetch(`/api/status/${jobId}`);
        if (!res.ok) return;
        const data = await res.json();

        if (data.status === "done") {
          setVideoUrl(data.video_url);
          setJobStatus("done");
          setIsUploading(false);
          clearInterval(interval);
        } else if (data.status === "error") {
          setJobStatus("error");
          setErrorMsg(data.error ?? "An unknown error occurred.");
          setIsUploading(false);
          clearInterval(interval);
        } else {
          setPipelineStep(data.step ?? "");
        }
      } catch (err) {
        console.error("[poll]", err);
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [jobId]);

  // ── WebSocket ────────────────────────────────────────────────────────────────
  const stopMic = useCallback(() => {
    processorRef.current?.disconnect();
    processorRef.current = null;
    sourceRef.current?.disconnect();
    sourceRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    audioCtxRef.current?.close();
    audioCtxRef.current = null;
  }, []);

  const scheduleAudioChunk = useCallback((pcm16: ArrayBuffer) => {
    // Gemini uses its own VAD — it can send audio back while mic is still open.
    // Always play incoming audio.

    if (!playbackCtxRef.current) {
      playbackCtxRef.current = new AudioContext({ sampleRate: 24000 });
      nextPlayTimeRef.current = 0;
    }
    const ctx = playbackCtxRef.current;
    
    // If context was suspended (e.g., interrupted), resume it
    if (ctx.state === "suspended") {
      ctx.resume().catch(e => console.error("Audio resume error", e));
    }

    const int16 = new Int16Array(pcm16);
    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) {
      float32[i] = int16[i] / 32768;
    }
    const buffer = ctx.createBuffer(1, float32.length, 24000);
    buffer.copyToChannel(float32, 0);
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);
    
    // Auto-advance play time if we fell behind (buffer underrun)
    const startAt = Math.max(ctx.currentTime, nextPlayTimeRef.current);
    source.start(startAt);
    nextPlayTimeRef.current = startAt + buffer.duration;
  }, []);

  const openWebSocket = useCallback(() => {
    if (wsRef.current || !jobId) return;

    const ws = new WebSocket(`${WS_BASE}/api/live/${jobId}`);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => console.log("[ws] connected");

    ws.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        // Audio chunk from Gemini — schedule for playback
        scheduleAudioChunk(event.data);
        return;
      }
      try {
        const data = JSON.parse(event.data as string);
        if (data.type === "turn_complete") {
          console.log("[ws] turn_complete received");

          // Auto-stop mic if still recording
          if (isRecordingRef.current) {
            stopMic();
            setIsRecording(false);
            isRecordingRef.current = false;
          }

          // Resume video after remaining audio finishes
          const ctx = playbackCtxRef.current;
          const delay = ctx
            ? Math.max(0, (nextPlayTimeRef.current - ctx.currentTime) * 1000 + 200)
            : 0;
          setTimeout(() => {
            videoRef.current?.play();
          }, delay);
        }
      } catch {
        // ignore parse errors
      }
    };

    ws.onclose = () => {
      wsRef.current = null;
      console.log("[ws] closed");
    };

    ws.onerror = (e) => console.error("[ws] error", e);

    wsRef.current = ws;
  }, [jobId, scheduleAudioChunk, stopMic]);

  // Clean up WS + mic on unmount
  useEffect(() => {
    return () => {
      wsRef.current?.close();
      stopMic();
    };
  }, [stopMic]);

  // ── Mic ─────────────────────────────────────────────────────────────────────
  const startMic = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    streamRef.current = stream;

    const ctx = new AudioContext({ sampleRate: 16000 });
    audioCtxRef.current = ctx;

    const source = ctx.createMediaStreamSource(stream);
    sourceRef.current = source;

    // ScriptProcessorNode: 4096 samples, mono input, mono output
    const processor = ctx.createScriptProcessor(4096, 1, 1);
    processorRef.current = processor;

    source.connect(processor);
    processor.connect(ctx.destination);

    processor.onaudioprocess = (e) => {
      const float32 = e.inputBuffer.getChannelData(0);
      const int16 = new Int16Array(float32.length);
      // Determine if there is actual voice activity (crude volume gate)
      let sumSquares = 0;
      for (let i = 0; i < float32.length; i++) {
        sumSquares += float32[i] * float32[i];
        int16[i] = Math.max(-32768, Math.min(32767, Math.round(float32[i] * 32767)));
      }
      const rms = Math.sqrt(sumSquares / float32.length);
      
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        // If the user starts talking while Gemini is playing audio,
        // signal an interrupt by clearing local playback.
        // Only fire once per turn to avoid spamming.
        if (rms > 0.15 && !interruptSentRef.current) {
            // Stop current playback to immediately silence Gemini locally
            if (playbackCtxRef.current && playbackCtxRef.current.state === "running") {
               playbackCtxRef.current.suspend();
               nextPlayTimeRef.current = 0;
               
               // Send ONE interrupt signal to backend
               interruptSentRef.current = true;
               wsRef.current.send(JSON.stringify({ type: "client_interrupt" }));
               console.log("[mic] Interrupt sent (RMS:", rms.toFixed(3), ")");
            }
        }
        wsRef.current.send(int16.buffer);
      }
    };
  };

  const handleMicToggle = async () => {
    if (isRecordingRef.current) {
      // User tapped mic to stop explicitly — send final message via backend
      // and let Gemini's turn_complete resume the video when it's done speaking.
      stopMic();
      setIsRecording(false);
      isRecordingRef.current = false;
      interruptSentRef.current = false;
      
      // Let the backend know we explicitly stopped talking so it can trigger turnaround
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: "client_turn_done" }));
      }
    } else {
      // User tapped mic to start speaking — pause video only, do NOT interrupt Gemini yet.
      // The interrupt will fire automatically via voice-activity detection if Gemini is mid-speech.
      videoRef.current?.pause();
      interruptSentRef.current = false;
      try {
        await startMic();
        setIsRecording(true);
        isRecordingRef.current = true;
      } catch (err) {
        console.error("[mic] getUserMedia failed:", err);
        videoRef.current?.play();
      }
    }
  };

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-50 font-sans selection:bg-purple-500/30">
      {/* Header */}
      <header className="border-b border-white/10 bg-black/50 p-6 backdrop-blur-md sticky top-0 z-50">
        <div className="mx-auto max-w-6xl flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="bg-purple-600 p-2 rounded-lg">
              <Play className="w-5 h-5 text-white fill-white" />
            </div>
            <h1 className="text-xl font-bold tracking-tight bg-gradient-to-r from-white to-zinc-400 bg-clip-text text-transparent">
              NeverRTFM
            </h1>
          </div>
          <p className="text-sm font-medium text-zinc-400">
            Reports to Video <span className="text-purple-400">+ Live Q&A</span>
          </p>
        </div>
      </header>

      {/* Main Content */}
      <main className="mx-auto max-w-6xl p-6 lg:p-12 grid grid-cols-1 lg:grid-cols-2 gap-12 lg:gap-24 items-center min-h-[calc(100vh-88px)]">

        {/* Left Column: Upload & Status */}
        <div className="flex flex-col gap-8">
          <div className="space-y-4">
            <h2 className="text-4xl lg:text-5xl font-extrabold tracking-tight leading-tight">
              Don&apos;t read the manual. <br />
              <span className="text-purple-500">Watch it.</span>
            </h2>
            <p className="text-lg text-zinc-400 leading-relaxed max-w-md">
              Upload any dense corporate report or PDF. We&apos;ll generate a punchy 60-second summary video. Ask questions during the video to get live answers.
            </p>
          </div>

          <div className="bg-zinc-900 border border-white/10 rounded-2xl p-6 shadow-2xl relative overflow-hidden group">
            {/* Ambient glow */}
            <div className="absolute inset-0 bg-gradient-to-br from-purple-500/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />

            <div className="relative flex flex-col items-center justify-center border-2 border-dashed border-zinc-700/50 rounded-xl p-10 text-center hover:border-purple-500/50 transition-colors">
              <UploadCloud className="w-12 h-12 text-zinc-500 mb-4" />
              <h3 className="text-lg font-medium text-zinc-200 mb-2">
                Drop your PDF report here
              </h3>
              <p className="text-sm text-zinc-500 mb-6">
                Limit 50MB. PDF only.
              </p>

              <label className="cursor-pointer bg-white text-black px-6 py-2.5 rounded-full font-semibold hover:bg-zinc-200 transition-colors shadow-lg shadow-white/10 text-sm">
                Browse Files
                <input
                  type="file"
                  accept="application/pdf"
                  className="hidden"
                  onChange={handleFileChange}
                />
              </label>

              {file && (
                <div className="mt-6 flex items-center gap-3 bg-zinc-950 px-4 py-2 rounded-lg border border-white/5 w-full text-left">
                  <FileText className="w-5 h-5 text-purple-400 shrink-0" />
                  <span className="text-sm text-zinc-300 truncate font-medium">
                    {file.name}
                  </span>
                </div>
              )}
            </div>

            <button
              onClick={handleUpload}
              disabled={!file || isUploading || jobStatus === "processing"}
              className="mt-4 w-full bg-purple-600 hover:bg-purple-500 disabled:bg-zinc-800 disabled:text-zinc-500 text-white font-bold py-3.5 rounded-xl transition-all shadow-lg flex items-center justify-center gap-2 group/btn"
            >
              {isUploading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Uploading...
                </>
              ) : jobStatus === "processing" ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Generating...
                </>
              ) : (
                <>
                  Generate Video
                  <Play className="w-4 h-4 fill-current opacity-70 group-hover/btn:opacity-100 transition-opacity" />
                </>
              )}
            </button>
          </div>

          {/* Status Tracker */}
          {jobStatus !== "idle" && (
            <div className={`border rounded-xl p-5 space-y-3 ${
              jobStatus === "error"
                ? "bg-red-950/30 border-red-500/20"
                : "bg-zinc-900/50 border-white/5"
            }`}>
              {/* Row 1: upload status */}
              <div className="flex items-center gap-3">
                {isUploading ? (
                  <Loader2 className="w-4 h-4 text-yellow-400 animate-spin shrink-0" />
                ) : jobStatus === "error" && !jobId ? (
                  <div className="w-3 h-3 bg-red-500 rounded-full shrink-0" />
                ) : (
                  <div className="w-3 h-3 bg-green-500 rounded-full shrink-0" />
                )}
                <span className="text-sm font-medium text-zinc-300">
                  {isUploading
                    ? "Uploading PDF..."
                    : jobStatus === "error" && !jobId
                    ? "Upload failed"
                    : "PDF uploaded successfully"}
                </span>
              </div>

              {/* Row 2: pipeline status (only once upload succeeded) */}
              {jobId && (
                <div className="flex items-start gap-3">
                  {jobStatus === "processing" ? (
                    <Loader2 className="w-4 h-4 text-yellow-400 animate-spin mt-0.5 shrink-0" />
                  ) : jobStatus === "error" ? (
                    <div className="w-3 h-3 bg-red-500 rounded-full mt-1 shrink-0" />
                  ) : (
                    <div className="w-3 h-3 bg-green-500 rounded-full mt-1 shrink-0" />
                  )}
                  <div className="flex flex-col gap-1 min-w-0">
                    <span className="text-sm font-medium text-zinc-300">
                      {jobStatus === "processing"
                        ? STEP_LABELS[pipelineStep] ?? "Starting pipeline..."
                        : jobStatus === "error"
                        ? "Generation failed"
                        : "Video ready"}
                    </span>
                    {jobStatus === "error" && errorMsg && (
                      <span className="text-xs text-red-400 break-words">{errorMsg}</span>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Right Column: Video Player & Q&A */}
        <div className="flex justify-center lg:justify-end perspective-1000">
          <div className="relative w-full max-w-[340px] aspect-[9/16] bg-black rounded-[2.5rem] border-[8px] border-zinc-900 shadow-2xl overflow-hidden ring-1 ring-white/10">
            {videoUrl ? (
              <video
                ref={videoRef}
                src={videoUrl}
                className="w-full h-full object-cover"
                controls
                autoPlay
                loop
                playsInline
                onPlay={openWebSocket}
              />
            ) : jobStatus === "processing" ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center p-8 text-center bg-zinc-950">
                <Loader2 className="w-10 h-10 text-purple-500 animate-spin mb-4" />
                <p className="text-sm font-semibold text-zinc-300">Generating your video…</p>
                <p className="text-xs text-zinc-500 mt-1">This takes a few minutes</p>
              </div>
            ) : (
              <div className="absolute inset-0 flex flex-col items-center justify-center p-8 text-center bg-zinc-950">
                <div className="w-16 h-16 rounded-full bg-zinc-900 border border-white/5 flex items-center justify-center mb-6 shadow-inner">
                  <Play className="w-6 h-6 text-zinc-700 ml-1" />
                </div>
                <p className="text-sm font-medium text-zinc-500 leading-relaxed">
                  Your customized video will appear here.
                </p>
              </div>
            )}

            {/* Live Q&A Mic Overlay */}
            {jobStatus === "done" && (
              <div className="absolute bottom-24 right-4 flex flex-col gap-4">
                <button
                  onClick={handleMicToggle}
                  className={`p-4 rounded-full shadow-2xl backdrop-blur-md transition-all ${
                    isRecording
                      ? "bg-red-500/90 text-white animate-pulse shadow-red-500/50 scale-110"
                      : "bg-black/60 text-white hover:bg-black border border-white/20"
                  }`}
                >
                  {isRecording ? <Pause className="w-6 h-6" /> : <Mic className="w-6 h-6" />}
                </button>
              </div>
            )}

            {/* Listening indicator */}
            {isRecording && (
              <div className="absolute bottom-8 left-0 right-0 px-6 text-center">
                <div className="bg-black/80 backdrop-blur-md px-4 py-2 rounded-full border border-red-500/30 text-xs font-semibold text-red-400 inline-block shadow-lg">
                  Listening... Video Paused
                </div>
              </div>
            )}
          </div>
        </div>

      </main>
    </div>
  );
}
