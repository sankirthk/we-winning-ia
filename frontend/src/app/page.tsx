"use client";

import { useState } from "react";
import { UploadCloud, FileText, Mic, Play, Pause, Loader2 } from "lucide-react";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [jobStatus, setJobStatus] = useState<"idle" | "processing" | "done" | "error">("idle");
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = () => {
    if (!file) return;
    setIsUploading(true);
    setJobStatus("processing");
    
    // TODO: Implement actual API call to /api/generate
    setTimeout(() => {
      setIsUploading(false);
      setJobStatus("done");
      // Dummy video for placeholder purposes
      setVideoUrl("https://www.w3schools.com/html/mov_bbb.mp4");
    }, 3000);
  };

  const handleMicToggle = () => {
    setIsRecording(!isRecording);
    // TODO: Implement LiveAgent WebSocket connection handling
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
              disabled={!file || isUploading}
              className="mt-4 w-full bg-purple-600 hover:bg-purple-500 disabled:bg-zinc-800 disabled:text-zinc-500 text-white font-bold py-3.5 rounded-xl transition-all shadow-lg flex items-center justify-center gap-2 group/btn"
            >
              {isUploading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Processing...
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
            <div className="bg-zinc-900/50 border border-white/5 rounded-xl p-5">
              <div className="flex items-center gap-3">
                {jobStatus === "processing" ? (
                  <div className="w-3 h-3 bg-yellow-500 rounded-full animate-pulse" />
                ) : (
                  <div className="w-3 h-3 bg-green-500 rounded-full" />
                )}
                <span className="text-sm font-medium text-zinc-300">
                  {jobStatus === "processing" ? "Agent pipeline running..." : "Generation complete"}
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Right Column: Video Player & Q&A */}
        <div className="flex justify-center lg:justify-end perspective-1000">
          <div className="relative w-full max-w-[340px] aspect-[9/16] bg-black rounded-[2.5rem] border-[8px] border-zinc-900 shadow-2xl overflow-hidden ring-1 ring-white/10">
            {videoUrl ? (
              <video 
                src={videoUrl} 
                className="w-full h-full object-cover" 
                controls 
                autoPlay 
                loop 
                playsInline
              />
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
