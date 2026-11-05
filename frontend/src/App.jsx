import { useState, useRef, useEffect } from "react";
import {
  Upload,
  Image as ImageIcon,
  ArrowRight,
  ShieldCheck,
  Microscope,
  Sparkles,
  X,
  ScanSearch,
  Check,
  LoaderCircle,
  FileText,
  RotateCcw,
  AlertCircle,
  ChevronDown,
  SlidersHorizontal,
  Layers,
  Info,
  ZoomIn,
  ZoomOut,
  Maximize2,
} from "lucide-react";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./App.css";

const API_BASE = "http://127.0.0.1:5000";

/* ---------- Six-stage processing pipeline ---------- */

const STAGES = [
  { label: "Uploading Image", detail: "Sending the image securely", icon: Upload },
  { label: "Preprocessing Image", detail: "Resizing and normalizing", icon: SlidersHorizontal },
  { label: "Extracting Features", detail: "Reading texture, color and shape", icon: Layers },
  { label: "Analyzing Lesion", detail: "Classifying with the model", icon: Microscope },
  { label: "Generating Grad-CAM", detail: "Building the visual explanation", icon: ScanSearch },
  { label: "Finalizing Results", detail: "Assembling your report", icon: FileText },
];

// While waiting for the backend, the pipeline advances up to this stage;
// the final stage completes only when the real response arrives.
const MAX_WAITING_STAGE = 4;
const STAGE_INTERVAL_MS = 1300;
const MIN_PROGRESS_DURATION_MS = 8000; // Combined with final-stage/transition timing: about 9–10 seconds

const FEATURES = [
  {
    icon: Microscope,
    title: "AI Classification",
    text: "Explore model predictions and class probabilities.",
  },
  {
    icon: Sparkles,
    title: "Visual Explanations",
    text: "Explore Grad-CAM heatmaps and model attention.",
  },
  {
    icon: ImageIcon,
    title: "Educational Insights",
    text: "Learn about lesion classes and their characteristics.",
  },
];

const PANELS = [
  { id: "result", label: "Result" },
  { id: "explain", label: "Explanation" },
  { id: "learn", label: "Learn" },
];

const VISUAL_TABS = [
  { id: "original", label: "Original" },
  { id: "heatmap", label: "Heatmap" },
  { id: "overlay", label: "Overlay" },
];

/* ---------- Small reusable components ---------- */

// Google-style four dots (pure CSS animation).
function GoogleDots({ label = "Processing" }) {
  return (
    <div className="g-dots" role="status" aria-label={label}>
      <span className="g-dot g-blue" />
      <span className="g-dot g-red" />
      <span className="g-dot g-yellow" />
      <span className="g-dot g-green" />
    </div>
  );
}

// Animated number for the confidence value; ends on the exact backend value.
function CountUp({ value }) {
  const target = Number(value);
  const isNumber = Number.isFinite(target);
  const decimals = isNumber ? (String(value).split(".")[1] || "").length : 0;
  const [display, setDisplay] = useState(0);

  useEffect(() => {
    if (!isNumber) return undefined;
    let frame;
    const start = performance.now();
    const duration = 900;

    const tick = (now) => {
      const t = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - t, 3);
      setDisplay(target * eased);
      if (t < 1) frame = requestAnimationFrame(tick);
    };

    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [target, isNumber]);

  if (!isNumber) return <>{value}</>;
  return <>{display.toFixed(decimals)}</>;
}

const RING_RADIUS = 52;
const RING_LENGTH = 2 * Math.PI * RING_RADIUS;

function ConfidenceRing({ value }) {
  const pct = Math.max(0, Math.min(Number(value) || 0, 100));
  const offset = RING_LENGTH * (1 - pct / 100);

  return (
    <div className="ring">
      <svg viewBox="0 0 120 120" aria-hidden="true">
        <circle className="ring-track" cx="60" cy="60" r={RING_RADIUS} />
        <circle
          className="ring-value"
          cx="60"
          cy="60"
          r={RING_RADIUS}
          style={{
            "--length": RING_LENGTH,
            "--offset": offset,
          }}
        />
      </svg>
      <div className="ring-center">
        <strong>
          <CountUp value={value} />
          <small>%</small>
        </strong>
        <span>confidence</span>
      </div>
    </div>
  );
}

const MIN_ZOOM = 1;
const MAX_ZOOM = 5;
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

// Interactive viewer: scroll / buttons / +,- keys to zoom, drag to pan,
// double-click to toggle zoom. At 1x the full, uncropped image is shown, and
// the zoom level is shared across Original / Heatmap / Overlay so they can be
// compared at identical framing.
function VisualViewer({ src, alt, ratio, resetKey }) {
  const [view, setView] = useState({ zoom: 1, x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const frameRef = useRef(null);
  const viewRef = useRef(view);
  const dragRef = useRef(null);
  viewRef.current = view;

  useEffect(() => {
    setView({ zoom: 1, x: 0, y: 0 });
  }, [resetKey]);

  // ox/oy: zoom origin relative to the frame center (keeps that point fixed).
  const zoomTo = (nextZoom, ox = 0, oy = 0) => {
    const el = frameRef.current;
    if (!el) return;
    const base = viewRef.current;
    const { width, height } = el.getBoundingClientRect();
    const zoom = clamp(nextZoom, MIN_ZOOM, MAX_ZOOM);
    const k = zoom / base.zoom;
    const maxX = ((zoom - 1) * width) / 2;
    const maxY = ((zoom - 1) * height) / 2;
    setView({
      zoom,
      x: clamp(ox - (ox - base.x) * k, -maxX, maxX),
      y: clamp(oy - (oy - base.y) * k, -maxY, maxY),
    });
  };

  const originOf = (event) => {
    const r = frameRef.current.getBoundingClientRect();
    return [
      event.clientX - r.left - r.width / 2,
      event.clientY - r.top - r.height / 2,
    ];
  };

  useEffect(() => {
    const el = frameRef.current;
    if (!el) return undefined;
    const onWheel = (event) => {
      event.preventDefault();
      const [ox, oy] = originOf(event);
      zoomTo(viewRef.current.zoom * Math.exp(-event.deltaY * 0.002), ox, oy);
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const onPointerDown = (event) => {
    if (view.zoom <= 1) return;
    dragRef.current = {
      sx: event.clientX,
      sy: event.clientY,
      x: view.x,
      y: view.y,
    };
    event.currentTarget.setPointerCapture(event.pointerId);
    setDragging(true);
  };

  const onPointerMove = (event) => {
    const d = dragRef.current;
    if (!d) return;
    const { width, height } = frameRef.current.getBoundingClientRect();
    const { zoom } = viewRef.current;
    const maxX = ((zoom - 1) * width) / 2;
    const maxY = ((zoom - 1) * height) / 2;
    setView({
      zoom,
      x: clamp(d.x + event.clientX - d.sx, -maxX, maxX),
      y: clamp(d.y + event.clientY - d.sy, -maxY, maxY),
    });
  };

  const endDrag = () => {
    dragRef.current = null;
    setDragging(false);
  };

  const onDoubleClick = (event) => {
    if (view.zoom > 1) {
      setView({ zoom: 1, x: 0, y: 0 });
    } else {
      const [ox, oy] = originOf(event);
      zoomTo(2.5, ox, oy);
    }
  };

  const onKeyDown = (event) => {
    if (event.key === "+" || event.key === "=") zoomTo(view.zoom * 1.4);
    else if (event.key === "-" || event.key === "_") zoomTo(view.zoom / 1.4);
    else if (event.key === "0") setView({ zoom: 1, x: 0, y: 0 });
  };

  return (
    <>
      <div
        ref={frameRef}
        className={`visual-frame ${view.zoom > 1 ? "zoomed" : ""} ${
          dragging ? "dragging" : ""
        }`}
        style={{ "--ratio": ratio }}
        tabIndex={0}
        role="img"
        aria-label={`${alt}. Scroll to zoom, drag to pan, double-click to toggle zoom.`}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        onDoubleClick={onDoubleClick}
        onKeyDown={onKeyDown}
      >
        <div
          className="visual-zoom"
          style={{
            transform: `translate(${view.x}px, ${view.y}px) scale(${view.zoom})`,
          }}
        >
          <img key={src} src={src} alt={alt} draggable={false} />
        </div>
      </div>

      <div className="zoom-toolbar" role="group" aria-label="Zoom controls">
        <button
          className="zoom-btn"
          onClick={() => zoomTo(view.zoom / 1.4)}
          disabled={view.zoom <= MIN_ZOOM}
          title="Zoom out (-)"
          aria-label="Zoom out"
        >
          <ZoomOut size={15} />
        </button>
        <span className="zoom-level">{Math.round(view.zoom * 100)}%</span>
        <button
          className="zoom-btn"
          onClick={() => zoomTo(view.zoom * 1.4)}
          disabled={view.zoom >= MAX_ZOOM}
          title="Zoom in (+)"
          aria-label="Zoom in"
        >
          <ZoomIn size={15} />
        </button>
        <button
          className="zoom-btn"
          onClick={() => setView({ zoom: 1, x: 0, y: 0 })}
          disabled={view.zoom === 1}
          title="Reset view (0)"
          aria-label="Reset view"
        >
          <Maximize2 size={14} />
        </button>
      </div>
    </>
  );
}

// Derived purely from the backend confidence number (no hardcoded lesion data).
function confidenceLevel(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return { text: "Unknown", tone: "neutral" };
  if (n >= 80) return { text: "High model confidence", tone: "green" };
  if (n >= 50) return { text: "Moderate model confidence", tone: "yellow" };
  return { text: "Low model confidence", tone: "red" };
}

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [screen, setScreen] = useState("home");
  const [progressStep, setProgressStep] = useState(0);
  const [report, setReport] = useState(null);
  const [activeVisualization, setActiveVisualization] = useState("overlay");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  // UI-only state
  const [isDragging, setIsDragging] = useState(false);
  const [panel, setPanel] = useState("result");
  const [disclaimerOpen, setDisclaimerOpen] = useState(false);
  const [imageRatio, setImageRatio] = useState(1);

  const inputRef = useRef(null);
  const progressTimer = useRef(null);
  const reportTimer = useRef(null);

  useEffect(() => {
    return () => {
      clearInterval(progressTimer.current);
      clearTimeout(reportTimer.current);
    };
  }, []);

  // Reset UI-only state whenever the screen changes.
  useEffect(() => {
    setPanel("result");
    setDisclaimerOpen(false);
  }, [screen]);

  const handleImage = (file) => {
    if (!file) return;

    if (!["image/jpeg", "image/png"].includes(file.type)) {
      setError("Please upload a JPG, JPEG, or PNG image.");
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      setError("Image size must be less than 10 MB.");
      return;
    }

    setError("");

    setPreview((oldPreview) => {
      if (oldPreview) URL.revokeObjectURL(oldPreview);
      return URL.createObjectURL(file);
    });

    setSelectedImage(file);
  };

  const removeImage = () => {
    setPreview((oldPreview) => {
      if (oldPreview) URL.revokeObjectURL(oldPreview);
      return null;
    });

    setSelectedImage(null);
    setError("");
  };

  const handleDrop = (event) => {
    event.preventDefault();
    handleImage(event.dataTransfer.files[0]);
  };

  // Paste an image straight from the clipboard on the home screen.
  useEffect(() => {
    if (screen !== "home") return undefined;
    const onPaste = (event) => {
      const item = Array.from(event.clipboardData?.items || []).find((i) =>
        i.type.startsWith("image/")
      );
      if (item) handleImage(item.getAsFile());
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  });

  // Keys 1 / 2 / 3 switch Original / Heatmap / Overlay on the report screen.
  useEffect(() => {
    if (screen !== "report") return undefined;
    const keys = { 1: "original", 2: "heatmap", 3: "overlay" };
    const onKey = (event) => {
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      if (keys[event.key]) setActiveVisualization(keys[event.key]);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [screen]);

  const handleAnalyze = async () => {
    if (!selectedImage || loading) return;

    setLoading(true);
    setError("");
    setReport(null);
    setProgressStep(0);
    setScreen("progress");
    const analysisStartedAt = Date.now();

    progressTimer.current = setInterval(() => {
      setProgressStep((current) => Math.min(current + 1, MAX_WAITING_STAGE));
    }, STAGE_INTERVAL_MS);

    try {
      const formData = new FormData();
      formData.append("image", selectedImage);

      const response = await fetch(`${API_BASE}/api/analyze`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.message || data.error || "Analysis failed."
        );
      }

      // Validate essential report fields.
      if (
        !data.prediction ||
        !data.visualization ||
        !data.prediction.predicted_label
      ) {
        throw new Error("The backend returned an incomplete report.");
      }

      // Keep the visual workflow on screen for a minimum duration even if the
      // backend responds quickly. This does not alter or delay backend work.
      const elapsed = Date.now() - analysisStartedAt;
      const remaining = Math.max(0, MIN_PROGRESS_DURATION_MS - elapsed);
      if (remaining > 0) {
        await new Promise((resolve) => {
          reportTimer.current = setTimeout(resolve, remaining);
        });
      }

      // Show the final stage explicitly before completing the workflow.
      clearInterval(progressTimer.current);
      setProgressStep(STAGES.length - 1);
      await new Promise((resolve) => {
        reportTimer.current = setTimeout(resolve, 900);
      });

      setProgressStep(STAGES.length);
      setReport(data);
      setActiveVisualization("overlay");

      reportTimer.current = setTimeout(() => {
        setScreen("report");
        setLoading(false);
      }, 600);
    } catch (err) {
      clearInterval(progressTimer.current);
      clearTimeout(reportTimer.current);

      setLoading(false);
      setScreen("home");

      setError(
        err.message === "Failed to fetch"
          ? "Cannot connect to Flask. Make sure your backend is running at 127.0.0.1:5000."
          : err.message
      );
    }
  };

  const resetApp = () => {
    clearInterval(progressTimer.current);
    clearTimeout(reportTimer.current);

    removeImage();
    setReport(null);
    setProgressStep(0);
    setScreen("home");
    setLoading(false);
    setError("");
  };

  
const imageUrl = (path) => {
  if (!path) return "";

  // Cloudinary URLs should be used directly.
  if (/^https?:\/\//i.test(path)) {
    return path;
  }

  // Local backend paths still work.
  return `${API_BASE}${path.startsWith("/") ? "" : "/"}${path}`;
};


  const visualizationPaths = report?.visualization
    ? {
        original: report.visualization.original_image,
        heatmap: report.visualization.gradcam_heatmap,
        overlay: report.visualization.gradcam_overlay,
      }
    : {};

  const currentVisualization =
    visualizationPaths[activeVisualization];

  const educationalInfo = report?.educational_information;

  // Read the real aspect ratio of the original image so all three
  // visualization tabs share one identical, uncropped frame.
  const originalPath = visualizationPaths.original;
  useEffect(() => {
    if (!originalPath) return undefined;
    let cancelled = false;
    const probe = new window.Image();
    probe.onload = () => {
      if (!cancelled && probe.naturalWidth && probe.naturalHeight) {
        setImageRatio(probe.naturalWidth / probe.naturalHeight);
      }
    };
    probe.src = imageUrl(originalPath);
    return () => {
      cancelled = true;
    };
  }, [originalPath]);

  // UI-only: which class has the highest probability (to highlight its bar).
  const probabilityEntries = Object.entries(
    report?.prediction?.class_probabilities || {}
  );
  const topLabel = probabilityEntries.reduce(
    (best, [label, value]) =>
      best === null || Number(value) > Number(best[1]) ? [label, value] : best,
    null
  )?.[0];

  const level = confidenceLevel(report?.prediction?.confidence);

  const done = progressStep >= STAGES.length;
  const progressPercent = done
    ? 100
    : Math.min(Math.round(((progressStep + 0.5) / STAGES.length) * 100), 95);

  return (
    <div className="app">
      {/* HEADER */}
      <header className="navbar">
        <button
          className="brand"
          onClick={screen === "report" ? resetApp : undefined}
          aria-label="AI-based skin lesion classification with explainability"
          tabIndex={screen === "report" ? 0 : -1}
        >
          AI-BASED SKIN LESION CLASSIFICATION WITH EXPLAINABILITY
        </button>

        <div className="nav-status">
          <span className="status-dot" />
          Research Mode
        </div>
      </header>

      {/* HOME */}
      {screen === "home" && (
        <main className="screen home">
          <section className="home-intro">
            <div className="hero-badge">
              <Sparkles size={13} />
              AI-POWERED SKIN LESION ANALYSIS
            </div>

            <h1>
              Understand your skin.
              <br />
              <span className="accent-text">Explore with AI.</span>
            </h1>

            <p className="hero-description">
              Upload a dermoscopic image to explore AI-generated
              lesion classification, visual explanations, and
              educational insights.
            </p>

            <div className="features">
              {FEATURES.map(({ icon: Icon, title, text }) => (
                <div className="feature-item" key={title}>
                  <div className="feature-icon">
                    <Icon size={18} />
                  </div>
                  <div>
                    <h3>{title}</h3>
                    <p>{text}</p>
                  </div>
                </div>
              ))}
            </div>
          </section>

          <section className="upload-section">
            <div
              className={`upload-card ${preview ? "has-image" : ""} ${
                isDragging ? "dragging" : ""
              }`}
              role={!preview ? "button" : undefined}
              tabIndex={!preview ? 0 : -1}
              aria-label={!preview ? "Upload a dermoscopic image" : undefined}
              onKeyDown={(event) => {
                if (!preview && (event.key === "Enter" || event.key === " ")) {
                  event.preventDefault();
                  inputRef.current?.click();
                }
              }}
              onClick={() => {
                if (!preview) inputRef.current?.click();
              }}
              onDragOver={(event) => {
                event.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={(event) => {
                setIsDragging(false);
                handleDrop(event);
              }}
            >
              {!preview ? (
                <div className="upload-empty">
                  <div className="upload-icon">
                    <Upload size={26} />
                  </div>

                  <h2>
                    {isDragging
                      ? "Drop to upload"
                      : "Upload your dermoscopic image"}
                  </h2>

                  <p className="upload-description">
                    Drag and drop your image here, or browse
                    files from your device.
                  </p>

                  <button className="browse-button" type="button">
                    <ImageIcon size={16} />
                    Browse images
                  </button>

                  <div className="file-info">
                    JPG · JPEG · PNG <span>•</span> Max 10 MB
                  </div>
                </div>
              ) : (
                <div className="preview-container">
                  <div className="preview-header">
                    <div className="preview-meta">
                      <span className="preview-label">
                        <Check size={12} />
                        IMAGE SELECTED
                      </span>
                      <p className="preview-filename">
                        {selectedImage?.name}
                      </p>
                    </div>

                    <button
                      className="remove-button"
                      onClick={removeImage}
                      title="Remove image"
                      aria-label="Remove image"
                    >
                      <X size={17} />
                    </button>
                  </div>

                  <div className="preview-stage">
                    <img
                      src={preview}
                      alt="Uploaded dermoscopic image"
                      className="image-preview"
                    />
                  </div>

                  <div className="image-ready">
                    <ShieldCheck size={15} />
                    Image ready for analysis
                  </div>
                </div>
              )}

              <input
                ref={inputRef}
                type="file"
                accept=".jpg,.jpeg,.png,image/jpeg,image/png"
                hidden
                onClick={(event) => event.stopPropagation()}
                onChange={(event) =>
                  handleImage(event.target.files[0])
                }
              />
            </div>

            {error && (
              <div className="error-message" role="alert">
                <AlertCircle size={16} />
                <span>{error}</span>
              </div>
            )}

            <button
              className="analyze-button"
              disabled={!selectedImage || loading}
              onClick={handleAnalyze}
            >
              <ScanSearch size={18} />
              Analyze image
              <ArrowRight size={18} className="arrow" />
            </button>

            <p className="upload-note">
              <ShieldCheck size={13} />
              For educational and research purposes only
            </p>
          </section>

          <footer className="footer">
            <ShieldCheck size={14} />
            For educational and research purposes only.
            Not a substitute for professional medical advice.
          </footer>
        </main>
      )}

      {/* PROGRESS */}
      {screen === "progress" && (
        <main className="screen progress-screen">
          <div className="progress-layout">
            <div className="progress-media">
              <div className={`scan-card ${loading && !done ? "scanning" : ""}`}>
                {preview ? (
                  <img className="scan-image" src={preview} alt="Image being analyzed" />
                ) : (
                  <div className="scan-placeholder">
                    <ImageIcon size={32} />
                  </div>
                )}

                {loading && !done && (
                  <div className="scan-overlay" aria-hidden="true">
                    <div className="scan-grid" />
                    <div className="scan-band" />
                    <div className="scan-line">
                      <span className="scan-line-glow" />
                    </div>
                    <div className="scan-corner scan-corner-tl" />
                    <div className="scan-corner scan-corner-tr" />
                    <div className="scan-corner scan-corner-bl" />
                    <div className="scan-corner scan-corner-br" />
                  </div>
                )}
              </div>
              <GoogleDots label="Analysis in progress" />
            </div>

            <div className="progress-info">
              <div className="progress-top">
                <div>
                  <div className="hero-badge">
                    <Sparkles size={13} />
                    {done ? "ANALYSIS COMPLETE" : "ANALYSIS IN PROGRESS"}
                  </div>
                  <h1>
                    {done ? "Report ready." : "Analyzing your image."}
                  </h1>
                </div>
                <div className="percent" aria-live="polite">
                  {progressPercent}
                  <small>%</small>
                </div>
              </div>

              <div
                className="progress-bar"
                role="progressbar"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={progressPercent}
              >
                <div style={{ width: `${progressPercent}%` }} />
              </div>

              <ol className="workflow-card">
                {STAGES.map(({ label, detail, icon: Icon }, index) => {
                  const state =
                    index < progressStep
                      ? "completed"
                      : index === progressStep
                      ? "active"
                      : "upcoming";

                  return (
                    <li key={label} className={`workflow-step ${state}`}>
                      <div className="workflow-marker">
                        {state === "completed" ? (
                          <Check size={16} />
                        ) : state === "active" ? (
                          <LoaderCircle size={16} className="spin" />
                        ) : (
                          <Icon size={15} />
                        )}
                      </div>

                      <div className="workflow-text">
                        <strong>{label}</strong>
                        <span>
                          {state === "completed"
                            ? "Completed"
                            : state === "active"
                            ? detail
                            : "Waiting"}
                        </span>
                      </div>

                      <span className="workflow-tag">
                        {state === "completed"
                          ? "Done"
                          : state === "active"
                          ? "Running"
                          : `${index + 1}/${STAGES.length}`}
                      </span>
                    </li>
                  );
                })}
              </ol>
            </div>
          </div>
        </main>
      )}

      {/* REPORT */}
      {screen === "report" && report && (
        <main className="screen report">
          <header className="report-bar">
            <div className="report-title">
              <div className="hero-badge">
                <Check size={13} />
                ANALYSIS COMPLETE
              </div>

              <h1>Analysis report</h1>
            </div>

            <button className="secondary-button" onClick={resetApp}>
              <RotateCcw size={15} />
              New analysis
            </button>
          </header>

          <div
            className="panel-switch"
            role="tablist"
            style={{ "--idx": PANELS.findIndex((p) => p.id === panel) }}
          >
            <span className="tab-indicator" aria-hidden="true" />
            {PANELS.map((item) => (
              <button
                key={item.id}
                role="tab"
                aria-selected={panel === item.id}
                className={panel === item.id ? "selected" : ""}
                onClick={() => setPanel(item.id)}
              >
                {item.label}
              </button>
            ))}
          </div>

          <div className="report-body" data-panel={panel}>
            {/* PREDICTION */}
            <section
              className="rcard prediction-card"
              data-panel-id="result"
              style={{ "--i": 0 }}
            >
              <div className="section-eyebrow">
                <Sparkles size={14} />
                PREDICTION SUMMARY
              </div>

              <div className="prediction-label">
                {report.prediction.predicted_label}
              </div>

              <div className="context-chips">
                <span className={`chip ${level.tone}`}>{level.text}</span>
                <span className="chip neutral">
                  <Info size={12} />
                  AI prediction · not a diagnosis
                </span>
              </div>

              <div className="confidence-block">
                <ConfidenceRing value={report.prediction.confidence} />
                <p className="confidence-caption">
                  Model confidence for the predicted class. This is an
                  AI estimate, not a confirmed medical diagnosis.
                </p>
              </div>

              <div className="probability-list">
                {probabilityEntries.map(([label, value], index) => (
                  <div
                    className={`probability-row ${
                      label === topLabel ? "top" : ""
                    }`}
                    key={label}
                    style={{ "--i": index }}
                  >
                    <span>{label}</span>

                    <div className="probability-track">
                      <div
                        style={{
                          width: `${Math.min(Number(value), 100)}%`,
                        }}
                      />
                    </div>

                    <strong>{value}%</strong>
                  </div>
                ))}
              </div>
            </section>

            {/* GRAD-CAM */}
            <section
              className="rcard visual-card"
              data-panel-id="explain"
              style={{ "--i": 1 }}
            >
              <div className="visual-head">
                <div className="section-eyebrow">
                  <ImageIcon size={14} />
                  VISUAL EXPLANATION
                </div>

                <div
                  className="visual-tabs"
                  role="tablist"
                  style={{
                    "--idx": Math.max(
                      0,
                      VISUAL_TABS.findIndex(
                        (t) => t.id === activeVisualization
                      )
                    ),
                  }}
                >
                  <span className="tab-indicator" aria-hidden="true" />
                  {VISUAL_TABS.map((tab, tabIndex) => (
                    <button
                      key={tab.id}
                      role="tab"
                      title={`${tab.label} (press ${tabIndex + 1})`}
                      aria-selected={activeVisualization === tab.id}
                      className={
                        activeVisualization === tab.id ? "selected" : ""
                      }
                      onClick={() => setActiveVisualization(tab.id)}
                    >
                      {tab.label}
                    </button>
                  ))}
                </div>
              </div>

              <div className="visual-stage">
                {currentVisualization ? (
                  <VisualViewer
                    src={imageUrl(currentVisualization)}
                    alt={`${activeVisualization} visualization`}
                    ratio={imageRatio}
                    resetKey={originalPath}
                  />
                ) : (
                  <p className="visual-unavailable">
                    Visualization unavailable
                  </p>
                )}
              </div>

              <div className="cam-legend">
                <span>Low activation</span>
                <div className="legend-gradient" />
                <span>High activation</span>
              </div>

              <p className="legend-note">
                Grad-CAM highlights image regions that influenced
                the model's prediction. It is not a lesion boundary
                or diagnostic map.
              </p>
            </section>

            {/* EDUCATIONAL INFORMATION */}
            <section
              className="rcard education-card"
              data-panel-id="learn"
              style={{ "--i": 2 }}
            >
              <div className="section-eyebrow">
                <FileText size={14} />
                EDUCATIONAL INFORMATION
              </div>

              <h2 className="education-title">
                {report.prediction.predicted_label}
                <span>
                  {" "}· {educationalInfo?.title || "Class information"}
                </span>
              </h2>

              <div className="education-scroll">
                {educationalInfo?.available &&
                educationalInfo?.content ? (
                  <div className="markdown-content">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {educationalInfo.content}
                    </ReactMarkdown>
                  </div>
                ) : (
                  <p className="education-unavailable">
                    Educational information is unavailable for this class.
                  </p>
                )}
              </div>
            </section>
          </div>

          {/* DISCLAIMER */}
          <button
            className={`medical-disclaimer ${disclaimerOpen ? "open" : ""}`}
            onClick={() => setDisclaimerOpen((open) => !open)}
            aria-expanded={disclaimerOpen}
          >
            <ShieldCheck size={16} />
            <p>{report.medical_disclaimer}</p>
            <ChevronDown size={16} className="chevron" />
          </button>
        </main>
      )}
    </div>
  );
}

export default App;