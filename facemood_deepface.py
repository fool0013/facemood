import time, threading, math
from collections import deque
import cv2
import numpy as np
from deepface import DeepFace
from pynput import keyboard

# util
def clamp01(x): return max(0.0, min(1.0, float(x)))

def ema_update(avg_dict, new_dict, alpha=0.2):
    """exponential moving avg for emotion probability dicts"""
    if avg_dict is None: return dict(new_dict)
    out = {}
    keys = set(avg_dict.keys()) | set(new_dict.keys())
    for k in keys:
        a = float(avg_dict.get(k, 0.0))
        b = float(new_dict.get(k, 0.0))
        out[k] = (1 - alpha) * a + alpha * b
    # renormalize
    s = sum(out.values()) + 1e-9
    for k in out: out[k] /= s
    return out

EMO_LIST = ["angry","disgust","fear","happy","sad","surprise","neutral"]

# check 4 kpm (keys/min) 
class KpmCounter:
    def __init__(self):
        self.buf = deque(maxlen=6000)
        self.lock = threading.Lock()
        self.listener = keyboard.Listener(on_press=self._on_press)
        self.listener.daemon = True
        self.listener.start()
    def _on_press(self, key):
        # count only printable chars (this so ass)
        try:
            if hasattr(key, 'char') and key.char and key.char.isprintable():
                with self.lock:
                    self.buf.append(time.time())
        except Exception:
            pass
    def value(self):
        now = time.time(); one_min = now - 60
        with self.lock:
            while self.buf and self.buf[0] < one_min:
                self.buf.popleft()
            return len(self.buf)

# arousal from motion
class MotionEstimator:
    def __init__(self):
        self.prev = None
        self.prev_avg = None
    def step(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (320, 180))
        if self.prev is None:
            self.prev = small; self.prev_avg = float(small.mean()); return 0.0
        mad = np.mean(np.abs(small.astype(np.int16) - self.prev.astype(np.int16))) / 255.0
        self.prev = small
        avg = float(small.mean())
        lum = min(abs(avg - self.prev_avg) / 50.0, 1.0)
        self.prev_avg = avg
        return clamp01(0.7*mad + 0.3*lum)

# mood fusion mix
def fuse_to_mood(em, arousal, kpm):
    """map deepFace emotions + arousal + kpm to a friendly mood"""
    p = lambda k: float(em.get(k, 0.0))
    # score buckets (tuned for fer prototype. adjust to ur taste)
    scores = {
        "energetic": 0.9*p("surprise") + 0.5*p("happy") + 0.35*arousal + 0.002*kpm,
        "focus":     0.8*p("neutral")  + 0.3*(1 - p("surprise")) + 0.0015*kpm + 0.2*(1 - p("happy")),
        "chill":     0.7*(1 - arousal) + 0.5*p("neutral") + 0.2*p("happy"),
        "melancholy":0.9*p("sad")      + 0.25*p("fear") + 0.15*p("disgust") + 0.6*(1 - arousal),
        "happy":     1.0*p("happy")    + 0.2*p("surprise") + 0.1*p("neutral"),
    }
    mood = max(scores, key=scores.get)
    conf = clamp01(scores[mood] / (sum(scores.values()) + 1e-9))
    return mood, conf, scores

# ,ain
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit("couldnt open webcam")

    # picks best available face detector for deepface
    DETECTORS = ["mediapipe", "retinaface", "opencv"]
    chosen = None
    for name in DETECTORS:
        try:
            _ = DeepFace.build_model("emotion")  # priming model cache
            chosen = name
            # quick dry-run (SHOULD skip if unavailable)
            break
        except Exception:
            continue
    if chosen is None:
        chosen = "opencv"  # contigency plan fallback

    print(f"[info] using deepFace detector backend: {chosen}")

    kpm = KpmCounter()
    motion = MotionEstimator()
    ema_emotions = None
    ring = deque(maxlen=30)  # ~1 second at ~30 FPS for median smoothing
    last_box = None
    find_every = 7           # re-detect every N frames (speed/accuracy tradeoff)
    frame_idx = 0

    def draw_bar(img, y, label, val):
        x0 = 20
        cv2.putText(img, label, (x0, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220,220,220), 1, cv2.LINE_AA)
        if val is None:
            cv2.putText(img, "—", (x0+210, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1, cv2.LINE_AA)
            return
        pct = int(100*clamp01(val))
        cv2.rectangle(img, (x0, y), (x0+200, y+12), (60,60,60), -1)
        cv2.rectangle(img, (x0, y), (x0+2*pct, y+12), (80,220,60), -1)
        cv2.putText(img, f"{pct}%", (x0+210, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        analyze_kwargs = dict(actions=["emotion"], enforce_detection=False, detector_backend=chosen)
        # Speed trick: crop around last face box, but every N frames re-detect full-frame
        roi = None
        if last_box is not None and frame_idx % find_every != 0:
            x,y,bw,bh = last_box
            pad = int(0.2*max(bw,bh))
            x0 = max(0, x - pad); y0 = max(0, y - pad)
            x1 = min(w, x + bw + pad); y1 = min(h, y + bh + pad)
            roi = frame[y0:y1, x0:x1]
            target = roi
        else:
            target = frame

        emotions = None
        box = None
        try:
            result = DeepFace.analyze(target, **analyze_kwargs)
            if isinstance(result, list): result = result[0]
            emotions = result.get("emotion") or result.get("emotions")
            # map back the box if we used ROI
            if "region" in result and result["region"]:
                rx = int(result["region"]["x"]); ry = int(result["region"]["y"])
                rw = int(result["region"]["w"]); rh = int(result["region"]["h"])
                if roi is not None:
                    x0 = max(0, last_box[0] - int(0.2*max(last_box[2], last_box[3])))
                    y0 = max(0, last_box[1] - int(0.2*max(last_box[2], last_box[3])))
                    box = (x0 + rx, y0 + ry, rw, rh)
                else:
                    box = (rx, ry, rw, rh)
        except Exception:
            pass

        if emotions:
            # Normalize to [0..1]
            total = sum(emotions.values()) + 1e-9
            emotions = {k: float(v)/total for k,v in emotions.items() if k in EMO_LIST}
            # median smoothing on a 1s ring buffer + EMA
            ring.append(emotions)
            med = {}
            for k in EMO_LIST:
                vals = [d.get(k,0.0) for d in ring]
                med[k] = float(np.median(vals))
            ema_emotions = ema_update(ema_emotions, med, alpha=0.25)
            if box: last_box = box

        # Arousal + calm
        arousal = motion.step(frame)
        calm = 1.0 - arousal
        keys_min = kpm.value()

        # HUD
        hud = frame.copy()
        if last_box:
            x,y,bw,bh = last_box
            cv2.rectangle(hud, (x,y), (x+bw, y+bh), (80,220,60), 2)

        y = 30
        em_show = ema_emotions or {k:None for k in EMO_LIST}
        for lbl in EMO_LIST:
            draw_bar(hud, y, lbl.capitalize(), em_show.get(lbl))
            y += 22

        draw_bar(hud, y+6, "Keys / min", min(keys_min/240.0, 1.0)); y += 28
        draw_bar(hud, y+6, "Calm", calm); y += 32

        # Fuse to mood (only if we have smoothed emotions)
        if ema_emotions:
            mood, conf, _ = fuse_to_mood(ema_emotions, arousal, keys_min)
            cv2.putText(hud, f"Mood: {mood}  (conf {int(conf*100)}%)", (20, y+18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40,255,220), 2, cv2.LINE_AA)
        else:
            cv2.putText(hud, "Detecting face/emotions…", (20, y+18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2, cv2.LINE_AA)

        cv2.imshow("Moodify (proto) — press Q to exit", hud)
        frame_idx += 1
        if (cv2.waitKey(1) & 0xFF) == ord('q'): break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
