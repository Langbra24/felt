import p5 from 'p5';
import { Pane } from 'tweakpane';
import Sentiment from 'sentiment';

// ═══════════════════════════════════════════════════════════════════════════
//  SIGNAL FORM — Experiment 02
//  Boids flocking driven by Anthropic streaming API + sentiment analysis
//  The flock = one mind. Its coherence, drift, and color = epistemic state.
// ═══════════════════════════════════════════════════════════════════════════

// ─── PARAMS ────────────────────────────────────────────────────────────────
const PARAMS = {
  // Boids shape
  count: 800,
  size: 3.5,
  sizeVariance: 0.6,

  // Flocking forces (these get driven by AI signals)
  separation: 1.4,
  alignment: 0.8,
  cohesion: 0.6,

  // Vision
  perceptionRadius: 60,
  separationRadius: 28,

  // Motion
  maxSpeed: 2.2,
  maxForce: 0.06,
  noiseStrength: 0.0,
  drag: 0.98,

  // Color
  baseHue: 0,        // white (idle default — saturation 0 means hue doesn't matter)
  hueRange: 30,
  saturation: 0,     // white when 0
  brightness: 100,   // full brightness for pure white
  trailOpacity: 18,

  // Signal mapping
  signalStrength: 1.0,      // overall blend of AI signal → boids (0=manual, 1=full AI)
  smoothing: 0.05,          // how fast params ease toward target values

  // Interaction
  mouseAttract: false,
  mouseRadius: 140,
  mouseForce: 0.8,
};

// ─── HEDGING LANGUAGE LEXICON ──────────────────────────────────────────────
// Words/phrases that signal uncertainty, hedging, or epistemic distance.
// Density of these in the response → flock drift / loss of cohesion.
const HEDGE_WORDS = [
  'perhaps', 'might', 'maybe', 'possibly', 'probably', 'likely',
  'some suggest', 'it seems', 'appears to', 'could be', 'may be',
  'it\'s possible', 'i think', 'i believe', 'arguably', 'supposedly',
  'allegedly', 'presumably', 'not entirely', 'not necessarily',
  'uncertain', 'unclear', 'unclearly', 'tentative', 'tentatively',
  'suggests', 'indicates', 'implies', 'seems to', 'tends to',
  'generally', 'usually', 'typically', 'often', 'sometimes',
  'in some cases', 'in many cases', 'to some extent', 'more or less',
  'sort of', 'kind of', 'in a way', 'up to a point',
  'i\'m not sure', 'hard to say', 'difficult to determine',
  'warrant further', 'remains to be seen', 'open question',
];

// ─── EPISTEMIC STATE ────────────────────────────────────────────────────────
// The central state object. All AI signals flow into here, then get mapped
// to boids parameters each frame.
const EPI = {
  stage: 'idle',           // idle | anticipating | reacting | streaming | settling
  text: '',                // accumulated response text
  tokenCount: 0,           // total tokens received
  tokenChunks: [],         // timestamps of recent chunks for speed calculation
  sentimentScore: 0,       // current sentiment (-5 to +5 roughly)
  sentimentNormalized: 0,  // -1 to 1
  hedgeDensity: 0,         // 0 to 1 (fraction of words that are hedges)
  tokenSpeed: 0,           // tokens per second (smoothed)
  stopReason: null,        // end_turn | max_tokens | stop_sequence
  streamStartTime: 0,
  lastChunkTime: 0,

  // Mapped targets (what boids params should ease toward)
  target: {
    separation: 1.4,
    alignment: 0.8,
    cohesion: 0.6,
    baseHue: 0,           // white idle
    brightness: 100,
    saturation: 0,        // 0 = white
    maxSpeed: 2.2,
    noiseStrength: 0.0,
    trailOpacity: 18,
  },
};

// ─── BOID CLASS ─────────────────────────────────────────────────────────────
class Boid {
  constructor(p) {
    this.p = p;
    this.pos = p.createVector(p.random(p.width), p.random(p.height));
    this.vel = p5.Vector.random2D().mult(p.random(1, PARAMS.maxSpeed));
    this.acc = p.createVector();
    this.size = PARAMS.size * (1 + p.random(-PARAMS.sizeVariance, PARAMS.sizeVariance));
  }

  edges() {
    const p = this.p;
    if (this.pos.x > p.width + 10) this.pos.x = -10;
    if (this.pos.x < -10) this.pos.x = p.width + 10;
    if (this.pos.y > p.height + 10) this.pos.y = -10;
    if (this.pos.y < -10) this.pos.y = p.height + 10;
  }

  flock(boids) {
    const p = this.p;
    let sep = p.createVector(), ali = p.createVector(), coh = p.createVector();
    let sepCount = 0, aliCount = 0, cohCount = 0;

    for (let other of boids) {
      if (other === this) continue;
      let d = p5.Vector.dist(this.pos, other.pos);

      if (d < PARAMS.separationRadius && d > 0) {
        let diff = p5.Vector.sub(this.pos, other.pos).div(d);
        sep.add(diff);
        sepCount++;
      }
      if (d < PARAMS.perceptionRadius) {
        ali.add(other.vel);
        coh.add(other.pos);
        aliCount++;
        cohCount++;
      }
    }

    if (sepCount > 0) {
      sep.div(sepCount).setMag(PARAMS.maxSpeed).sub(this.vel).limit(PARAMS.maxForce).mult(PARAMS.separation);
      this.acc.add(sep);
    }
    if (aliCount > 0) {
      ali.div(aliCount).setMag(PARAMS.maxSpeed).sub(this.vel).limit(PARAMS.maxForce).mult(PARAMS.alignment);
      this.acc.add(ali);
    }
    if (cohCount > 0) {
      coh.div(cohCount).sub(this.pos).setMag(PARAMS.maxSpeed).sub(this.vel).limit(PARAMS.maxForce).mult(PARAMS.cohesion);
      this.acc.add(coh);
    }

    if (PARAMS.noiseStrength > 0) {
      let n = p.noise(this.pos.x * 0.003, this.pos.y * 0.003, p.frameCount * 0.005);
      let noiseAngle = n * p.TWO_PI * 2;
      let noiseForce = p5.Vector.fromAngle(noiseAngle).mult(PARAMS.noiseStrength * 0.1);
      this.acc.add(noiseForce);
    }
  }

  attract(mx, my) {
    if (!PARAMS.mouseAttract) return;
    const p = this.p;
    let mouse = p.createVector(mx, my);
    let d = p5.Vector.dist(this.pos, mouse);
    if (d < PARAMS.mouseRadius && d > 5) {
      let force = p5.Vector.sub(mouse, this.pos).setMag(PARAMS.maxSpeed).sub(this.vel).limit(PARAMS.maxForce * PARAMS.mouseForce * 8);
      this.acc.add(force);
    }
  }

  update() {
    this.vel.add(this.acc).mult(PARAMS.drag).limit(PARAMS.maxSpeed);
    this.pos.add(this.vel);
    this.acc.mult(0);
    // Ease this.size toward PARAMS.size with small random jitter
    const jitter = this.p.random(-PARAMS.sizeVariance, PARAMS.sizeVariance) * 0.05;
    const target = PARAMS.size * (1 + jitter);
    this.size = this.size * 0.9 + target * 0.1;
    this.size = Math.max(0.5, this.size);
  }

  getColor() {
    const p = this.p;
    let speed = this.vel.mag() / PARAMS.maxSpeed;
    let h = PARAMS.baseHue + speed * PARAMS.hueRange;
    let b = 60 + speed * 35;
    return p.color(h, PARAMS.saturation, b);
  }

  draw() {
    const p = this.p;
    let angle = this.vel.heading();
    let speed = this.vel.mag() / PARAMS.maxSpeed;
    let skew = 1 + speed * 1.8;
    let w = this.size;
    let h = this.size * skew;

    p.push();
    p.translate(this.pos.x, this.pos.y);
    p.rotate(angle + p.HALF_PI);
    p.noStroke();
    p.fill(this.getColor());
    p.ellipse(0, 0, w, h);
    p.pop();
  }
}

// ─── SKETCH ─────────────────────────────────────────────────────────────────
let boids = [];
let needsReset = false;
const sentiment = new Sentiment();

new p5(function (p) {
  p.setup = function () {
    p.pixelDensity(Math.min(window.devicePixelRatio, 2));
    const container = document.getElementById('canvas-area');
    const w = container.offsetWidth;
    const h = container.offsetHeight;
    let cnv = p.createCanvas(w, h);
    cnv.parent(container);
    cnv.style('display', 'block');
    cnv.style('position', 'absolute');
    cnv.style('top', '0');
    cnv.style('left', '0');
    p.colorMode(p.HSB, 360, 100, 100, 100);
    p.background(0, 0, 4);
    spawnBoids(p);
  };

  p.draw = function () {
    if (needsReset) { spawnBoids(p); p.background(0, 0, 4); needsReset = false; }

    // Ease params toward AI-driven targets
    easeParams();

    // Fade trail
    p.push();
    p.blendMode(p.BLEND);
    p.fill(0, 0, 4, PARAMS.trailOpacity);
    p.noStroke();
    p.rect(0, 0, p.width, p.height);
    p.pop();

    p.blendMode(p.ADD);

    for (let b of boids) {
      b.flock(boids);
      b.attract(p.mouseX, p.mouseY);
      b.update();
      b.edges();
      b.draw();
    }

    p.blendMode(p.BLEND);

    // Update status readout
    updateStatus();
  };

  p.windowResized = function () {
    const container = document.getElementById('canvas-area');
    if (container) {
      p.resizeCanvas(container.offsetWidth, container.offsetHeight);
    }
  };
});

function spawnBoids(p) {
  boids = [];
  for (let i = 0; i < PARAMS.count; i++) {
    boids.push(new Boid(p));
  }
}

// ═══════════════════════════════════════════════════════════════════════════
//  SIGNAL MAPPING — AI epistemic state → boids parameters
// ═══════════════════════════════════════════════════════════════════════════

function easeParams() {
  if (PARAMS.signalStrength <= 0) return;

  const s = PARAMS.smoothing;
  const blend = PARAMS.signalStrength;

  // Lerp current params toward targets, modulated by signalStrength
  PARAMS.separation = lerp(PARAMS.separation, EPI.target.separation, s * blend);
  PARAMS.alignment = lerp(PARAMS.alignment, EPI.target.alignment, s * blend);
  PARAMS.cohesion = lerp(PARAMS.cohesion, EPI.target.cohesion, s * blend);
  PARAMS.baseHue = lerp(PARAMS.baseHue, EPI.target.baseHue, s * blend);
  PARAMS.brightness = lerp(PARAMS.brightness, EPI.target.brightness, s * blend);
  PARAMS.saturation = lerp(PARAMS.saturation, EPI.target.saturation, s * blend);
  PARAMS.maxSpeed = lerp(PARAMS.maxSpeed, EPI.target.maxSpeed, s * blend);
  PARAMS.noiseStrength = lerp(PARAMS.noiseStrength, EPI.target.noiseStrength, s * blend);
  PARAMS.trailOpacity = lerp(PARAMS.trailOpacity, EPI.target.trailOpacity, s * blend);

  // ─── Hard state override ───
  // Idle = white, Positive = teal-blue, Negative = red.
  // During the 2-second transition to idle, skip the snap so the easing
  // is visible. After that, hard-snap to idle values.
  const sent = EPI.sentimentNormalized;
  const NEG_THRESHOLD = -0.1;

  if (EPI.stage === 'idle' && idleTransitionStart !== null) {
    const elapsed = performance.now() - idleTransitionStart;
    if (elapsed < IDLE_TRANSITION_DURATION) {
      // Mid-transition — let easing happen naturally toward idle targets
      // (don't snap, don't override)
      return;
    }
    // Transition complete — clear the flag
    idleTransitionStart = null;
  }

  if (EPI.stage === 'idle') {
    // White, calm, minimal motion
    PARAMS.baseHue = 0;
    PARAMS.saturation = 0;
    PARAMS.brightness = 100;
    PARAMS.trailOpacity = 18;
    PARAMS.maxSpeed = 0.4;
    PARAMS.maxForce = 0.04;
    PARAMS.cohesion = 0.9;
    PARAMS.alignment = 1.0;
    PARAMS.separation = 1.0;
    PARAMS.noiseStrength = 0.02;
  } else if (sent < NEG_THRESHOLD) {
    // Negative — red, full saturation, exploded flock
    PARAMS.baseHue = 360;
    PARAMS.saturation = 100;
    PARAMS.brightness = 100;
    PARAMS.trailOpacity = 1;
    PARAMS.separation = 4.0;       // max — pushed apart
    PARAMS.alignment = 0;
    PARAMS.cohesion = 0;
    PARAMS.maxSpeed = 4.7;
    PARAMS.maxForce = 0.170;
    PARAMS.noiseStrength = 0.3;
  } else {
    // Positive / neutral — teal-blue, half intensity of negative
    PARAMS.baseHue = 200;
    PARAMS.saturation = 100;
    PARAMS.brightness = 100;
    PARAMS.trailOpacity = 9;
    PARAMS.separation = 2.0;
    PARAMS.alignment = 0;
    PARAMS.cohesion = 0;
    PARAMS.maxSpeed = 2.35;
    PARAMS.maxForce = 0.085;
    PARAMS.noiseStrength = 0.15;
  }
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

// Recalculate epistemic state and update target params.
// Called after each token chunk arrives.
function updateEpistemicTargets() {
  // ─── Token speed → breathing rate (maxSpeed) ───
  // Fast tokens = energized, faster movement. Slow tokens = contemplative.
  const speed = EPI.tokenSpeed; // tokens/sec
  const speedNorm = Math.min(speed / 30, 1.0); // normalize ~30 tokens/sec as "fast"
  EPI.target.maxSpeed = 1.2 + speedNorm * 1.8; // 1.2 (calm) → 3.0 (energized)

  // ─── Sentiment → color, trail, separation ───
  // Positive sentiment = calm, tight, royal blue
  // Negative sentiment = anxious, scattered, red, max separation
  const sent = EPI.sentimentNormalized; // -1 to 1
  const SENTIMENT_THRESHOLD = -0.1;

  // Map sentiment to separation: positive (1.0) → 1.0, negative (-1.0) → 4.0 (max)
  // Clamp sent to range first, then lerp from 1.0 (positive) to 4.0 (negative)
  const sentClamped = Math.max(-1, Math.min(1, sent));
  EPI.target.separation = 1.0 + (1 - sentClamped) / 2 * 3.0; // 1.0 (positive) → 4.0 (negative)

  if (sent < SENTIMENT_THRESHOLD) {
    // Negative — red, anxious
    EPI.target.baseHue = 0;           // red
    EPI.target.saturation = 80;       // strong red
    EPI.target.brightness = 90;       // bright
    EPI.target.trailOpacity = 1;      // minimal trail — exposed, raw
  } else {
    // Positive/calm — royal blue palette
    EPI.target.baseHue = 220;       // royal blue
    EPI.target.saturation = 60;
    EPI.target.brightness = 85;
    EPI.target.trailOpacity = 18;
  }

  // ─── Hedging density → flock cohesion/drift (still contributes) ───
  // High hedging = flock loses cohesion, drifts apart, noise increases.
  const hedge = EPI.hedgeDensity; // 0 to 1
  EPI.target.cohesion = 0.6 - hedge * 0.5;      // 0.6 (tight) → 0.1 (loose)
  EPI.target.alignment = 0.8 - hedge * 0.5;      // 0.8 (aligned) → 0.3 (drifting)
  EPI.target.noiseStrength = hedge * 0.4;        // 0 → 0.4 (turbulent when hedging)

  // ─── Stage-based overrides ───
  if (EPI.stage === 'idle') {
    // No input — tight, slow, calm white
    EPI.target.maxSpeed = 0.6;
    EPI.target.cohesion = 0.9;
    EPI.target.alignment = 1.0;
    EPI.target.separation = 1.0;
    EPI.target.noiseStrength = 0.02;
    EPI.target.baseHue = 0;           // white
    EPI.target.saturation = 0;
    EPI.target.brightness = 100;
    EPI.target.trailOpacity = 18;
  } else if (EPI.stage === 'anticipating') {
    // Stage 1: prompt sent, waiting for stream. Tense, compressed, slow breathing.
    EPI.target.maxSpeed = 0.8;
    EPI.target.cohesion = 0.9;
    EPI.target.alignment = 1.0;
    EPI.target.separation = 1.0;
    EPI.target.noiseStrength = 0.05;
  } else if (EPI.stage === 'settling') {
    // Stage 4: stream complete. Show "activation" — speed up briefly.
    EPI.target.maxSpeed = 2.5;
    EPI.target.noiseStrength = 0.0;
    // Keep cohesion/alignment at whatever the final sentiment/hedging dictated
    // but ease toward calm
    EPI.target.cohesion = Math.max(EPI.target.cohesion, 0.5);
    EPI.target.alignment = Math.max(EPI.target.alignment, 0.6);
  }
}

// ═══════════════════════════════════════════════════════════════════════════
//  SENTIMENT ANALYSIS
// ═══════════════════════════════════════════════════════════════════════════

function analyzeSentiment(text) {
  if (!text || text.trim().length === 0) return { score: 0, normalized: 0 };

  const result = sentiment.analyze(text);
  const score = result.score;                        // raw score
  // Normalize: divide by word count to get density, then scale
  const wordCount = text.split(/\s+/).length;
  const normalized = Math.max(-1, Math.min(1, score / Math.max(wordCount * 0.5, 1)));

  return { score, normalized };
}

function countHedges(text) {
  if (!text || text.trim().length === 0) return { count: 0, density: 0 };

  const lowerText = text.toLowerCase();
  const wordCount = text.split(/\s+/).length;
  let count = 0;

  for (const hedge of HEDGE_WORDS) {
    // Use word-boundary matching for single words
    const escaped = hedge.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const regex = new RegExp('\\b' + escaped + '\\b', 'gi');
    const matches = lowerText.match(regex);
    if (matches) count += matches.length;
  }

  const density = Math.min(count / Math.max(wordCount * 0.1, 1), 1.0);
  return { count, density };
}

// ═══════════════════════════════════════════════════════════════════════════
//  TOKEN SPEED CALCULATION
// ═══════════════════════════════════════════════════════════════════════════

function updateTokenSpeed() {
  const now = performance.now();
  // Prune chunks older than 2 seconds
  EPI.tokenChunks = EPI.tokenChunks.filter(t => now - t < 2000);

  if (EPI.tokenChunks.length < 2) {
    EPI.tokenSpeed = 0;
    return;
  }

  const span = (now - EPI.tokenChunks[0]) / 1000; // seconds
  EPI.tokenSpeed = EPI.tokenChunks.length / Math.max(span, 0.1);
}

// ═══════════════════════════════════════════════════════════════════════════
//  ANTHROPIC STREAMING CLIENT
// ═══════════════════════════════════════════════════════════════════════════

const API_KEY = import.meta.env.VITE_ANTHROPIC_API_KEY;
const API_URL = 'https://api.anthropic.com/v1/messages';
const MODEL = 'claude-sonnet-4-5-20250929';

async function streamPrompt(promptText) {
  if (!API_KEY) {
    console.error('No API key found. Set VITE_ANTHROPIC_API_KEY in .env');
    addSystemMessage('No API key — set VITE_ANTHROPIC_API_KEY in .env', true);
    setStage('idle');
    return;
  }

  // Cancel any pending idle-return timer from a previous response
  if (settleTimer) {
    clearTimeout(settleTimer);
    settleTimer = null;
  }
  idleTransitionStart = null;

  // Add user message to chat
  addUserMessage(promptText);

  // Reset epistemic state
  EPI.text = '';
  EPI.tokenCount = 0;
  EPI.tokenChunks = [];
  EPI.sentimentScore = 0;
  EPI.sentimentNormalized = 0;
  EPI.hedgeDensity = 0;
  EPI.tokenSpeed = 0;
  EPI.stopReason = null;
  EPI.streamStartTime = performance.now();
  EPI.lastChunkTime = 0;

  // Stage 1: Anticipation
  setStage('anticipating');
  updateEpistemicTargets();

  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': API_KEY,
        'anthropic-version': '2023-06-01',
        'anthropic-dangerous-direct-browser-access': 'true',
      },
      body: JSON.stringify({
        model: MODEL,
        max_tokens: 1024,
        stream: true,
        messages: [{ role: 'user', content: promptText }],
      }),
    });

    if (!response.ok) {
      const errText = await response.text();
      console.error('API error:', response.status, errText);
      addSystemMessage('API error: ' + response.status, true);
      setStage('idle');
      return;
    }

    // Stage 2: Reacting — stream is open
    setStage('reacting');

    // Create assistant message element for streaming
    const assistantEl = addAssistantMessage('');
    assistantEl.classList.add('streaming');

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Process complete SSE lines
      const lines = buffer.split('\n');
      buffer = lines.pop(); // keep incomplete line in buffer

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const data = line.slice(6).trim();
        if (!data || data === '[DONE]') continue;

        try {
          const event = JSON.parse(data);

          if (event.type === 'content_block_delta') {
            const delta = event.delta;
            if (delta && delta.type === 'text_delta' && delta.text) {
              handleTokenChunk(delta.text);
              // Append text to the assistant message element
              assistantEl.textContent += delta.text;
              scrollMessagesToBottom();
            }
          } else if (event.type === 'message_start') {
            // Stream is live — transition to streaming stage
            setStage('streaming');
          } else if (event.type === 'message_delta') {
            if (event.delta && event.delta.stop_reason) {
              EPI.stopReason = event.delta.stop_reason;
            }
          } else if (event.type === 'message_stop') {
            // Stage 4: Settle
            assistantEl.classList.remove('streaming');
            handleStreamComplete();
          }
        } catch (e) {
          // Incomplete JSON, skip
        }
      }
    }

    // If we never got a message_stop event, handle completion
    if (EPI.stage !== 'settling' && EPI.stage !== 'idle') {
      assistantEl.classList.remove('streaming');
      if (!EPI.stopReason) EPI.stopReason = 'end_turn';
      handleStreamComplete();
    }

  } catch (err) {
    console.error('Stream error:', err);
    addSystemMessage('Stream error: ' + err.message, true);
    setStage('idle');
  }
}

function handleTokenChunk(text) {
  EPI.text += text;
  EPI.tokenCount++;
  EPI.tokenChunks.push(performance.now());
  EPI.lastChunkTime = performance.now();

  // Update token speed
  updateTokenSpeed();

  // Re-analyze sentiment on accumulated text (throttled — every ~5 chunks)
  if (EPI.tokenCount % 5 === 0 || EPI.tokenCount < 10) {
    const sent = analyzeSentiment(EPI.text);
    EPI.sentimentScore = sent.score;
    EPI.sentimentNormalized = sent.normalized;

    const hedge = countHedges(EPI.text);
    EPI.hedgeDensity = hedge.density;
  }

  // Update target params
  updateEpistemicTargets();
}

// ─── IDLE TIMER ─────────────────────────────────────────────────────────────
// After stream completes, wait 4 seconds then return to idle state with a
// 2-second eased animation (so the transition is visible, not instant).
let settleTimer = null;
let idleTransitionStart = null;
const IDLE_TRANSITION_DURATION = 2000; // ms

function handleStreamComplete() {
  setStage('settling');

  // Final analysis on complete text
  const sent = analyzeSentiment(EPI.text);
  EPI.sentimentScore = sent.score;
  EPI.sentimentNormalized = sent.normalized;

  const hedge = countHedges(EPI.text);
  EPI.hedgeDensity = hedge.density;

  updateEpistemicTargets();

  // Save assistant response to chat history
  saveAssistantMessage(EPI.text);

  // Haptic pulse on supported devices
  if (navigator.vibrate) {
    navigator.vibrate(50);
  }

  // Re-enable input
  const input = document.getElementById('prompt-input');
  const submit = document.getElementById('prompt-submit');
  if (input) input.disabled = false;
  if (submit) submit.disabled = false;
  if (input) input.focus();

  // After 4 seconds, return to idle state — animate transition over 2 seconds
  if (settleTimer) clearTimeout(settleTimer);
  settleTimer = setTimeout(() => {
    idleTransitionStart = performance.now();
    setStage('idle');
    settleTimer = null;
  }, 4000);
}

// ═══════════════════════════════════════════════════════════════════════════
//  STAGE MANAGEMENT
// ═══════════════════════════════════════════════════════════════════════════

function setStage(stage) {
  EPI.stage = stage;
  updateEpistemicTargets();
}

// ═══════════════════════════════════════════════════════════════════════════
//  STATUS READOUT
// ═══════════════════════════════════════════════════════════════════════════

function updateStatus() {
  const el = (id) => document.getElementById(id);
  if (!el('s-stage')) return;

  el('s-stage').textContent = EPI.stage;
  el('s-tokens').textContent = EPI.tokenCount;

  const sentLabel = EPI.sentimentNormalized > 0.1 ? '+' : '';
  el('s-sentiment').textContent = EPI.sentimentNormalized !== 0
    ? `${sentLabel}${EPI.sentimentNormalized.toFixed(2)}`
    : '—';

  el('s-hedging').textContent = EPI.hedgeDensity > 0
    ? EPI.hedgeDensity.toFixed(2)
    : '—';

  el('s-stop').textContent = EPI.stopReason || '—';

  // Color the stage based on state
  const stageEl = el('s-stage');
  stageEl.className = 'metric-val';
  if (EPI.stage === 'anticipating') stageEl.classList.add('accent');
  else if (EPI.stage === 'streaming') stageEl.classList.add('accent');
  else if (EPI.stage === 'settling') stageEl.classList.add('warn');
}

// ═══════════════════════════════════════════════════════════════════════════
//  TWEAKPANE PANEL
// ═══════════════════════════════════════════════════════════════════════════

const pane = new Pane({ container: document.getElementById('panel') });

// Signal mapping
const signalFolder = pane.addFolder({ title: 'Signal', expanded: true });
signalFolder.addBinding(PARAMS, 'signalStrength', { min: 0, max: 1, step: 0.05, label: 'AI Drive' });
signalFolder.addBinding(PARAMS, 'smoothing', { min: 0.01, max: 0.2, step: 0.01, label: 'Smoothing' });

// Flocking (live values — driven by AI, but you can override)
const flockFolder = pane.addFolder({ title: 'Flocking', expanded: true });
flockFolder.addBinding(PARAMS, 'separation', { min: 0, max: 4, step: 0.05, label: 'Separation' });
flockFolder.addBinding(PARAMS, 'alignment', { min: 0, max: 4, step: 0.05, label: 'Alignment' });
flockFolder.addBinding(PARAMS, 'cohesion', { min: 0, max: 4, step: 0.05, label: 'Cohesion' });
flockFolder.addBinding(PARAMS, 'noiseStrength', { min: 0, max: 1, step: 0.05, label: 'Noise' });

// Color
const colorFolder = pane.addFolder({ title: 'Color', expanded: false });
colorFolder.addBinding(PARAMS, 'baseHue', { min: 0, max: 360, step: 1, label: 'Hue' });
colorFolder.addBinding(PARAMS, 'brightness', { min: 20, max: 100, step: 1, label: 'Brightness' });
colorFolder.addBinding(PARAMS, 'saturation', { min: 0, max: 100, step: 1, label: 'Saturation' });
colorFolder.addBinding(PARAMS, 'trailOpacity', { min: 1, max: 60, step: 1, label: 'Trail' });

// Motion
const motionFolder = pane.addFolder({ title: 'Motion', expanded: false });
motionFolder.addBinding(PARAMS, 'maxSpeed', { min: 0.2, max: 8, step: 0.1, label: 'Max Speed' });
motionFolder.addBinding(PARAMS, 'maxForce', { min: 0.01, max: 0.3, step: 0.005, label: 'Max Force' });
motionFolder.addBinding(PARAMS, 'drag', { min: 0.85, max: 1.0, step: 0.005, label: 'Drag' });

// Shape
const shapeFolder = pane.addFolder({ title: 'Shape', expanded: false });
shapeFolder.addBinding(PARAMS, 'count', { min: 100, max: 3000, step: 50, label: 'Count' })
  .on('change', () => { needsReset = true; });
shapeFolder.addBinding(PARAMS, 'size', { min: 1, max: 12, step: 0.1, label: 'Size' });
shapeFolder.addBinding(PARAMS, 'sizeVariance', { min: 0, max: 1, step: 0.05, label: 'Size Variance' });

// Interaction (read-only — not in scope for current testing)
const interFolder = pane.addFolder({ title: 'Interaction', expanded: false });
interFolder.addBinding(PARAMS, 'mouseAttract', { label: 'Mouse Attract', readonly: true });
interFolder.addBinding(PARAMS, 'mouseRadius', { min: 20, max: 400, step: 10, label: 'Mouse Radius', readonly: true });
interFolder.addBinding(PARAMS, 'mouseForce', { min: 0.1, max: 3, step: 0.1, label: 'Mouse Force', readonly: true });

// Actions
pane.addButton({ title: 'Restart' }).on('click', () => { needsReset = true; });

// ═══════════════════════════════════════════════════════════════════════════
//  CHAT MESSAGE RENDERING + PERSISTENCE
// ═══════════════════════════════════════════════════════════════════════════

const messagesEl = document.getElementById('messages');
const STORAGE_KEY = 'felt-chat-history';

function saveMessage(type, text, error = false) {
  const history = loadHistory();
  history.push({ type, text, error });
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
  } catch (e) {
    // Storage full or unavailable — silently skip
  }
}

function loadHistory() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch (e) {
    return [];
  }
}

function clearHistory() {
  localStorage.removeItem(STORAGE_KEY);
}

function renderHistory() {
  const history = loadHistory();
  for (const msg of history) {
    const el = document.createElement('div');
    if (msg.type === 'user') {
      el.className = 'msg msg-user';
    } else if (msg.type === 'assistant') {
      el.className = 'msg msg-assistant';
    } else if (msg.type === 'system') {
      el.className = 'msg msg-system' + (msg.error ? ' error' : '');
    }
    el.textContent = msg.text;
    messagesEl.appendChild(el);
  }
  if (history.length > 0) scrollMessagesToBottom();
}

function addUserMessage(text) {
  const el = document.createElement('div');
  el.className = 'msg msg-user';
  el.textContent = text;
  messagesEl.appendChild(el);
  saveMessage('user', text);
  scrollMessagesToBottom();
}

function addAssistantMessage(text) {
  const el = document.createElement('div');
  el.className = 'msg msg-assistant';
  el.textContent = text;
  messagesEl.appendChild(el);
  scrollMessagesToBottom();
  return el;
}

// Called when streaming completes to save the full assistant response
function saveAssistantMessage(text) {
  saveMessage('assistant', text);
}

function addSystemMessage(text, isError = false) {
  const el = document.createElement('div');
  el.className = 'msg msg-system' + (isError ? ' error' : '');
  el.textContent = text;
  messagesEl.appendChild(el);
  saveMessage('system', text, isError);
  scrollMessagesToBottom();
}

function scrollMessagesToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

// Load chat history on page load
renderHistory();

// ═══════════════════════════════════════════════════════════════════════════
//  PROMPT INPUT WIRING
// ═══════════════════════════════════════════════════════════════════════════

const promptForm = document.getElementById('prompt-form');
const promptInput = document.getElementById('prompt-input');
const promptSubmit = document.getElementById('prompt-submit');

if (promptForm) {
  promptForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const text = promptInput.value.trim();
    if (!text || EPI.stage === 'anticipating' || EPI.stage === 'reacting' || EPI.stage === 'streaming') return;

    promptInput.disabled = true;
    promptSubmit.disabled = true;
    streamPrompt(text);
  });
}

// Focus the input on load
if (promptInput) promptInput.focus();
