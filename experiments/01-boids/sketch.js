import p5 from 'p5';
import { Pane } from 'tweakpane';

// ─── PARAMS ────────────────────────────────────────────────────────────────
const PARAMS = {
  // Shape
  count: 800,
  size: 3.5,
  sizeVariance: 0.6,

  // Flocking forces
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
  baseHue: 145,
  hueRange: 30,
  saturation: 40,
  brightness: 85,
  colorMode: 'speed', // speed | direction | uniform

  // Trail
  trailOpacity: 18,

  // Interaction
  mouseAttract: true,
  mouseRadius: 140,
  mouseForce: 0.8,
};

// ─── BOID CLASS ────────────────────────────────────────────────────────────
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

    // Noise
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
    this.size = PARAMS.size * (1 + (this.p.random(-PARAMS.sizeVariance, PARAMS.sizeVariance)) * 0.05 + this.size * 0.95 - PARAMS.size);
    this.size = Math.max(0.5, this.size);
  }

  getColor() {
    const p = this.p;
    let speed = this.vel.mag() / PARAMS.maxSpeed;
    let angle = this.vel.heading();

    if (PARAMS.colorMode === 'speed') {
      let h = PARAMS.baseHue + speed * PARAMS.hueRange;
      let b = 60 + speed * 35;
      return p.color(h, PARAMS.saturation, b);
    } else if (PARAMS.colorMode === 'direction') {
      let h = p.map(angle, -p.PI, p.PI, PARAMS.baseHue - PARAMS.hueRange, PARAMS.baseHue + PARAMS.hueRange);
      return p.color(h, PARAMS.saturation, PARAMS.brightness);
    } else {
      return p.color(PARAMS.baseHue, PARAMS.saturation * 0.6, PARAMS.brightness * 0.9);
    }
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

// ─── SKETCH ────────────────────────────────────────────────────────────────
let boids = [];
let needsReset = false;

new p5(function (p) {
  p.setup = function () {
    p.pixelDensity(Math.min(window.devicePixelRatio, 2));
    let cnv = p.createCanvas(p.windowWidth, p.windowHeight);
    cnv.style('display', 'block');
    p.colorMode(p.HSB, 360, 100, 100, 100);
    p.background(0, 0, 4);
    spawnBoids(p);
  };

  p.draw = function () {
    if (needsReset) { spawnBoids(p); p.background(0, 0, 4); needsReset = false; }

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
  };

  p.windowResized = function () {
    p.resizeCanvas(p.windowWidth, p.windowHeight);
  };
});

function spawnBoids(p) {
  boids = [];
  for (let i = 0; i < PARAMS.count; i++) {
    boids.push(new Boid(p));
  }
}

// ─── TWEAKPANE PANEL ───────────────────────────────────────────────────────
const pane = new Pane({ container: document.getElementById('panel') });

// Shape
const shapeFolder = pane.addFolder({ title: 'Shape', expanded: true });
shapeFolder.addBinding(PARAMS, 'count', { min: 100, max: 3000, step: 50, label: 'Count' })
  .on('change', () => { needsReset = true; });
shapeFolder.addBinding(PARAMS, 'size', { min: 1, max: 12, step: 0.1, label: 'Size' });
shapeFolder.addBinding(PARAMS, 'sizeVariance', { min: 0, max: 1, step: 0.05, label: 'Size Variance' });

// Flocking
const flockFolder = pane.addFolder({ title: 'Flocking', expanded: true });
flockFolder.addBinding(PARAMS, 'separation', { min: 0, max: 4, step: 0.05, label: 'Separation' });
flockFolder.addBinding(PARAMS, 'alignment', { min: 0, max: 4, step: 0.05, label: 'Alignment' });
flockFolder.addBinding(PARAMS, 'cohesion', { min: 0, max: 4, step: 0.05, label: 'Cohesion' });
flockFolder.addBinding(PARAMS, 'perceptionRadius', { min: 10, max: 200, step: 1, label: 'Vision' });
flockFolder.addBinding(PARAMS, 'separationRadius', { min: 5, max: 100, step: 1, label: 'Personal Space' });

// Motion
const motionFolder = pane.addFolder({ title: 'Motion', expanded: false });
motionFolder.addBinding(PARAMS, 'maxSpeed', { min: 0.2, max: 8, step: 0.1, label: 'Max Speed' });
motionFolder.addBinding(PARAMS, 'maxForce', { min: 0.01, max: 0.3, step: 0.005, label: 'Max Force' });
motionFolder.addBinding(PARAMS, 'drag', { min: 0.85, max: 1.0, step: 0.005, label: 'Drag' });
motionFolder.addBinding(PARAMS, 'noiseStrength', { min: 0, max: 1, step: 0.05, label: 'Noise' });

// Color
const colorFolder = pane.addFolder({ title: 'Color', expanded: false });
colorFolder.addBinding(PARAMS, 'colorMode', {
  label: 'Mode',
  options: { Speed: 'speed', Direction: 'direction', Uniform: 'uniform' }
});
colorFolder.addBinding(PARAMS, 'baseHue', { min: 0, max: 360, step: 1, label: 'Hue' });
colorFolder.addBinding(PARAMS, 'hueRange', { min: 0, max: 120, step: 1, label: 'Hue Range' });
colorFolder.addBinding(PARAMS, 'saturation', { min: 0, max: 100, step: 1, label: 'Saturation' });
colorFolder.addBinding(PARAMS, 'brightness', { min: 20, max: 100, step: 1, label: 'Brightness' });
colorFolder.addBinding(PARAMS, 'trailOpacity', { min: 1, max: 60, step: 1, label: 'Trail' });

// Interaction
const interFolder = pane.addFolder({ title: 'Interaction', expanded: false });
interFolder.addBinding(PARAMS, 'mouseAttract', { label: 'Mouse Attract' });
interFolder.addBinding(PARAMS, 'mouseRadius', { min: 20, max: 400, step: 10, label: 'Mouse Radius' });
interFolder.addBinding(PARAMS, 'mouseForce', { min: 0.1, max: 3, step: 0.1, label: 'Mouse Force' });

// Actions
pane.addButton({ title: 'Restart' }).on('click', () => { needsReset = true; });
