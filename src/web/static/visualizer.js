import * as THREE from "./vendor/three.module.js";

const TRACK_VARS = ["--track-a", "--track-b", "--track-c", "--track-d"];

export class TrackVisualizer {
  constructor(container) {
    this.container = container;
    this.scene = new THREE.Scene();
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, -2000, 2000);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.container.appendChild(this.renderer.domElement);

    this.world = new THREE.Group();
    this.trackGroup = new THREE.Group();
    this.headGroup = new THREE.Group();
    this.gridGroup = new THREE.Group();
    this.world.add(this.gridGroup, this.trackGroup, this.headGroup);
    this.scene.add(this.world);

    this.bounds = null;
    this.materials = [];
    this.isDragging = false;
    this.lastPointer = { x: 0, y: 0 };
    this.world.rotation.x = -0.18;
    this.world.rotation.z = 0.03;

    this._bind();
    this.syncTheme();
    this.resize();
    this.drawEmptyGrid();
    this.animate();
  }

  _bind() {
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.container);
    this.renderer.domElement.addEventListener("pointerdown", (event) => {
      this.isDragging = true;
      this.lastPointer = { x: event.clientX, y: event.clientY };
      this.renderer.domElement.setPointerCapture(event.pointerId);
    });
    this.renderer.domElement.addEventListener("pointermove", (event) => {
      if (!this.isDragging) return;
      const dx = event.clientX - this.lastPointer.x;
      const dy = event.clientY - this.lastPointer.y;
      this.world.rotation.z += dx * 0.004;
      this.world.rotation.x = clamp(this.world.rotation.x + dy * 0.003, -0.9, 0.55);
      this.lastPointer = { x: event.clientX, y: event.clientY };
    });
    this.renderer.domElement.addEventListener("pointerup", (event) => {
      this.isDragging = false;
      this.renderer.domElement.releasePointerCapture(event.pointerId);
    });
  }

  resize() {
    const width = Math.max(this.container.clientWidth, 320);
    const height = Math.max(this.container.clientHeight, 260);
    this.renderer.setSize(width, height, false);
    const aspect = width / height;
    const span = this.bounds ? this._sceneSpan(this.bounds) : 900;
    this.camera.left = (-span * aspect) / 2;
    this.camera.right = (span * aspect) / 2;
    this.camera.top = span / 2;
    this.camera.bottom = -span / 2;
    this.camera.position.set(0, -span * 0.82, span * 0.74);
    this.camera.lookAt(0, 0, 0);
    this.camera.updateProjectionMatrix();
  }

  syncTheme() {
    const styles = getComputedStyle(document.documentElement);
    this.renderer.setClearColor(new THREE.Color(cssVar(styles, "--three-bg", "#0a0f0e")), 1);
    this.gridColor = new THREE.Color(cssVar(styles, "--three-grid", "#244239"));
    this.trackColors = TRACK_VARS.map((name) => new THREE.Color(cssVar(styles, name, "#49e4ae")));
    for (const item of this.materials) {
      item.material.color.copy(this.trackColors[item.index % this.trackColors.length]);
    }
    for (const child of this.gridGroup.children) {
      if (child.material) child.material.color.copy(this.gridColor);
    }
  }

  drawEmptyGrid() {
    this._clearGroup(this.gridGroup);
    const grid = new THREE.GridHelper(760, 16, this.gridColor, this.gridColor);
    grid.material.transparent = true;
    grid.material.opacity = 0.34;
    this.gridGroup.add(grid);
    this.resize();
  }

  loadTracks(payload) {
    this._clearGroup(this.trackGroup);
    this._clearGroup(this.headGroup);
    this.materials = [];
    this.bounds = payload?.bounds || null;
    this.drawEmptyGrid();

    const tracks = payload?.tracks || [];
    if (!tracks.length || this.bounds?.minX === null || this.bounds?.minX === undefined) {
      this.resize();
      return;
    }

    const minFrame = Number(this.bounds.minFrame || 0);
    const maxFrame = Number(this.bounds.maxFrame || minFrame);
    const depth = Math.max(maxFrame - minFrame, 1) * 0.08;
    const centerX = (Number(this.bounds.minX) + Number(this.bounds.maxX)) / 2;
    const centerY = (Number(this.bounds.minY) + Number(this.bounds.maxY)) / 2;

    tracks.forEach((track, index) => {
      const color = this.trackColors[index % this.trackColors.length];
      const points = track.points.map((point) =>
        new THREE.Vector3(
          Number(point.x) - centerX,
          (Number(point.f) - minFrame) * 0.08 - depth / 2,
          -(Number(point.y) - centerY),
        ),
      );
      if (points.length < 2) return;
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      const material = new THREE.LineBasicMaterial({
        color,
        transparent: true,
        opacity: 0.86,
      });
      const line = new THREE.Line(geometry, material);
      line.userData = { id: track.id };
      this.trackGroup.add(line);
      this.materials.push({ material, index });

      const head = this._makeHead(points[points.length - 1], color, track.id);
      this.headGroup.add(head);
    });
    this.resize();
  }

  _makeHead(position, color, id) {
    const geometry = new THREE.SphereGeometry(6, 16, 16);
    const material = new THREE.MeshBasicMaterial({ color });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.copy(position);
    mesh.userData = { id };
    this.materials.push({ material, index: Number(id) || 0 });
    return mesh;
  }

  _sceneSpan(bounds) {
    const width = Number(bounds.maxX || 0) - Number(bounds.minX || 0);
    const height = Number(bounds.maxY || 0) - Number(bounds.minY || 0);
    const frames = Number(bounds.maxFrame || 0) - Number(bounds.minFrame || 0);
    return Math.max(width, height, frames * 0.09, 420) * 1.28;
  }

  _clearGroup(group) {
    for (const child of [...group.children]) {
      group.remove(child);
      child.geometry?.dispose();
      if (Array.isArray(child.material)) {
        child.material.forEach((material) => material.dispose?.());
      } else {
        child.material?.dispose?.();
      }
    }
  }

  animate() {
    requestAnimationFrame(() => this.animate());
    if (!this.isDragging && !window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      this.world.rotation.y = Math.sin(performance.now() / 4800) * 0.035;
      const scale = 1 + Math.sin(performance.now() / 520) * 0.045;
      this.headGroup.scale.setScalar(scale);
    }
    this.renderer.render(this.scene, this.camera);
  }
}

function cssVar(styles, name, fallback) {
  return styles.getPropertyValue(name).trim() || fallback;
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}
