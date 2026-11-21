import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import type { NavSection } from '../types/echomind';

interface NeuralNodeMeta {
  id: string;
  label: string;
  stage: 'Encoder' | 'Fusion' | 'Task Head';
  description: string;
  targetSection: NavSection;
  color: number;
  position: [number, number, number];
}

const ARCH_NODES: NeuralNodeMeta[] = [
  {
    id: 'enc-visual',
    label: 'Visual Frame Encoder',
    stage: 'Encoder',
    description: 'Extracts sampled RGB facial regions, sharpness variance & temporal deltas',
    targetSection: 'forensics',
    color: 0x3b82f6,
    position: [-5.2, 2.0, 0.8],
  },
  {
    id: 'enc-audio',
    label: 'Acoustic Prosody Encoder',
    stage: 'Encoder',
    description: '16kHz PCM DFT spectral bands, RMS envelope, pitch & vocoder inspection',
    targetSection: 'forensics',
    color: 0x60a5fa,
    position: [-5.2, 0.0, -0.6],
  },
  {
    id: 'enc-text',
    label: 'Transcript Context Encoder',
    stage: 'Encoder',
    description: 'Encodes dialogue turns and subtitles when textual modality is available',
    targetSection: 'emotion',
    color: 0x818cf8,
    position: [-5.2, -2.0, 0.8],
  },
  {
    id: 'fusion-core',
    label: 'Cross-Attention Fusion Core',
    stage: 'Fusion',
    description: '256-d modality projections fused with 8-head cross-attention & temporal layer',
    targetSection: 'architecture',
    color: 0x8b5cf6,
    position: [0, 0, 0],
  },
  {
    id: 'head-forensics',
    label: 'Deepfake Forensics Head',
    stage: 'Task Head',
    description: 'Authenticity scoring, AV consistency & visual saliency region attribution',
    targetSection: 'forensics',
    color: 0x38bdf8,
    position: [5.0, 2.1, 1.0],
  },
  {
    id: 'head-emotion',
    label: 'Emotion Intelligence Head',
    stage: 'Task Head',
    description: '7-class probabilistic affective timeline with modality disagreement tracking',
    targetSection: 'emotion',
    color: 0xa78bfa,
    position: [5.0, 0.7, -1.0],
  },
  {
    id: 'head-continual',
    label: 'Continual Learning Head',
    stage: 'Task Head',
    description: 'EWC Fisher Information regularization & stratified Experience Replay buffer',
    targetSection: 'learning',
    color: 0x34d399,
    position: [5.0, -0.7, 1.0],
  },
  {
    id: 'head-unlearning',
    label: 'Selective Unlearning Head',
    stage: 'Task Head',
    description: 'SISA shard isolation, Exact Retrain & Gradient Ascent parameter scrubbing',
    targetSection: 'unlearning',
    color: 0xf59e0b,
    position: [5.0, -2.1, -0.8],
  },
];

interface NeuralNetworkCanvasProps {
  onNavigate?: (section: NavSection) => void;
  compact?: boolean;
}

export function NeuralNetworkCanvas({ onNavigate, compact = false }: NeuralNetworkCanvasProps) {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const navigateRef = useRef(onNavigate);
  navigateRef.current = onNavigate;

  const [selectedNode, setSelectedNode] = useState<NeuralNodeMeta>(ARCH_NODES[3]);
  const [controlsCollapsed, setControlsCollapsed] = useState(false);

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const isSoftwareGl = (() => {
      try {
        const probe = document.createElement('canvas').getContext('webgl2');
        if (!probe) return true;
        const ext = probe.getExtension('WEBGL_debug_renderer_info');
        const param = ext ? ext.UNMASKED_RENDERER_WEBGL : probe.RENDERER;
        const gpu = String(probe.getParameter(param));
        probe.getExtension('WEBGL_lose_context')?.loseContext();
        return /SwiftShader|llvmpipe|softpipe|Software|Basic Render/i.test(gpu);
      } catch {
        return true;
      }
    })();

    const width = Math.max(320, container.clientWidth);
    const height = Math.max(260, container.clientHeight);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b0d10);
    scene.fog = new THREE.FogExp2(0x0b0d10, 0.032);

    const aspect = width / height;
    const camera = new THREE.PerspectiveCamera(42, aspect, 0.1, 120);
    camera.position.set(0.3, 2.2, aspect < 1 ? 18.2 : 13.8);

    const renderer = new THREE.WebGLRenderer({
      antialias: !isSoftwareGl,
      powerPreference: 'high-performance',
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(isSoftwareGl ? 1 : Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;
    renderer.shadowMap.enabled = !isSoftwareGl;
    container.innerHTML = '';
    container.appendChild(renderer.domElement);

    const pmremGenerator = new THREE.PMREMGenerator(renderer);
    const envTexture = pmremGenerator.fromScene(new RoomEnvironment(), 0.04).texture;
    scene.environment = envTexture;
    pmremGenerator.dispose();

    // Warm key light + cool hemisphere fill
    const hemi = new THREE.HemisphereLight(0x93c5fd, 0x0f172a, 0.85);
    scene.add(hemi);

    const keyLight = new THREE.DirectionalLight(0xfff4e6, 2.0);
    keyLight.position.set(6, 10, 8);
    scene.add(keyLight);

    const rimLight = new THREE.DirectionalLight(0x6366f1, 1.1);
    rimLight.position.set(-8, -4, -6);
    scene.add(rimLight);

    // Subtle spatial reference grid beneath the architecture
    const grid = new THREE.GridHelper(22, 22, 0x262c38, 0x161a22);
    grid.position.y = -3.4;
    scene.add(grid);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = !isSoftwareGl;
    controls.dampingFactor = 0.06;
    controls.minDistance = 7;
    controls.maxDistance = 22;
    controls.maxPolarAngle = Math.PI * 0.65;
    controls.target.set(0, 0, 0);

    // Build composite 3-part node assemblies (avoiding single primitive placeholders)
    const interactables: THREE.Object3D[] = [];
    const nodeGroups: THREE.Group[] = [];
    const disposableGeos: THREE.BufferGeometry[] = [];
    const disposableMats: THREE.Material[] = [];

    const coreGeo = new THREE.IcosahedronGeometry(0.46, 1);
    const ringGeo = new THREE.TorusGeometry(0.68, 0.045, 12, 32);
    const collarGeo = new THREE.CylinderGeometry(0.22, 0.3, 0.14, 16);
    const fusionCoreGeo = new THREE.OctahedronGeometry(0.85, 1);
    const fusionRingOuterGeo = new THREE.TorusGeometry(1.35, 0.06, 14, 48);
    const fusionRingInnerGeo = new THREE.TorusGeometry(1.08, 0.04, 12, 40);
    disposableGeos.push(
      coreGeo,
      ringGeo,
      collarGeo,
      fusionCoreGeo,
      fusionRingOuterGeo,
      fusionRingInnerGeo
    );

    const chassisMat = new THREE.MeshStandardMaterial({
      color: 0x1f2530,
      metalness: 0.85,
      roughness: 0.25,
    });
    disposableMats.push(chassisMat);

    for (const meta of ARCH_NODES) {
      const group = new THREE.Group();
      group.position.set(...meta.position);

      const accentMat = new THREE.MeshStandardMaterial({
        color: meta.color,
        emissive: meta.color,
        emissiveIntensity: 0.38,
        metalness: 0.35,
        roughness: 0.22,
      });
      const ringMat = new THREE.MeshStandardMaterial({
        color: 0x9ca3af,
        metalness: 0.88,
        roughness: 0.18,
      });
      disposableMats.push(accentMat, ringMat);

      if (meta.id === 'fusion-core') {
        const core = new THREE.Mesh(fusionCoreGeo, accentMat);
        const outerRing = new THREE.Mesh(fusionRingOuterGeo, ringMat);
        const innerRing = new THREE.Mesh(fusionRingInnerGeo, chassisMat);
        outerRing.rotation.x = Math.PI * 0.35;
        innerRing.rotation.y = Math.PI * 0.4;
        group.add(core, outerRing, innerRing);
      } else {
        const core = new THREE.Mesh(coreGeo, accentMat);
        const ring = new THREE.Mesh(ringGeo, ringMat);
        ring.rotation.x = Math.PI / 2.4;
        const collarTop = new THREE.Mesh(collarGeo, chassisMat);
        collarTop.position.y = 0.5;
        const collarBottom = new THREE.Mesh(collarGeo, chassisMat);
        collarBottom.position.y = -0.5;
        collarBottom.rotation.z = Math.PI;
        group.add(core, ring, collarTop, collarBottom);
      }

      // Proxy hit volume on layer 1 for fast raycasting
      const proxyGeo = new THREE.SphereGeometry(meta.id === 'fusion-core' ? 1.4 : 0.85, 10, 10);
      const proxyMat = new THREE.MeshBasicMaterial({ visible: false });
      disposableGeos.push(proxyGeo);
      disposableMats.push(proxyMat);
      const hitProxy = new THREE.Mesh(proxyGeo, proxyMat);
      hitProxy.layers.set(1);
      hitProxy.userData.nodeMeta = meta;
      group.add(hitProxy);
      interactables.push(hitProxy);

      group.userData.meta = meta;
      nodeGroups.push(group);
      scene.add(group);
    }

    // Synaptic pathways connecting Encoders -> Fusion Core -> Task Heads
    const fusionPos = new THREE.Vector3(0, 0, 0);
    const pathways: { curve: THREE.QuadraticBezierCurve3; color: number }[] = [];

    for (const meta of ARCH_NODES) {
      if (meta.id === 'fusion-core') continue;
      const nodeVec = new THREE.Vector3(...meta.position);
      const mid = nodeVec.clone().lerp(fusionPos, 0.5);
      mid.y += meta.stage === 'Encoder' ? 0.55 : -0.45;
      const curve =
        meta.stage === 'Encoder'
          ? new THREE.QuadraticBezierCurve3(nodeVec, mid, fusionPos)
          : new THREE.QuadraticBezierCurve3(fusionPos, mid, nodeVec);
      pathways.push({ curve, color: meta.color });

      const tubeGeo = new THREE.TubeGeometry(curve, 24, 0.028, 8, false);
      const tubeMat = new THREE.MeshStandardMaterial({
        color: meta.color,
        emissive: meta.color,
        emissiveIntensity: 0.18,
        roughness: 0.4,
        metalness: 0.5,
        transparent: true,
        opacity: 0.62,
      });
      disposableGeos.push(tubeGeo);
      disposableMats.push(tubeMat);
      scene.add(new THREE.Mesh(tubeGeo, tubeMat));
    }

    // Instanced signal pulse spheres traveling along pathways
    const pulsesPerPath = 4;
    const totalPulses = pathways.length * pulsesPerPath;
    const pulseGeo = new THREE.SphereGeometry(0.08, 10, 10);
    const pulseMat = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      emissive: 0x60a5fa,
      emissiveIntensity: 0.9,
      roughness: 0.1,
      metalness: 0.2,
    });
    disposableGeos.push(pulseGeo);
    disposableMats.push(pulseMat);

    const pulseInstanced = new THREE.InstancedMesh(pulseGeo, pulseMat, totalPulses);
    pulseInstanced.frustumCulled = false;
    const _dummy = new THREE.Object3D();
    const _pos = new THREE.Vector3();
    const _color = new THREE.Color();

    pathways.forEach((p, pIdx) => {
      _color.setHex(p.color);
      for (let k = 0; k < pulsesPerPath; k++) {
        pulseInstanced.setColorAt(pIdx * pulsesPerPath + k, _color);
      }
    });
    if (pulseInstanced.instanceColor) pulseInstanced.instanceColor.needsUpdate = true;
    scene.add(pulseInstanced);

    // Raycaster for node selection
    const raycaster = new THREE.Raycaster();
    raycaster.layers.set(1);
    const pointer = new THREE.Vector2();
    let pointerDownPos = { x: 0, y: 0 };

    const onPointerDown = (e: PointerEvent) => {
      pointerDownPos = { x: e.clientX, y: e.clientY };
    };

    const onPointerUp = (e: PointerEvent) => {
      const dist = Math.hypot(e.clientX - pointerDownPos.x, e.clientY - pointerDownPos.y);
      if (dist > 6) return; // Was dragging orbit camera
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const hits = raycaster.intersectObjects(interactables, false);
      if (hits.length > 0) {
        const meta = hits[0].object.userData.nodeMeta as NeuralNodeMeta | undefined;
        if (meta) setSelectedNode(meta);
      }
    };

    renderer.domElement.addEventListener('pointerdown', onPointerDown);
    renderer.domElement.addEventListener('pointerup', onPointerUp);

    const onResize = () => {
      if (!container) return;
      const w = Math.max(320, container.clientWidth);
      const h = Math.max(260, container.clientHeight);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', onResize);

    if (!isSoftwareGl) {
      renderer.compile(scene, camera);
    }

    let frameId = 0;
    let disposed = false;
    const clock = new THREE.Clock();

    function afterDraw(cb: () => void) {
      const gl = renderer.getContext() as WebGL2RenderingContext;
      const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
      gl.flush();
      const poll = () => {
        if (disposed) {
          if (sync) gl.deleteSync(sync);
          return;
        }
        const state = sync && gl.getSyncParameter(sync, gl.SYNC_STATUS);
        if (state === gl.UNSIGNALED) {
          frameId = window.setTimeout(poll, 5);
        } else {
          if (sync) gl.deleteSync(sync);
          frameId = window.setTimeout(cb, 80);
        }
      };
      frameId = window.setTimeout(poll, 5);
    }

    function tick() {
      if (disposed) return;
      const elapsed = clock.getElapsedTime();

      for (let i = 0; i < nodeGroups.length; i++) {
        const g = nodeGroups[i];
        if (g.userData.meta?.id === 'fusion-core') {
          g.rotation.y = elapsed * 0.35;
          if (g.children[1]) g.children[1].rotation.z = elapsed * 0.45;
          if (g.children[2]) g.children[2].rotation.x = -elapsed * 0.32;
        } else {
          g.rotation.y = elapsed * 0.25 + i * 0.6;
          if (g.children[1]) g.children[1].rotation.z = elapsed * 0.5;
        }
      }

      let instIdx = 0;
      for (let pIdx = 0; pIdx < pathways.length; pIdx++) {
        const pathItem = pathways[pIdx];
        for (let k = 0; k < pulsesPerPath; k++) {
          const t = (elapsed * 0.28 + k / pulsesPerPath + pIdx * 0.11) % 1;
          pathItem.curve.getPoint(t, _pos);
          _dummy.position.copy(_pos);
          const s = 0.8 + 0.35 * Math.sin(t * Math.PI);
          _dummy.scale.setScalar(s);
          _dummy.updateMatrix();
          pulseInstanced.setMatrixAt(instIdx++, _dummy.matrix);
        }
      }
      pulseInstanced.instanceMatrix.needsUpdate = true;

      controls.update();
      renderer.render(scene, camera);

      if (isSoftwareGl) {
        afterDraw(tick);
      } else {
        frameId = requestAnimationFrame(tick);
      }
    }

    tick();

    return () => {
      disposed = true;
      cancelAnimationFrame(frameId);
      clearTimeout(frameId);
      window.removeEventListener('resize', onResize);
      renderer.domElement.removeEventListener('pointerdown', onPointerDown);
      renderer.domElement.removeEventListener('pointerup', onPointerUp);
      controls.dispose();
      disposableGeos.forEach((g) => g.dispose());
      disposableMats.forEach((m) => m.dispose());
      envTexture.dispose();
      renderer.dispose();
    };
  }, []);

  return (
    <div
      className={`relative w-full overflow-hidden rounded-lg border border-white/[0.08] bg-[#0B0D10] ${
        compact ? 'h-[320px]' : 'h-[430px]'
      }`}
    >
      <div
        ref={mountRef}
        data-testid="neural-canvas"
        aria-label="Interactive 3D Multimodal Neural Architecture"
        className="h-full w-full"
      />

      {/* Top-left quiet spec label (opaque surface, no backdrop-filter over WebGL) */}
      <div className="pointer-events-none absolute top-3 left-3 rounded-md border border-white/[0.08] bg-[#12151A]/95 px-3 py-1.5">
        <div className="text-xs font-medium text-[#F3F4F6]">
          Shared Cross-Attention Topology
        </div>
        <div className="text-[11px] text-[#9CA3AF]">
          Drag to orbit · Scroll to zoom · Click any node to inspect
        </div>
      </div>

      {/* Collapsible bottom dock (zero viewport center occlusion) */}
      <div className="absolute bottom-3 left-1/2 w-[min(680px,94%)] -translate-x-1/2">
        <div className="mb-1 flex justify-center">
          <button
            type="button"
            onClick={() => setControlsCollapsed((prev) => !prev)}
            className="rounded border border-white/[0.1] bg-[#12151A] px-2.5 py-0.5 text-[11px] font-medium text-[#9CA3AF] hover:text-white transition-colors"
          >
            {controlsCollapsed ? 'Show Node Inspector' : 'Hide Node Inspector'}
          </button>
        </div>

        {!controlsCollapsed && (
          <div className="flex flex-col gap-2 rounded-lg border border-white/[0.1] bg-[#12151A] px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
            <div className="min-w-0">
              <div className="flex items-center gap-2 text-xs text-[#9CA3AF]">
                <span>{selectedNode.stage}</span>
                <span aria-hidden="true">·</span>
                <span className="font-mono-tabular text-[#60A5FA]">{selectedNode.id}</span>
              </div>
              <div className="truncate text-sm font-semibold text-[#F3F4F6]">
                {selectedNode.label}
              </div>
              <p className="truncate text-xs text-[#9CA3AF]">{selectedNode.description}</p>
            </div>

            {navigateRef.current && (
              <button
                type="button"
                onClick={() => navigateRef.current?.(selectedNode.targetSection)}
                className="shrink-0 whitespace-nowrap rounded-md bg-[#3B82F6] px-3.5 py-1.5 text-xs font-medium text-white hover:bg-[#2563EB] transition-colors"
              >
                Open Module
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
