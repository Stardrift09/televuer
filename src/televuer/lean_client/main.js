// TeleVuer lean WebXR client: shows the camera images head-locked in front of the user and sends
// the head / controller / hand poses to the PC once per XR frame. Protocol: see lean.py.
//
// The poses use the same conventions as the vuer client, so the PC side does not change:
//   reference space 'local-floor', matrices column-major
//   head        three.js' XR "union camera" (left eye, moved to between and slightly behind the eyes)
//   controller  targetRaySpace pose, buttons / axes in gamepad order
//   hand        the 25 joint matrices, wrist first
'use strict';
(() => {
  const canvas = document.getElementById('view');
  const button = document.getElementById('enter');
  const statusEl = document.getElementById('status');
  const WS_BASE = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host;

  let config = null;          // sent by the PC on every (re)connect
  let poseWs = null;
  let worker = null, videoOpen = false;
  let session = null, refSpace = null, arSession = false;
  let lastError = '';
  const supported = {'immersive-vr': false, 'immersive-ar': false};

  // ------------------------------------------------------------------ reporting to the PC
  let reports = 0;
  function report(message) {
    lastError = String(message);
    console.warn('[TeleVuer]', message);
    if (reports++ < 20 && poseWs && poseWs.readyState === WebSocket.OPEN) {
      poseWs.send(JSON.stringify({type: 'log', message: lastError}));
    }
    updateUI();
  }
  window.addEventListener('error', (e) => report(`${e.message} (${(e.filename || '').split('/').pop()}:${e.lineno})`));
  window.addEventListener('unhandledrejection', (e) => report(`unhandled: ${e.reason && e.reason.message || e.reason}`));

  // ------------------------------------------------------------------ GL: one textured quad per eye
  const gl = canvas.getContext('webgl2', {
    xrCompatible: true, antialias: false, alpha: true, depth: false, stencil: false, powerPreference: 'high-performance',
  });
  if (!gl) {
    statusEl.textContent = 'WebGL2 is not available in this browser.';
    return;
  }

  function shader(type, source) {
    const s = gl.createShader(type);
    gl.shaderSource(s, source);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  }
  const program = gl.createProgram();
  gl.attachShader(program, shader(gl.VERTEX_SHADER, `#version 300 es
    layout(location = 0) in vec2 a_pos;
    layout(location = 1) in vec2 a_uv;
    uniform mat4 u_mvp;
    out vec2 v_uv;
    void main() { v_uv = a_uv; gl_Position = u_mvp * vec4(a_pos, 0.0, 1.0); }`));
  gl.attachShader(program, shader(gl.FRAGMENT_SHADER, `#version 300 es
    precision mediump float;
    uniform sampler2D u_tex;
    in vec2 v_uv;
    out vec4 color;
    void main() { color = vec4(texture(u_tex, v_uv).rgb, 1.0); }`));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  gl.useProgram(program);
  gl.uniform1i(gl.getUniformLocation(program, 'u_tex'), 0);
  const uMvp = gl.getUniformLocation(program, 'u_mvp');

  // unit quad; v = 0 at the top because the images are uploaded as decoded, first row first (no flip)
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
    -0.5, -0.5, 0, 1,
     0.5, -0.5, 1, 1,
    -0.5,  0.5, 0, 0,
     0.5,  0.5, 1, 0,
  ]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 16, 0);
  gl.enableVertexAttribArray(1);
  gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 16, 8);

  const textures = [0, 1].map(() => {
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return {tex, width: 0, height: 0};
  });
  let eyes = 0;        // images per frame: 2 stereo, 1 mono
  let frameSeq = 0;    // seq of the camera frame now in the textures (0: none yet)

  function upload(i, bitmap) {
    const t = textures[i];
    gl.bindTexture(gl.TEXTURE_2D, t.tex);
    if (t.width !== bitmap.width || t.height !== bitmap.height) {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, bitmap.width, bitmap.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      t.width = bitmap.width;
      t.height = bitmap.height;
    }
    // the storage is reused: one upload per frame, nothing allocated, nothing left for the GC
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, gl.RGBA, gl.UNSIGNED_BYTE, bitmap);
  }

  function drawQuad(eye, mvp) {
    gl.bindTexture(gl.TEXTURE_2D, textures[eye].tex);
    gl.uniformMatrix4fv(uMvp, false, mvp);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  // ------------------------------------------------------------------ small column-major matrix helpers
  const head = new Float64Array(16);
  const model = new Float32Array(16), viewInv = new Float32Array(16), mv = new Float32Array(16), mvp = new Float32Array(16);

  function multiply(out, a, b) {  // out = a * b, out must not be a or b
    for (let c = 0; c < 16; c += 4) {
      const b0 = b[c], b1 = b[c + 1], b2 = b[c + 2], b3 = b[c + 3];
      out[c] = a[0] * b0 + a[4] * b1 + a[8] * b2 + a[12] * b3;
      out[c + 1] = a[1] * b0 + a[5] * b1 + a[9] * b2 + a[13] * b3;
      out[c + 2] = a[2] * b0 + a[6] * b1 + a[10] * b2 + a[14] * b3;
      out[c + 3] = a[3] * b0 + a[7] * b1 + a[11] * b2 + a[15] * b3;
    }
  }

  function invertRigid(out, m) {  // [R t] -> [R^T  -R^T t]
    const x = m[12], y = m[13], z = m[14];
    out[0] = m[0]; out[1] = m[4]; out[2] = m[8]; out[3] = 0;
    out[4] = m[1]; out[5] = m[5]; out[6] = m[9]; out[7] = 0;
    out[8] = m[2]; out[9] = m[6]; out[10] = m[10]; out[11] = 0;
    out[12] = -(m[0] * x + m[1] * y + m[2] * z);
    out[13] = -(m[4] * x + m[5] * y + m[6] * z);
    out[14] = -(m[8] * x + m[9] * y + m[10] * z);
    out[15] = 1;
  }

  // Head pose as the vuer client reports it: the camera three.js (WebXRManager.setProjectionFromUnion)
  // derives from the two eyes. Left eye orientation, position moved along the eye's x and z axes to the
  // apex of the frustum that contains both eye frusta.
  function unionCamera(views) {
    const L = views[0].transform.matrix;
    head.set(L);
    if (views.length === 2) {
      const R = views[1].transform.matrix, projL = views[0].projectionMatrix, projR = views[1].projectionMatrix;
      const ipd = Math.hypot(L[12] - R[12], L[13] - R[13], L[14] - R[14]);
      const leftFov = (projL[8] - 1) / projL[0], rightFov = (projR[8] + 1) / projR[0];
      const zOffset = ipd / (rightFov - leftFov), xOffset = zOffset * -leftFov;
      head[12] += xOffset * L[0] + zOffset * L[8];
      head[13] += xOffset * L[1] + zOffset * L[9];
      head[14] += xOffset * L[2] + zOffset * L[10];
    }
  }

  // image plane: width x height metres, `distance` metres in front of the head, facing it
  function planeModel(width, height, distance) {
    for (let i = 0; i < 3; i++) {
      model[i] = head[i] * width;
      model[4 + i] = head[4 + i] * height;
      model[8 + i] = head[8 + i];
      model[12 + i] = head[12 + i] - head[8 + i] * distance;
    }
    model[3] = model[7] = model[11] = 0;
    model[15] = 1;
  }

  // ------------------------------------------------------------------ poses -> PC
  const JOINTS = [
    'wrist',
    'thumb-metacarpal', 'thumb-phalanx-proximal', 'thumb-phalanx-distal', 'thumb-tip',
    'index-finger-metacarpal', 'index-finger-phalanx-proximal', 'index-finger-phalanx-intermediate', 'index-finger-phalanx-distal', 'index-finger-tip',
    'middle-finger-metacarpal', 'middle-finger-phalanx-proximal', 'middle-finger-phalanx-intermediate', 'middle-finger-phalanx-distal', 'middle-finger-tip',
    'ring-finger-metacarpal', 'ring-finger-phalanx-proximal', 'ring-finger-phalanx-intermediate', 'ring-finger-phalanx-distal', 'ring-finger-tip',
    'pinky-finger-metacarpal', 'pinky-finger-phalanx-proximal', 'pinky-finger-phalanx-intermediate', 'pinky-finger-phalanx-distal', 'pinky-finger-tip',
  ];
  const THUMB_TIP = 4, INDEX_TIP = 9, MIDDLE_PROXIMAL = 11, MIDDLE_TIP = 14;
  const HEAD_AT = 4, BLOCKS_AT = 20;            // float32 indices (bytes 16 and 80)
  const CTRL_FLOATS = 21, HAND_FLOATS = 402;    // per side

  function poseMessage(sideFloats, kind) {
    const buf = new ArrayBuffer((BLOCKS_AT + 2 * sideFloats) * 4);
    const dv = new DataView(buf), f32 = new Float32Array(buf);
    dv.setUint8(12, kind);
    return {buf, dv, f32};
  }
  const ctrlMsg = poseMessage(CTRL_FLOATS, 0);
  const handMsg = poseMessage(HAND_FLOATS, 1);
  handMsg.joints = [0, 1].map((k) => handMsg.f32.subarray(BLOCKS_AT + k * HAND_FLOATS, BLOCKS_AT + k * HAND_FLOATS + 400));
  const jointSpaces = new WeakMap();  // XRHand -> its 25 joint spaces

  function readController(frame, src, base) {
    if (src.targetRayMode !== 'tracked-pointer' || !src.targetRaySpace) return false;
    const pose = frame.getPose(src.targetRaySpace, refSpace);
    if (!pose) return false;
    const f = ctrlMsg.f32, buttons = src.gamepad.buttons, axes = src.gamepad.axes;
    f.set(pose.transform.matrix, base);
    f[base + 16] = buttons[0].value;  // trigger
    f[base + 17] = buttons[1].value;  // squeeze
    f[base + 18] = axes[2] || 0;      // thumbstick x, y
    f[base + 19] = axes[3] || 0;
    let pressed = 0;                  // bits: trigger, squeeze, touchpad, thumbstick, a, b
    for (let i = 0; i < 6; i++) if (buttons[i].pressed) pressed |= 1 << i;
    ctrlMsg.dv.setUint32((base + 20) * 4, pressed, true);
    return true;
  }

  function distance(m, a, b) {
    return Math.hypot(m[a * 16 + 12] - m[b * 16 + 12], m[a * 16 + 13] - m[b * 16 + 13], m[a * 16 + 14] - m[b * 16 + 14]);
  }

  function readHand(frame, hand, k) {
    let spaces = jointSpaces.get(hand);
    if (spaces === undefined) {
      spaces = JOINTS.map((name) => hand.get(name));
      jointSpaces.set(hand, spaces.every(Boolean) ? spaces : null);
    }
    if (!spaces) return false;
    const out = handMsg.joints[k];
    if (frame.fillPoses) {
      if (!frame.fillPoses(spaces, refSpace, out)) return false;
    } else {
      for (let j = 0; j < 25; j++) {
        const pose = frame.getJointPose(spaces[j], refSpace);
        if (!pose) return false;
        out.set(pose.transform.matrix, j * 16);
      }
    }
    // vuer: pinch = thumb tip to index tip, squeeze = middle tip to middle proximal (the PC applies the thresholds).
    // The two distances follow the 400 joint floats in the message.
    const base = BLOCKS_AT + k * HAND_FLOATS;
    handMsg.f32[base + 400] = distance(out, THUMB_TIP, INDEX_TIP);
    handMsg.f32[base + 401] = distance(out, MIDDLE_TIP, MIDDLE_PROXIMAL);
    return true;
  }

  function sendPose(frame, haveHead) {
    if (!poseWs || poseWs.readyState !== WebSocket.OPEN) return;
    const hand = config.hand, msg = hand ? handMsg : ctrlMsg;
    if (poseWs.bufferedAmount > 4 * msg.buf.byteLength) {
      xrStat.skipped++;  // the link is behind: skip, queued poses would only arrive stale
      return;
    }
    let flags = 0;
    if (haveHead) {
      msg.f32.set(head, HEAD_AT);
      flags |= 1;
    }
    // first matching input source per side, like the vuer client
    let left = null, right = null;
    for (const src of frame.session.inputSources) {
      if (hand ? !src.hand : !(src.gamepad && src.gamepad.buttons.length >= 6)) continue;
      if (src.handedness === 'left') left = left || src;
      else if (src.handedness === 'right') right = right || src;
    }
    if (hand) {
      if (left && readHand(frame, left.hand, 0)) flags |= 2;
      if (right && readHand(frame, right.hand, 1)) flags |= 4;
    } else {
      if (left && readController(frame, left, BLOCKS_AT)) flags |= 2;
      if (right && readController(frame, right, BLOCKS_AT + CTRL_FLOATS)) flags |= 4;
    }
    msg.dv.setFloat64(0, performance.timeOrigin + performance.now(), true);
    msg.dv.setUint32(8, frameSeq, true);
    msg.dv.setUint8(13, flags);
    poseWs.send(msg.buf);
  }

  // ------------------------------------------------------------------ XR frame loop
  const xrStat = {t0: 0, last: 0, n: 0, busy: 0, maxBusy: 0, maxGap: 0, skipped: 0, errors: 0};
  const videoStat = {shown: 0, decode: 0, upload: 0, maxUpload: 0, received: 0, dropped: 0, received0: 0, dropped0: 0};

  function onXRFrame(time, frame) {
    const s = frame.session;
    s.requestAnimationFrame(onXRFrame);
    const t0 = performance.now();
    if (xrStat.last) xrStat.maxGap = Math.max(xrStat.maxGap, t0 - xrStat.last);
    xrStat.last = t0;
    try {
      const viewerPose = frame.getViewerPose(refSpace);
      if (viewerPose) unionCamera(viewerPose.views);
      sendPose(frame, !!viewerPose);  // first thing in the frame: the pose leaves before any drawing

      const layer = s.renderState.baseLayer;
      gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
      gl.clearColor(0, 0, 0, arSession ? 0 : 1);  // pass-through shows wherever alpha is 0
      gl.clear(gl.COLOR_BUFFER_BIT);
      if (viewerPose && frameSeq && config.video) {
        planeModel(config.height * config.aspect, config.height, config.distance);
        for (const view of viewerPose.views) {
          const vp = layer.getViewport(view);
          gl.viewport(vp.x, vp.y, vp.width, vp.height);
          invertRigid(viewInv, view.transform.matrix);
          multiply(mv, viewInv, model);
          multiply(mvp, view.projectionMatrix, mv);
          drawQuad(eyes === 2 && view.eye === 'right' ? 1 : 0, mvp);
        }
      }
    } catch (err) {
      if (xrStat.errors++ < 3) report(`XR frame: ${err.message}`);
    }
    const busy = performance.now() - t0;
    xrStat.n++;
    xrStat.busy += busy;
    xrStat.maxBusy = Math.max(xrStat.maxBusy, busy);
    if (config.stats > 0 && t0 - xrStat.t0 >= config.stats * 1000) sendStats(t0);
  }

  function resetStats(now) {
    Object.assign(xrStat, {t0: now, n: 0, busy: 0, maxBusy: 0, maxGap: 0, skipped: 0});
    Object.assign(videoStat, {shown: 0, decode: 0, upload: 0, maxUpload: 0, received0: videoStat.received, dropped0: videoStat.dropped});
  }

  function sendStats(now) {
    const dt = (now - xrStat.t0) / 1000, shown = Math.max(videoStat.shown, 1);
    if (poseWs && poseWs.readyState === WebSocket.OPEN) {
      poseWs.send(JSON.stringify({
        type: 'stats',
        xr_fps: xrStat.n / dt, xr_gap_ms: xrStat.maxGap, target_fps: session && session.frameRate || 0,
        frame_ms: xrStat.busy / Math.max(xrStat.n, 1), frame_max_ms: xrStat.maxBusy, pose_skipped: xrStat.skipped,
        video_recv_fps: (videoStat.received - videoStat.received0) / dt, video_shown_fps: videoStat.shown / dt,
        video_dropped: videoStat.dropped - videoStat.dropped0,
        decode_ms: videoStat.decode / shown, upload_ms: videoStat.upload / shown, upload_max_ms: videoStat.maxUpload,
      }));
    }
    resetStats(now);
  }

  async function enterXR() {
    if (session || !config) return;
    const ar = config.mode !== 'immersive';
    const s = await navigator.xr.requestSession(ar ? 'immersive-ar' : 'immersive-vr', {
      requiredFeatures: ['local-floor'],
      optionalFeatures: ['hand-tracking'],
    });
    try {
      await gl.makeXRCompatible();
      const layer = new XRWebGLLayer(s, gl, {antialias: false, depth: false, stencil: false, alpha: ar});
      if ('fixedFoveation' in layer) layer.fixedFoveation = 0;  // a single textured quad: keep the whole image sharp
      s.updateRenderState({baseLayer: layer});
      refSpace = await s.requestReferenceSpace('local-floor');
      const rates = s.supportedFrameRates;
      if (rates && rates.length && s.updateTargetFrameRate) {
        // poses are sampled once per XR frame, so the display rate is the control rate: ask for the highest
        s.updateTargetFrameRate(Math.max(...rates)).catch(() => {});
      }
    } catch (err) {
      s.end().catch(() => {});
      throw err;
    }
    s.addEventListener('end', () => {
      session = null;
      refSpace = null;
      updateUI();
      schedulePreview();
    });
    arSession = ar;
    session = s;
    xrStat.last = 0;
    resetStats(performance.now());
    s.requestAnimationFrame(onXRFrame);
  }
  button.addEventListener('click', () => enterXR().catch((err) => report(`cannot start XR: ${err.message}`)));

  // ------------------------------------------------------------------ 2D page: preview of the left eye
  let previewQueued = false;
  function schedulePreview() {
    if (!session && !previewQueued) {
      previewQueued = true;
      requestAnimationFrame(drawPreview);
    }
  }

  function drawPreview() {
    previewQueued = false;
    if (session) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(canvas.clientWidth * dpr)), h = Math.max(1, Math.round(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, w, h);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    if (!frameSeq || !config || !config.video) return;
    const image = textures[0].width / textures[0].height, view = w / h;
    mvp.fill(0);
    mvp[0] = image > view ? 2 : 2 * image / view;
    mvp[5] = image > view ? 2 * view / image : 2;
    mvp[10] = mvp[15] = 1;
    drawQuad(0, mvp);
  }
  window.addEventListener('resize', schedulePreview);

  // ------------------------------------------------------------------ video worker
  function onWorkerMessage(e) {
    const m = e.data;
    if (m.type === 'frame') {
      const t0 = performance.now();
      for (let i = 0; i < m.bitmaps.length; i++) {
        upload(i, m.bitmaps[i]);
        m.bitmaps[i].close();
      }
      eyes = m.bitmaps.length;
      frameSeq = m.seq;
      const dt = performance.now() - t0;
      videoStat.shown++;
      videoStat.decode += m.decodeMs;
      videoStat.upload += dt;
      videoStat.maxUpload = Math.max(videoStat.maxUpload, dt);
      videoStat.received = m.received;
      videoStat.dropped = m.dropped;
      pageFrames++;
      schedulePreview();
    } else if (m.type === 'link') {
      videoOpen = m.open;
      updateUI();
    } else if (m.type === 'error') {
      report(`video: ${m.message}`);
    }
  }

  function setVideo(on) {
    if (on && !worker) {
      worker = new Worker('video_worker.js');
      worker.onmessage = onWorkerMessage;
      worker.onerror = (e) => report(`video worker: ${e.message}`);
      worker.postMessage({type: 'start', url: WS_BASE + '/video'});
    } else if (!on && worker) {
      worker.terminate();
      worker = null;
      videoOpen = false;
      frameSeq = 0;
      schedulePreview();
    }
  }

  // ------------------------------------------------------------------ link to the PC
  function applyConfig(next) {
    if (next.version !== window.TV_VERSION && sessionStorage.getItem('tvReloadedFor') !== next.version) {
      // the PC now serves a different client than this page: load it
      sessionStorage.setItem('tvReloadedFor', next.version);
      location.reload();
      return;
    }
    const modeChanged = config && (config.mode === 'immersive') !== (next.mode === 'immersive');
    config = next;
    if (session && modeChanged) session.end().catch(() => {});  // VR <-> pass-through needs a new session
    setVideo(next.video);
    updateUI();
  }

  function connectPose() {
    const ws = new WebSocket(WS_BASE + '/pose');
    ws.binaryType = 'arraybuffer';
    ws.onopen = () => {
      poseWs = ws;
      updateUI();
    };
    ws.onmessage = (e) => {
      if (typeof e.data !== 'string') return;
      const m = JSON.parse(e.data);
      if (m.type === 'config') applyConfig(m);
    };
    ws.onclose = () => {
      if (poseWs === ws) poseWs = null;
      updateUI();
      setTimeout(connectPose, 1000);
    };
  }

  // ------------------------------------------------------------------ page UI (not visible while in XR)
  let pageFrames = 0, pageFps = 0;
  function updateUI() {
    if (session) return;
    const connected = !!(poseWs && config);
    const xrMode = config && config.mode !== 'immersive' ? 'immersive-ar' : 'immersive-vr';
    button.disabled = !(connected && supported[xrMode]);
    button.textContent = !connected ? 'Connecting…'
      : !supported[xrMode] ? 'XR not available'
      : xrMode === 'immersive-vr' ? 'Enter VR' : 'Enter pass-through';
    const lines = [connected ? `PC connected · ${config.mode} · ${config.hand ? 'hand tracking' : 'controllers'}` : 'Waiting for the PC…'];
    if (connected && config.video) {
      lines.push(frameSeq && videoOpen
        ? `video ${pageFps.toFixed(0)} fps · ${textures[0].width}×${textures[0].height}${eyes === 2 ? ' ×2' : ''}`
        : 'video: waiting for images');
    }
    if (lastError) lines.push(lastError);
    statusEl.textContent = lines.join('\n');
  }
  setInterval(() => {
    pageFps = pageFrames;
    pageFrames = 0;
    updateUI();
  }, 1000);

  if (navigator.xr) {
    for (const mode of Object.keys(supported)) {
      navigator.xr.isSessionSupported(mode).then((ok) => {
        supported[mode] = ok;
        updateUI();
      }).catch(() => {});
    }
  }
  connectPose();
  updateUI();
})();
