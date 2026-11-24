// Receives the camera frames and decodes them off the main thread, so the XR frame loop
// (which samples and sends the poses) never waits for a JPEG.
// One message per frame: u32 seq, u8 n, 3 pad, u32[n] sizes, n JPEGs (see lean.py).
'use strict';

let url = null, ws = null, retry = null;
let decoding = false, waiting = null;
let received = 0, dropped = 0;

onmessage = (e) => {
  if (e.data.type === 'start') {
    url = e.data.url;
    connect();
  } else if (e.data.type === 'stop') {
    url = null;
    clearTimeout(retry);
    if (ws) ws.close();
  }
};

function connect() {
  if (!url) return;
  const sock = ws = new WebSocket(url);
  sock.binaryType = 'arraybuffer';
  sock.onopen = () => postMessage({type: 'link', open: true});
  sock.onmessage = (e) => {
    if (typeof e.data === 'string') return;
    received++;
    if (decoding) {
      // latest frame wins: never build a queue, a queue is latency
      if (waiting) dropped++;
      waiting = e.data;
    } else {
      decode(e.data);
    }
  };
  sock.onclose = () => {
    if (ws !== sock) return;
    postMessage({type: 'link', open: false});
    if (url) retry = setTimeout(connect, 500);
  };
}

async function decode(buf) {
  decoding = true;
  const t0 = performance.now();
  try {
    const dv = new DataView(buf);
    const seq = dv.getUint32(0, true), n = dv.getUint8(4);
    let offset = 8 + 4 * n;
    const jobs = [];
    for (let i = 0; i < n; i++) {
      const size = dv.getUint32(8 + 4 * i, true);
      jobs.push(createImageBitmap(new Blob([new Uint8Array(buf, offset, size)], {type: 'image/jpeg'})));
      offset += size;
    }
    // both eyes are handed over together, so the pair on screen is always from the same camera frame
    const bitmaps = await Promise.all(jobs);
    postMessage({type: 'frame', seq, bitmaps, decodeMs: performance.now() - t0, received, dropped}, bitmaps);
  } catch (err) {
    postMessage({type: 'error', message: String(err)});
  }
  decoding = false;
  if (waiting) {
    const next = waiting;
    waiting = null;
    decode(next);
  }
}
