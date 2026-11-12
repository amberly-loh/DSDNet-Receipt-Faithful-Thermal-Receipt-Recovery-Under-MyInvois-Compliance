// Phone scanner page: sends receipt photos to the paired computer.
// Each photo is checked first: the computer suggests the receipt's corners,
// the person can turn the photo and move the corners, and only then is it sent.
(function () {
  const $ = id => document.getElementById(id);
  // The pairing code arrives in the address after "#", so it is never sent in
  // page requests or saved in server logs. Keep it for this tab only.
  if (location.hash.length > 1) {
    sessionStorage.setItem('phoneToken', location.hash.slice(1));
    history.replaceState(null, '', location.pathname);
  }
  const token = sessionStorage.getItem('phoneToken') || '';
  const queue = [];        // confirmed photos waiting to be sent
  const pending = [];      // photos waiting to be checked
  let sending = false, connected = false;

  function setState(text, bad) {
    $('state').textContent = text;
    $('state').classList.toggle('bad', Boolean(bad));
  }
  function enable(on) {
    connected = on;
    $('cameraLabel').classList.toggle('disabled', !on);
    $('galleryLabel').classList.toggle('disabled', !on);
  }

  async function check() {
    if (!token) { setState('Open this page by scanning the QR code shown on the computer (Scan with phone).', true); return; }
    try {
      const response = await fetch('/api/phone/check', {headers: {'X-Phone-Token': token}, cache: 'no-store'});
      const data = await response.json().catch(() => ({}));
      if (!response.ok) { enable(false); setState(data.error || 'This pairing has ended. Scan the QR code on the computer again.', true); return; }
      enable(true); setState(`Connected to ${data.name ? data.name + "'s" : 'the'} receipts on the computer.`);
    } catch (_) {
      enable(false); setState('Cannot reach the computer. Check the phone is on the same Wi-Fi and the app is running.', true);
    }
  }

  // ------------------------------------------------------------ check & adjust
  const FULL = [[0, 0], [1, 0], [1, 1], [0, 1]];
  const canvas = $('cropCanvas'), ctx = canvas.getContext('2d');
  const adjust = {file: null, img: null, url: '', rot: 0, corners: FULL.map(p => p.slice()), auto: null, fromCamera: false, drag: -1, box: null,
                  turned: false, moved: false};   // what the person changed themselves

  const turnRight = ([u, v]) => [1 - v, u];   // where a point goes when the view turns 90° clockwise
  const turnLeft = ([u, v]) => [v, 1 - u];
  function setRotation(target) {
    while (adjust.rot !== target) { adjust.corners = adjust.corners.map(turnRight); adjust.rot = (adjust.rot + 90) % 360; }
  }

  function loadImage(file) {
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => resolve({img, url});
      img.onerror = () => { URL.revokeObjectURL(url); reject(new Error('unreadable')); };
      img.src = url;   // browsers turn the photo upright from its camera tag
    });
  }

  function previewBlob(img) {
    const scale = Math.min(1, 1280 / Math.max(img.naturalWidth, img.naturalHeight));
    const c = document.createElement('canvas');
    c.width = Math.round(img.naturalWidth * scale); c.height = Math.round(img.naturalHeight * scale);
    c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
    return new Promise(resolve => c.toBlob(resolve, 'image/jpeg', 0.85));
  }

  async function detect() {
    const form = new FormData(); form.append('preview', await previewBlob(adjust.img), 'preview.jpg');
    const response = await fetch('/api/phone/detect', {method: 'POST', headers: {'X-Phone-Token': token}, body: form});
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || 'detect failed');
    return data;
  }

  function hint(text, bad) { $('cropHint').textContent = text; $('cropHint').classList.toggle('bad', Boolean(bad)); }

  function applyAuto() {
    adjust.rot = 0; adjust.turned = false; adjust.moved = false;
    if (adjust.auto && adjust.auto.found) {
      adjust.corners = adjust.auto.corners.map(p => p.slice());
      setRotation(adjust.auto.suggested_rotation || 0);
      hint('Receipt found. Drag a corner if the outline is off, and turn the photo until the print reads upright.');
    } else {
      adjust.corners = FULL.map(p => p.slice());
      hint('Could not find the receipt edges. Drag the four corners onto the receipt, or retake it flat on a darker surface.', true);
    }
    draw();
  }

  async function openNext() {
    if (adjust.file || !pending.length) return;
    const {file, fromCamera, replaces} = pending.shift();
    let loaded;
    try { loaded = await loadImage(file); }
    catch (_) { addFailed('This photo could not be opened on the phone.'); return openNext(); }
    Object.assign(adjust, {file, fromCamera, replaces: replaces || null, img: loaded.img, url: loaded.url, rot: 0, corners: FULL.map(p => p.slice()), auto: null, drag: -1, turned: false, moved: false});
    $('cropRetake').textContent = fromCamera ? 'Retake' : 'Skip';
    $('cropCount').textContent = pending.length ? `${pending.length} more after this` : '';
    $('adjust').hidden = false; document.body.classList.add('adjusting');
    hint('Finding the receipt…');
    requestAnimationFrame(draw);
    try { adjust.auto = await detect(); } catch (_) { adjust.auto = {found: false}; }
    if (adjust.file === file) applyAuto();
  }

  function closeAdjust() {
    adjust.file = null; adjust.img = null;
    $('adjust').hidden = true; document.body.classList.remove('adjusting');
  }

  // the photo, turned by adjust.rot, fitted into the canvas
  function layout() {
    const ratio = window.devicePixelRatio || 1;
    const rect = $('cropStage').getBoundingClientRect();
    canvas.width = Math.round(rect.width * ratio); canvas.height = Math.round(rect.height * ratio);
    canvas.style.width = rect.width + 'px'; canvas.style.height = rect.height + 'px';
    const w = adjust.img.naturalWidth, h = adjust.img.naturalHeight, sideways = adjust.rot % 180 !== 0;
    const viewW = sideways ? h : w, viewH = sideways ? w : h, pad = 22 * ratio;
    const scale = Math.min((canvas.width - 2 * pad) / viewW, (canvas.height - 2 * pad) / viewH);
    const dw = viewW * scale, dh = viewH * scale;
    adjust.box = {x: (canvas.width - dw) / 2, y: (canvas.height - dh) / 2, w: dw, h: dh, scale, ratio};
  }
  const toCanvas = ([u, v]) => [adjust.box.x + u * adjust.box.w, adjust.box.y + v * adjust.box.h];

  function draw() {
    if (!adjust.img) return;
    layout();
    const {box} = adjust, img = adjust.img;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.translate(box.x + box.w / 2, box.y + box.h / 2);
    ctx.rotate(adjust.rot * Math.PI / 180);
    const iw = img.naturalWidth * box.scale, ih = img.naturalHeight * box.scale;
    ctx.drawImage(img, -iw / 2, -ih / 2, iw, ih);
    ctx.restore();
    const points = adjust.corners.map(toCanvas);
    // shade everything outside the outline
    ctx.save(); ctx.beginPath();
    ctx.rect(box.x, box.y, box.w, box.h);
    ctx.moveTo(points[0][0], points[0][1]); points.slice(1).forEach(p => ctx.lineTo(p[0], p[1])); ctx.closePath();
    ctx.fillStyle = 'rgba(8, 25, 21, .55)'; ctx.fill('evenodd'); ctx.restore();
    ctx.save(); ctx.lineWidth = 2.5 * box.ratio; ctx.strokeStyle = '#3ee0a8';
    ctx.beginPath(); ctx.moveTo(points[0][0], points[0][1]); points.slice(1).forEach(p => ctx.lineTo(p[0], p[1])); ctx.closePath(); ctx.stroke();
    points.forEach((p, i) => {
      ctx.beginPath(); ctx.arc(p[0], p[1], (i === adjust.drag ? 15 : 12) * box.ratio, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255,255,255,.95)'; ctx.fill(); ctx.lineWidth = 3 * box.ratio; ctx.strokeStyle = '#1b6b5a'; ctx.stroke();
    });
    ctx.restore();
  }

  function pointerPosition(event) {
    const rect = canvas.getBoundingClientRect(), ratio = adjust.box.ratio;
    return [(event.clientX - rect.left) * ratio, (event.clientY - rect.top) * ratio];
  }
  canvas.addEventListener('pointerdown', event => {
    if (!adjust.img || !adjust.box) return;
    const [x, y] = pointerPosition(event);
    let best = -1, bestDistance = 40 * adjust.box.ratio;
    adjust.corners.map(toCanvas).forEach((p, i) => { const d = Math.hypot(p[0] - x, p[1] - y); if (d < bestDistance) { best = i; bestDistance = d; } });
    if (best < 0) return;
    adjust.drag = best; adjust.moved = true; canvas.setPointerCapture(event.pointerId); event.preventDefault(); draw();
  });
  canvas.addEventListener('pointermove', event => {
    if (adjust.drag < 0) return;
    const [x, y] = pointerPosition(event), box = adjust.box;
    adjust.corners[adjust.drag] = [Math.min(1, Math.max(0, (x - box.x) / box.w)), Math.min(1, Math.max(0, (y - box.y) / box.h))];
    event.preventDefault(); draw();
  });
  const endDrag = () => { if (adjust.drag >= 0) { adjust.drag = -1; draw(); } };
  canvas.addEventListener('pointerup', endDrag); canvas.addEventListener('pointercancel', endDrag);
  window.addEventListener('resize', () => requestAnimationFrame(draw));

  $('cropRotateLeft').onclick = () => { adjust.turned = true; adjust.corners = adjust.corners.map(turnLeft); adjust.rot = (adjust.rot + 270) % 360; draw(); };
  $('cropRotateRight').onclick = () => { adjust.turned = true; adjust.corners = adjust.corners.map(turnRight); adjust.rot = (adjust.rot + 90) % 360; draw(); };
  $('cropAuto').onclick = () => applyAuto();
  $('cropWhole').onclick = () => { adjust.moved = true; adjust.corners = FULL.map(p => p.slice()); hint('Using the whole photo. The receipt will be read as shown.'); draw(); };
  $('cropRetake').onclick = () => {
    const again = adjust.fromCamera;
    URL.revokeObjectURL(adjust.url); closeAdjust();
    if (again && !pending.length) $('cameraInput').click(); else openNext();
  };
  $('cropSend').onclick = () => {
    if (!adjust.img) return;
    // Send a cut only when there is one: found automatically or set by hand. With
    // neither, the computer makes its own cut. An untouched turn is only a
    // suggestion, so the computer still checks the print reads upright.
    const hasCut = adjust.moved || Boolean(adjust.auto && adjust.auto.found);
    const entry = {file: adjust.file, rotation: adjust.rot, rotationConfirmed: adjust.turned, replaces: adjust.replaces,
                   corners: hasCut ? adjust.corners.map(([u, v]) => [+u.toFixed(5), +v.toFixed(5)]) : null};
    entry.item = listItem(thumbnail(), 'Receipt photo', 'Waiting to send…');
    URL.revokeObjectURL(adjust.url); closeAdjust();
    queue.push(entry); sendNext(); openNext();
  };

  // a small picture of what was confirmed: the outlined area, turned
  function thumbnail() {
    const {img, rot, corners} = adjust, sideways = rot % 180 !== 0;
    const viewW = sideways ? img.naturalHeight : img.naturalWidth, viewH = sideways ? img.naturalWidth : img.naturalHeight;
    const xs = corners.map(p => p[0] * viewW), ys = corners.map(p => p[1] * viewH);
    const left = Math.min(...xs), top = Math.min(...ys);
    const width = Math.max(8, Math.max(...xs) - left), height = Math.max(8, Math.max(...ys) - top);
    const scale = Math.min(104 / width, 136 / height);
    const c = document.createElement('canvas'); c.width = Math.max(1, Math.round(width * scale)); c.height = Math.max(1, Math.round(height * scale));
    const g = c.getContext('2d');
    g.scale(scale, scale); g.translate(-left, -top);
    g.translate(viewW / 2, viewH / 2); g.rotate(rot * Math.PI / 180);
    g.drawImage(img, -img.naturalWidth / 2, -img.naturalHeight / 2);
    return c.toDataURL('image/jpeg', 0.8);
  }

  // ------------------------------------------------------------ list & sending
  function listItem(src, title, detail) {
    const item = document.createElement('li');
    const preview = document.createElement('img'); preview.alt = ''; if (src) preview.src = src;
    const text = document.createElement('div'); text.innerHTML = '<b></b><small></small>';
    text.querySelector('b').textContent = title; text.querySelector('small').textContent = detail;
    item.append(preview, text); $('list').prepend(item); $('listTitle').hidden = false;
    return item;
  }
  function addFailed(message) { listItem('', 'Not sent', message).className = 'failed'; }

  let retakeReplaces = null;   // a camera retake that replaces a scan the computer asked to redo
  function addPhotos(files, fromCamera) {
    const replaces = fromCamera ? retakeReplaces : null; retakeReplaces = null;
    [...files].filter(file => file.type.startsWith('image/') || !file.type).forEach(file => pending.push({file, fromCamera, replaces}));
    openNext();
  }

  // After a scan that looks turned or cut wrongly: fix the same photo, or take a new one.
  // The new scan replaces the bad one on the computer, so nothing is kept twice.
  function offerRetry(entry, publicId) {
    const row = document.createElement('div'); row.className = 'retry-row';
    const fix = document.createElement('button'); fix.type = 'button'; fix.className = 'button'; fix.textContent = 'Turn or crop again';
    const retake = document.createElement('button'); retake.type = 'button'; retake.className = 'button secondary'; retake.textContent = 'Retake photo';
    const done = () => row.remove();
    fix.onclick = () => { done(); entry.item.remove(); pending.unshift({file: entry.file, fromCamera: false, replaces: publicId}); openNext(); };
    retake.onclick = () => { done(); entry.item.remove(); retakeReplaces = publicId; $('cameraInput').click(); };
    row.append(fix, retake); entry.item.querySelector('div').append(row);
  }

  function status(entry, title, detail, kind) {
    entry.item.className = kind || '';
    entry.item.querySelector('b').textContent = title;
    entry.item.querySelector('small').textContent = detail;
  }

  function sendNext() {
    if (sending || !queue.length) return;
    sending = true;
    const entry = queue.shift();
    const form = new FormData();
    form.append('receipt', entry.file, entry.file.name || `receipt-${Date.now()}.jpg`);
    form.append('rotation', String(entry.rotation));
    form.append('rotation_confirmed', entry.rotationConfirmed ? '1' : '0');
    if (entry.corners) form.append('corners', JSON.stringify(entry.corners));
    if (entry.replaces) form.append('replaces', entry.replaces);
    const request = new XMLHttpRequest();
    request.open('POST', '/api/phone/upload');
    request.setRequestHeader('X-Phone-Token', token);
    request.upload.onprogress = event => { if (event.lengthComputable) status(entry, 'Receipt photo', `Sending… ${Math.round(100 * event.loaded / event.total)}%`); };
    request.upload.onload = () => status(entry, 'Receipt photo', 'Sent. Reading it on the computer…');
    request.onload = () => {
      let data = {}; try { data = JSON.parse(request.responseText); } catch (_) { /* keep empty */ }
      if (request.status >= 200 && request.status < 300) {
        const r = data.receipt || {};
        const check = data.capture_check || {};
        if (check.status === 'retry') {
          status(entry, 'Please try this one again', check.message || 'The photo may be turned or cut wrongly.', 'warn');
          offerRetry(entry, r.public_id);
          sending = false; sendNext(); return;
        }
        const missing = [!r.total_amount && 'total', !r.transaction_date && 'date'].filter(Boolean);
        if (missing.length) {
          status(entry, r.merchant || 'Saved, needs checking',
                 `Saved, but the ${missing.join(' and ')} could not be read. Retake it flat and upright, or fix it on the computer.`, 'warn');
        } else {
          status(entry, r.merchant || 'Receipt saved', `Saved on the computer · RM ${r.total_amount} · ${r.transaction_date}`, 'done');
        }
      } else {
        status(entry, 'Not saved', data.error || 'The computer could not read this photo. Try again.', 'failed');
        if (request.status === 401) { enable(false); setState(data.error, true); }
      }
      sending = false; sendNext();
    };
    request.onerror = () => {
      status(entry, 'Not sent', 'Lost connection to the computer. Check the Wi-Fi and try again.', 'failed');
      sending = false; sendNext();
    };
    status(entry, 'Receipt photo', 'Sending…');
    request.send(form);
  }

  $('cameraInput').onchange = event => { if (connected) addPhotos(event.target.files, true); event.target.value = ''; };
  $('galleryInput').onchange = event => { if (connected) addPhotos(event.target.files, false); event.target.value = ''; };
  check();
})();
