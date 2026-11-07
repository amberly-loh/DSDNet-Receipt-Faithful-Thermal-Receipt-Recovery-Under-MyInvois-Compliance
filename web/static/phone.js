// Phone scanner page: sends receipt photos to the paired computer.
(function () {
  const $ = id => document.getElementById(id);
  // The pairing code arrives in the address after "#", so it is never sent in
  // page requests or saved in server logs. Keep it for this tab only.
  if (location.hash.length > 1) {
    sessionStorage.setItem('phoneToken', location.hash.slice(1));
    history.replaceState(null, '', location.pathname);
  }
  const token = sessionStorage.getItem('phoneToken') || '';
  const queue = [];
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

  function addPhotos(files) {
    [...files].filter(file => file.type.startsWith('image/')).forEach(file => {
      const item = document.createElement('li');
      const preview = document.createElement('img'); preview.alt = '';
      preview.src = URL.createObjectURL(file);
      const text = document.createElement('div');
      text.innerHTML = '<b>Receipt photo</b><small>Waiting to send…</small>';
      item.append(preview, text); $('list').prepend(item); $('listTitle').hidden = false;
      queue.push({file, item});
    });
    sendNext();
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
    const form = new FormData(); form.append('receipt', entry.file, entry.file.name || `receipt-${Date.now()}.jpg`);
    const request = new XMLHttpRequest();
    request.open('POST', '/api/phone/upload');
    request.setRequestHeader('X-Phone-Token', token);
    request.upload.onprogress = event => { if (event.lengthComputable) status(entry, 'Receipt photo', `Sending… ${Math.round(100 * event.loaded / event.total)}%`); };
    request.upload.onload = () => status(entry, 'Receipt photo', 'Sent. Reading it on the computer…');
    request.onload = () => {
      let data = {}; try { data = JSON.parse(request.responseText); } catch (_) { /* keep empty */ }
      if (request.status >= 200 && request.status < 300) {
        const r = data.receipt || {};
        const parts = [r.total_amount ? `RM ${r.total_amount}` : null, r.transaction_date].filter(Boolean).join(' · ');
        status(entry, r.merchant || 'Receipt saved', `Saved on the computer${parts ? ' · ' + parts : ''}`, 'done');
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

  $('cameraInput').onchange = event => { if (connected) addPhotos(event.target.files); event.target.value = ''; };
  $('galleryInput').onchange = event => { if (connected) addPhotos(event.target.files); event.target.value = ''; };
  check();
})();
