const $ = id => document.getElementById(id);
const pages = ['overview', 'scan', 'history', 'detail', 'admin'];
let csrf = '', user = null, authMode = 'login', selectedFile = null, cameraStream = null;
let currentReceipt = null, historyMode = 'grid', searchTimer = null;

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
}

async function api(path, options = {}) {
  const config = {...options, headers: {...(options.headers || {})}};
  if (['POST', 'PATCH', 'DELETE'].includes((config.method || 'GET').toUpperCase())) config.headers['X-CSRF-Token'] = csrf;
  if (config.body && !(config.body instanceof FormData)) {
    config.headers['Content-Type'] = 'application/json';
    if (typeof config.body !== 'string') config.body = JSON.stringify(config.body);
  }
  const response = await fetch(path, config);
  const type = response.headers.get('content-type') || '';
  const data = type.includes('application/json') ? await response.json() : null;
  if (data?.csrf_token) csrf = data.csrf_token;
  if (!response.ok) {
    if (response.status === 401 && !path.includes('/login')) showAuth();
    const error = new Error(data?.error || 'Something went wrong. Please try again.'); error.details = data; throw error;
  }
  return data;
}

function showAuth() {
  user = null;
  $('workspace').classList.add('hidden');
  $('authView').classList.remove('hidden');
}

function showWorkspace() {
  $('authView').classList.add('hidden');
  $('workspace').classList.remove('hidden');
  $('profileName').textContent = user.name;
  $('profileEmail').textContent = user.email;
  $('avatar').textContent = user.name.trim().charAt(0).toUpperCase();
  $('welcomeName').textContent = `Ready to recover a receipt, ${user.name.split(' ')[0]}?`;
  $('adminNav').classList.toggle('hidden', user.role !== 'admin');
  const requested = location.hash.slice(1) || 'overview';
  const lastReceipt = sessionStorage.getItem('lastReceiptId');
  if (requested === 'detail' && lastReceipt) openReceipt(lastReceipt); else route(requested, false);
}

function setAuthMode(mode) {
  authMode = mode;
  const register = mode === 'register';
  $('nameLabel').classList.toggle('hidden', !register);
  $('passwordHint').classList.toggle('hidden', !register);
  $('policyLabel').classList.toggle('hidden', !register);
  $('confirmationLabel').classList.toggle('hidden', !register);
  $('authTitle').textContent = register ? 'Create your workspace' : 'Welcome back';
  $('authSubtitle').textContent = register ? 'Save, organise and review your recovered receipts.' : 'Sign in to access your saved receipts.';
  $('authSubmit').textContent = register ? 'Create account' : 'Sign in';
  $('authSwitchText').textContent = register ? 'Already have an account?' : 'New here?';
  $('authSwitch').textContent = register ? 'Sign in' : 'Create an account';
  $('authPassword').autocomplete = register ? 'new-password' : 'current-password';
  $('authError').classList.add('hidden');
  clearFieldErrors();
}

function clearFieldErrors() {
  ['name','email','password','confirmation','policy'].forEach(name => { const node = $(`${name}Error`); if (node) node.textContent = ''; });
  ['authName','authEmail','authPassword','authConfirmation'].forEach(id => $(id).removeAttribute('aria-invalid'));
}

function showFieldErrors(fields = {}) {
  clearFieldErrors();
  Object.entries(fields).forEach(([name, message]) => {
    const node = $(`${name}Error`); if (node) node.textContent = message;
    const input = $(`auth${name.charAt(0).toUpperCase()}${name.slice(1)}`); if (input) input.setAttribute('aria-invalid', 'true');
  });
}

$('authSwitch').onclick = () => setAuthMode(authMode === 'login' ? 'register' : 'login');
$('authForm').onsubmit = async event => {
  event.preventDefault();
  clearFieldErrors();
  const button = $('authSubmit');
  button.disabled = true; button.textContent = authMode === 'login' ? 'Signing in…' : 'Creating account…';
  try {
    const payload = {email: $('authEmail').value, password: $('authPassword').value};
    if (authMode === 'register') { payload.name = $('authName').value; payload.password_confirmation = $('authConfirmation').value; payload.policy_accepted = $('policyAccepted').checked; }
    const data = await api(`/api/${authMode === 'register' ? 'register' : 'login'}`, {method: 'POST', body: payload});
    user = data.user; showWorkspace();
  } catch (error) {
    showFieldErrors(error.details?.fields);
    $('authError').textContent = error.message; $('authError').classList.remove('hidden');
  } finally {
    button.disabled = false; setAuthMode(authMode);
  }
};

$('logoutButton').onclick = async () => {
  try { await api('/api/logout', {method: 'POST'}); } finally {
    csrf = ''; const state = await api('/api/auth'); csrf = state.csrf_token; showAuth();
  }
};

const pageCopy = {
  overview: ['Overview', 'Your receipt recovery workspace'],
  scan: ['Scan receipt', 'Recover a faded thermal receipt'],
  history: ['Receipt history', 'Search, review and organise saved receipts'],
  detail: ['Receipt details', 'Compare the image and verify extracted fields']
  ,admin: ['Administration', 'Manage authorised users and audit activity']
};

function route(name, updateHash = true) {
  if (!pages.includes(name)) name = 'overview';
  pages.forEach(p => $(`${p}Page`).classList.toggle('hidden', p !== name));
  document.querySelectorAll('[data-route]').forEach(el => el.classList.toggle('active', el.dataset.route === name));
  [$('pageTitle').textContent, $('pageSubtitle').textContent] = pageCopy[name];
  if (updateHash) history.replaceState(null, '', `#${name}`);
  document.body.classList.remove('nav-open');
  window.scrollTo({top: 0, behavior: 'smooth'});
  if (name === 'overview') loadDashboard();
  if (name === 'history') loadHistory();
  if (name === 'scan') resetScan();
  if (name === 'admin') { if (user.role !== 'admin') return route('overview'); loadAdmin(); }
}

document.querySelectorAll('[data-route]').forEach(el => el.onclick = event => { event.preventDefault(); route(el.dataset.route); });
$('menuButton').onclick = () => document.body.classList.toggle('nav-open');
$('backButton').onclick = () => route('history');

function formatDate(value) {
  if (!value) return 'Date unavailable';
  const date = new Date(value); return Number.isNaN(date.valueOf()) ? value : date.toLocaleDateString('en-MY', {day: 'numeric', month: 'short', year: 'numeric'});
}
function statusLabel(status) { return ({needs_review: 'Needs review', reviewed: 'Reviewed', archived: 'Archived'})[status] || status; }
function amountLabel(value) { if (!value) return 'Amount unavailable'; return /^(RM|MYR)/i.test(value) ? value : `RM ${value}`; }

function emptyState(kind) {
  const history = kind === 'history';
  return `<div class="empty-state"><span>${history ? '⌕' : '▧'}</span><h3>${history ? 'No matching receipts' : 'No saved receipts yet'}</h3><p>${history ? 'Try a different search or status filter.' : 'Your scanned receipts will appear here.'}</p>${history ? '' : '<button class="button primary" data-route="scan">Scan your first receipt</button>'}</div>`;
}

function receiptRow(receipt) {
  return `<button class="receipt-row" data-receipt="${escapeHtml(receipt.public_id)}"><img src="${escapeHtml(receipt.restored_url)}" alt=""><span class="receipt-main"><b>${escapeHtml(receipt.merchant || receipt.source_name)}</b><small>${escapeHtml(formatDate(receipt.created_at))} · ${escapeHtml(receipt.receipt_number || receipt.public_id)}</small></span><strong>${escapeHtml(amountLabel(receipt.total_amount))}</strong><span class="status-pill ${receipt.status}">${escapeHtml(statusLabel(receipt.status))}</span><span class="row-arrow">›</span></button>`;
}

function wireReceiptLinks(root = document) {
  root.querySelectorAll('[data-receipt]').forEach(el => el.onclick = () => openReceipt(el.dataset.receipt));
  root.querySelectorAll('[data-route]').forEach(el => el.onclick = () => route(el.dataset.route));
}

async function loadDashboard() {
  try {
    const data = await api('/api/dashboard');
    $('statTotal').textContent = data.stats.total;
    $('statReview').textContent = data.stats.needs_review;
    $('statReviewed').textContent = data.stats.reviewed;
    $('recentList').innerHTML = data.recent.length ? data.recent.map(receiptRow).join('') : emptyState('overview');
    wireReceiptLinks($('recentList'));
  } catch (error) { toast(error.message, true); }
}

function receiptCard(receipt) {
  return `<button class="receipt-card" data-receipt="${escapeHtml(receipt.public_id)}"><div class="card-image"><img src="${escapeHtml(receipt.restored_url)}" alt="Recovered image of ${escapeHtml(receipt.source_name)}"><span class="status-pill ${receipt.status}">${escapeHtml(statusLabel(receipt.status))}</span></div><div class="receipt-card-copy"><h3>${escapeHtml(receipt.merchant || receipt.source_name)}</h3><p>${escapeHtml(formatDate(receipt.created_at))}</p><div><span>${escapeHtml(receipt.receipt_number || 'No receipt number')}</span><strong>${escapeHtml(amountLabel(receipt.total_amount))}</strong></div></div></button>`;
}

async function loadHistory() {
  const params = new URLSearchParams({q: $('historySearch').value, status: $('statusFilter').value});
  $('historyGallery').innerHTML = '<div class="loading-state">Loading receipts…</div>';
  try {
    const data = await api(`/api/receipts?${params}`);
    $('historyCount').textContent = `${data.receipts.length} receipt${data.receipts.length === 1 ? '' : 's'}`;
    $('historyGallery').className = `gallery ${historyMode === 'list' ? 'list-mode' : ''}`;
    $('historyGallery').innerHTML = data.receipts.length ? (historyMode === 'grid' ? data.receipts.map(receiptCard).join('') : data.receipts.map(receiptRow).join('')) : emptyState('history');
    wireReceiptLinks($('historyGallery'));
  } catch (error) { $('historyGallery').innerHTML = `<div class="error-inline">${escapeHtml(error.message)}</div>`; }
}

$('historySearch').oninput = () => { clearTimeout(searchTimer); searchTimer = setTimeout(loadHistory, 250); };
$('statusFilter').onchange = loadHistory;
$('gridViewButton').onclick = () => { historyMode = 'grid'; $('gridViewButton').classList.add('active'); $('listViewButton').classList.remove('active'); loadHistory(); };
$('listViewButton').onclick = () => { historyMode = 'list'; $('listViewButton').classList.add('active'); $('gridViewButton').classList.remove('active'); loadHistory(); };

function resetScan() {
  selectedFile = null; $('fileInput').value = '';
  ['previewPanel','processingPanel','scanError'].forEach(id => $(id).classList.add('hidden'));
  $('uploadPanel').classList.remove('hidden');
  $('restoreStep').classList.remove('done'); $('fieldStep').classList.remove('done');
}

function failScan(message) {
  ['uploadPanel','previewPanel','processingPanel'].forEach(id => $(id).classList.add('hidden'));
  $('errorMessage').textContent = message; $('scanError').classList.remove('hidden');
}

function chooseFile(file) {
  if (!file) return;
  if (!file.type.startsWith('image/')) return failScan('Choose an image file such as JPG, PNG, BMP or TIFF.');
  if (file.size > 20 * 1024 * 1024) return failScan('The image is larger than 20 MB. Choose a smaller image.');
  selectedFile = file; $('previewImage').src = URL.createObjectURL(file);
  $('fileName').textContent = file.name; $('fileSize').textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB`;
  $('uploadPanel').classList.add('hidden'); $('scanError').classList.add('hidden'); $('previewPanel').classList.remove('hidden');
}

$('fileInput').onchange = event => chooseFile(event.target.files[0]);
['dragenter','dragover'].forEach(type => $('dropZone').addEventListener(type, event => { event.preventDefault(); $('dropZone').classList.add('dragging'); }));
['dragleave','drop'].forEach(type => $('dropZone').addEventListener(type, event => { event.preventDefault(); $('dropZone').classList.remove('dragging'); }));
$('dropZone').addEventListener('drop', event => chooseFile(event.dataTransfer.files[0]));
$('changeButton').onclick = resetScan; $('tryAgainButton').onclick = resetScan;

let bulkFiles = [];
$('bulkInput').onchange = event => {
  bulkFiles = [...event.target.files].slice(0, 25);
  $('bulkSelection').textContent = bulkFiles.length ? `${bulkFiles.length} receipt${bulkFiles.length === 1 ? '' : 's'} selected` : '';
  $('bulkButton').disabled = bulkFiles.length === 0;
};
$('bulkButton').onclick = async () => {
  if (!bulkFiles.length) return;
  const button = $('bulkButton'); button.disabled = true; button.textContent = `Processing ${bulkFiles.length}…`;
  const body = new FormData(); bulkFiles.forEach(file => body.append('receipts', file));
  try {
    const result = await api('/api/bulk-analyze', {method: 'POST', body});
    toast(`${result.summary.completed} receipt(s) processed; ${result.summary.failed} failed.`);
    bulkFiles = []; $('bulkInput').value = ''; $('bulkSelection').textContent = ''; route('history');
  } catch (error) { toast(error.message, true); }
  finally { button.disabled = bulkFiles.length === 0; button.textContent = 'Process batch'; }
};

$('analyseButton').onclick = async () => {
  if (!selectedFile) return;
  $('previewPanel').classList.add('hidden'); $('processingPanel').classList.remove('hidden');
  const titles = ['Preparing your image…', 'Recovering faded text…', 'Finding receipt fields…', 'Saving your record…']; let index = 0;
  const timer = setInterval(() => { index = Math.min(index + 1, titles.length - 1); $('processingTitle').textContent = titles[index]; if (index > 0) $('restoreStep').classList.add('done'); if (index > 1) $('fieldStep').classList.add('done'); }, 1500);
  try {
    const body = new FormData(); body.append('receipt', selectedFile);
    const data = await api('/api/analyze', {method: 'POST', body});
    toast('Receipt recovered and saved to history.'); await openReceipt(data.receipt.public_id);
  } catch (error) { failScan(error.message); }
  finally { clearInterval(timer); }
};

async function openReceipt(id) {
  try {
    currentReceipt = await api(`/api/receipts/${encodeURIComponent(id)}`);
    sessionStorage.setItem('lastReceiptId', id);
    $('detailTitle').textContent = currentReceipt.merchant || currentReceipt.source_name;
    $('detailMeta').textContent = `${formatDate(currentReceipt.created_at)} · ${currentReceipt.public_id}`;
    $('detailStatus').textContent = statusLabel(currentReceipt.status); $('detailStatus').className = `status-pill ${currentReceipt.status}`;
    $('originalImage').src = currentReceipt.original_url; $('restoredImage').src = currentReceipt.restored_url;
    $('merchantField').value = currentReceipt.merchant || ''; $('dateField').value = currentReceipt.transaction_date || '';
    $('numberField').value = currentReceipt.receipt_number || ''; $('totalField').value = currentReceipt.total_amount || '';
    $('reviewStatus').value = currentReceipt.status; $('tagsField').value = currentReceipt.tags.join(', '); $('notesField').value = currentReceipt.notes || '';
    $('runtimeStatus').textContent = `${Number(currentReceipt.runtime_seconds).toFixed(2)} seconds`; $('recordId').textContent = currentReceipt.public_id;
    const integrity = currentReceipt.integrity?.status || 'legacy_unverified';
    $('integrityStatus').textContent = integrity === 'verified' ? 'Evidence files verified' : integrity === 'failed' ? 'Integrity check failed' : 'Legacy record · not hashed';
    $('integrityStatus').className = `status-pill integrity-${integrity}`;
    const missing = ['merchant','transaction_date','receipt_number','total_amount'].filter(key => !currentReceipt[key]);
    const arithmetic = currentReceipt.audit?.arithmetic_consistency?.status;
    const arithmeticMessage = arithmetic === 'inconsistent' ? ' The detected line amounts do not agree; correct them before using this record.' : arithmetic === 'consistent' ? ' The detected line arithmetic is consistent, but still requires comparison with the original.' : ' Line arithmetic could not be checked from the detected content.';
    $('readingMessage').textContent = (missing.length ? `We could not confidently read ${missing.length} important ${missing.length === 1 ? 'detail' : 'details'}. Enter them manually before marking this receipt as reviewed.` : 'All key details were found. Compare them with the original before marking this receipt as reviewed.') + arithmeticMessage;
    const lines = dedupeOcrLines(currentReceipt.audit?.ocr_lines || []);
    const suggestions = new Map((currentReceipt.audit?.ocr_suggestions || []).map(item => [Number(item.line_index), item]));
    $('ocrTable').innerHTML = lines.length ? `<table><thead><tr><th>#</th><th>Scanner reading</th><th>Possible correction</th><th>Confidence</th></tr></thead><tbody>${lines.map((line, index) => { const suggestion = suggestions.get(index); return `<tr><td>${index + 1}</td><td>${escapeHtml(line.text)}</td><td>${suggestion ? `<span class="suggested-reading">${escapeHtml(suggestion.suggested)}</span><small>Verify against original</small>` : '<span class="no-suggestion">—</span>'}</td><td>${line.confidence == null ? '—' : `${(Number(line.confidence) * 100).toFixed(1)}%`}</td></tr>`; }).join('')}</tbody></table>` : '<p class="empty-copy">No text was recognised. Enter the receipt details manually.</p>';
    $('auditDownload').href = `/api/receipts/${encodeURIComponent(id)}/audit.json`; $('ocrDownload').href = `/api/receipts/${encodeURIComponent(id)}/ocr.xlsx`;
    setImageMode('improved'); route('detail');
  } catch (error) { toast(error.message, true); }
}

function dedupeOcrLines(lines) {
  const kept = [];
  const normalise = value => String(value || '').toLowerCase().replace(/[^\p{L}\p{N}]/gu, '');
  const bounds = line => {
    const box = line?.box;
    if (!Array.isArray(box) || !box.length) return null;
    const xs = box.map(point => Number(point?.[0])), ys = box.map(point => Number(point?.[1]));
    return xs.every(Number.isFinite) && ys.every(Number.isFinite) ? [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)] : null;
  };
  const samePlace = (a, b) => {
    const x = bounds(a), y = bounds(b); if (!x || !y) return !x && !y;
    const aw = Math.max(x[2]-x[0],1), ah = Math.max(x[3]-x[1],1), bw = Math.max(y[2]-y[0],1), bh = Math.max(y[3]-y[1],1);
    return Math.abs((x[0]+x[2]-y[0]-y[2])/2) <= Math.max(12,.18*Math.max(aw,bw)) && Math.abs((x[1]+x[3]-y[1]-y[3])/2) <= Math.max(8,.45*Math.max(ah,bh));
  };
  for (const line of lines) {
    const text = normalise(line.text); if (!text) continue;
    const index = kept.findIndex(old => normalise(old.text) === text && samePlace(old,line));
    if (index < 0) kept.push(line); else if (Number(line.confidence || 0) > Number(kept[index].confidence || 0)) kept[index] = line;
  }
  return kept;
}

function updateComparison(value) { $('restoredWrap').style.clipPath = `inset(0 ${100 - value}% 0 0)`; $('compareLine').style.left = `${value}%`; }
$('compareSlider').oninput = event => updateComparison(event.target.value);

function setImageMode(mode) {
  const comparison = $('comparison');
  comparison.classList.remove('improved-mode','original-mode','compare-mode'); comparison.classList.add(`${mode}-mode`);
  for (const [name,id] of [['improved','improvedViewButton'],['original','originalViewButton'],['compare','compareViewButton']]) $(''+id).classList.toggle('active', name === mode);
  if (mode === 'compare') { $('compareSlider').value = 50; updateComparison(50); }
  $('comparisonViewLabel').textContent = mode === 'improved' ? 'Clean copy · complete image' : mode === 'original' ? 'Original receipt · complete image' : 'Original and clean-copy comparison';
  $('comparisonHelp').textContent = mode === 'compare' ? 'Drag the divider' : 'Select Compare for the split view';
}
$('improvedViewButton').onclick = () => setImageMode('improved');
$('originalViewButton').onclick = () => setImageMode('original');
$('compareViewButton').onclick = () => setImageMode('compare');

let imageZoom = 1;
function updateFullImageZoom() {
  $('fullImage').style.width = `${Math.round(imageZoom * 100)}%`;
  $('zoomLevel').textContent = `${Math.round(imageZoom * 100)}%`;
}
function openFullImage() {
  if (!currentReceipt) return;
  $('fullImage').src = currentReceipt.restored_url;
  imageZoom = 1; updateFullImageZoom(); $('imageDialog').showModal();
}
$('fullImageButton').onclick = openFullImage;
$('comparison').ondblclick = openFullImage;
$('closeImageButton').onclick = () => $('imageDialog').close();
$('zoomInButton').onclick = () => { imageZoom = Math.min(3, imageZoom + .25); updateFullImageZoom(); };
$('zoomOutButton').onclick = () => { imageZoom = Math.max(.5, imageZoom - .25); updateFullImageZoom(); };
$('zoomResetButton').onclick = () => { imageZoom = 1; updateFullImageZoom(); };

$('reviewForm').onsubmit = async event => {
  event.preventDefault(); if (!currentReceipt) return;
  const payload = {merchant: $('merchantField').value.trim(), transaction_date: $('dateField').value.trim(), receipt_number: $('numberField').value.trim(), total_amount: $('totalField').value.trim(), status: $('reviewStatus').value, tags: $('tagsField').value.split(',').map(v => v.trim()).filter(Boolean), notes: $('notesField').value.trim()};
  try {
    currentReceipt = await api(`/api/receipts/${encodeURIComponent(currentReceipt.public_id)}`, {method: 'PATCH', body: payload});
    $('detailStatus').textContent = statusLabel(currentReceipt.status); $('detailStatus').className = `status-pill ${currentReceipt.status}`;
    $('detailTitle').textContent = currentReceipt.merchant || currentReceipt.source_name;
    $('saveMessage').textContent = 'Saved successfully.'; toast('Receipt review saved.'); setTimeout(() => $('saveMessage').textContent = '', 2500);
  } catch (error) { $('saveMessage').textContent = error.message; }
};

$('deleteButton').onclick = async () => {
  if (!currentReceipt || !confirm('Remove this receipt from your history? This cannot be undone from the app.')) return;
  try { await api(`/api/receipts/${encodeURIComponent(currentReceipt.public_id)}`, {method: 'DELETE'}); toast('Receipt removed.'); route('history'); }
  catch (error) { toast(error.message, true); }
};

async function loadAdmin() {
  try {
    const data = await api('/api/admin/overview');
    $('adminUsers').textContent = data.stats.users; $('adminPending').textContent = data.users.filter(account => account.status === 'suspended').length; $('adminReceipts').textContent = data.stats.receipts;
    $('userTable').innerHTML = `<table><thead><tr><th>Name</th><th>Email</th><th>Role</th><th>Status</th><th>Action</th></tr></thead><tbody>${data.users.map(account => `<tr><td>${escapeHtml(account.name)}</td><td>${escapeHtml(account.email)}</td><td>${escapeHtml(account.role)}</td><td><span class="status-pill ${account.status}">${escapeHtml(account.status)}</span></td><td>${account.public_id === user.public_id ? '<small>Current account</small>' : `<select data-user-action="${escapeHtml(account.public_id)}"><option value="">Change…</option><option value="active">Approve / activate</option><option value="suspended">Suspend</option><option value="role:admin">Make admin</option><option value="role:user">Make user</option></select>`}</td></tr>`).join('')}</tbody></table>`;
    $('userTable').querySelectorAll('[data-user-action]').forEach(select => select.onchange = async () => {
      if (!select.value) return; const payload = select.value.startsWith('role:') ? {role: select.value.split(':')[1]} : {status: select.value};
      try { await api(`/api/admin/users/${encodeURIComponent(select.dataset.userAction)}`, {method: 'PATCH', body: payload}); toast('Account updated.'); loadAdmin(); } catch (error) { toast(error.message, true); }
    });
    $('eventList').innerHTML = data.events.length ? data.events.map(event => `<div><b>${escapeHtml(event.action)}</b><span>${escapeHtml(event.actor_name || 'System')} · ${escapeHtml(formatDate(event.occurred_at))}</span><small>${escapeHtml(event.object_type)}${event.object_id ? ` · ${escapeHtml(event.object_id)}` : ''} · ${escapeHtml(event.outcome)}</small></div>`).join('') : '<p>No recorded activity.</p>';
  } catch (error) { toast(error.message, true); }
}
$('refreshAdmin').onclick = loadAdmin;

const cameraDialog = $('cameraDialog');
$('cameraButton').onclick = async () => {
  try { cameraStream = await navigator.mediaDevices.getUserMedia({video: {facingMode: {ideal: 'environment'}}}); $('cameraVideo').srcObject = cameraStream; cameraDialog.showModal(); }
  catch (_) { failScan('The camera could not be opened. Check camera permission or upload an image instead.'); }
};
function stopCamera() { if (cameraStream) cameraStream.getTracks().forEach(track => track.stop()); cameraStream = null; }
cameraDialog.addEventListener('close', stopCamera);
$('captureButton').onclick = () => {
  const video = $('cameraVideo'), canvas = $('cameraCanvas'); canvas.width = video.videoWidth; canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0); canvas.toBlob(blob => { stopCamera(); cameraDialog.close(); chooseFile(new File([blob], `receipt-${Date.now()}.jpg`, {type: 'image/jpeg'})); }, 'image/jpeg', .92);
};

function toast(message, error = false) {
  const el = $('toast'); el.textContent = message; el.className = `toast ${error ? 'error' : ''}`;
  setTimeout(() => el.classList.add('hidden'), 3200);
}

window.addEventListener('hashchange', () => { if (user) route(location.hash.slice(1), false); });

(async function boot() {
  try {
    const providers = await api('/api/auth/providers');
    if (!providers.google.enabled) {
      $('googleLogin').classList.add('disabled'); $('googleLogin').setAttribute('aria-disabled', 'true');
      $('googleLogin').onclick = event => event.preventDefault();
      $('googleStatus').textContent = 'Google sign-in requires the installation owner to configure OAuth credentials.';
    } else {
      $('googleStatus').textContent = 'Sign in securely with your Google Account. Receipt data remains on this device.';
    }
    const authError = new URLSearchParams(location.search).get('auth_error');
    if (authError) {
      const messages = {google_not_configured:'Google sign-in is not configured yet.',google_state:'Google sign-in could not be verified. Try again.',google_denied:'Google sign-in was cancelled.',google_unverified:'Google did not confirm this email address.',account_suspended:'This account has been suspended. Contact the application administrator.',google_failed:'Google sign-in could not be completed.'};
      $('authError').textContent = messages[authError] || 'Sign-in could not be completed.'; $('authError').classList.remove('hidden');
      history.replaceState(null, '', '/');
    }
    const state = await api('/api/auth'); csrf = state.csrf_token; user = state.user;
    if (user) showWorkspace(); else showAuth();
  } catch (_) { showAuth(); $('authError').textContent = 'We could not connect just now. Please refresh the page and try again.'; $('authError').classList.remove('hidden'); }
})();
