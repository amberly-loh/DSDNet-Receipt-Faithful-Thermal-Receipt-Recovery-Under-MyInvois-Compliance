const $ = id => document.getElementById(id);
const setValue = (id, value = '') => {
  const element = $(id);
  if (element) element.value = value ?? '';
  return element;
};
const fieldValue = id => String($(id)?.value ?? '').trim();
const pages = ['overview', 'scan', 'history', 'detail', 'admin'];
let csrf = '', user = null, authMode = 'login', selectedFile = null, cameraStream = null;
let currentReceipt = null, historyMode = 'grid', searchTimer = null;

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
}

async function api(path, options = {}, allowSessionRetry = true) {
  const timeoutMs = Number(options.timeoutMs || 0);
  const config = {...options, headers: {...(options.headers || {})}};
  delete config.timeoutMs;
  const controller = timeoutMs > 0 ? new AbortController() : null;
  const timeout = controller ? setTimeout(() => controller.abort(), timeoutMs) : null;
  if (controller) config.signal = controller.signal;
  if (['POST', 'PATCH', 'DELETE'].includes((config.method || 'GET').toUpperCase())) config.headers['X-CSRF-Token'] = csrf;
  if (config.body && !(config.body instanceof FormData)) {
    config.headers['Content-Type'] = 'application/json';
    if (typeof config.body !== 'string') config.body = JSON.stringify(config.body);
  }
  let response;
  try { response = await fetch(path, config); }
  catch (error) {
    if (error?.name === 'AbortError') throw new Error('Processing timed out. The file remains queued for retry.');
    throw error;
  } finally { if (timeout) clearTimeout(timeout); }
  const type = response.headers.get('content-type') || '';
  const data = type.includes('application/json') ? await response.json() : null;
  if (data?.csrf_token) csrf = data.csrf_token;
  if (!response.ok) {
    if (response.status === 403 && allowSessionRetry && data?.error === 'Your session changed. Refresh the page and try again.') {
      const refreshed = await fetch('/api/auth', {cache: 'no-store'});
      const state = await refreshed.json();
      if (refreshed.ok && state?.csrf_token) {
        csrf = state.csrf_token;
        user = state.user;
        return api(path, options, false);
      }
    }
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
  $('profileEmail').textContent = user.is_guest ? 'Guest session · local device' : user.email;
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
    button.disabled = false; button.textContent = authMode === 'login' ? 'Sign in' : 'Create account';
  }
};

$('guestButton').onclick = async () => {
  const button = $('guestButton'); button.disabled = true;
  try {
    const data = await api('/api/guest', {method: 'POST'});
    user = data.user; sessionStorage.removeItem('lastReceiptId'); history.replaceState(null, '', '#overview'); showWorkspace();
  } catch (error) { $('authError').textContent = error.message; $('authError').classList.remove('hidden'); }
  finally { button.disabled = false; }
};

$('logoutButton').onclick = async () => {
  if (user?.is_guest && !confirm('Guest receipts cannot be reopened after signing out. Export what you need first. Files remain on this computer. Sign out?')) return;
  try { await api('/api/logout', {method: 'POST'}); } finally {
    selectedReceipts.clear(); sessionStorage.removeItem('lastReceiptId'); currentReceipt = null;
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
function statusLabel(status) { return ({needs_review: 'To verify', reviewed: 'Reviewed', archived: 'Archived'})[status] || status; }
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

const selectedReceipts = new Set();
let selectionRequest = 0;
function updateHistorySelection() {
  const boxes = [...$('historyGallery').querySelectorAll('[data-select-receipt]')];
  $('selectionCount').textContent = `${selectedReceipts.size} selected`;
  $('deleteSelectedReceipts').disabled = selectedReceipts.size === 0;
  $('selectVisibleReceipts').disabled = boxes.length === 0;
  $('selectVisibleReceipts').checked = boxes.length > 0 && boxes.every(box => box.checked);
  $('selectVisibleReceipts').indeterminate = boxes.some(box => box.checked) && !boxes.every(box => box.checked);
  const request = ++selectionRequest;
  const summary = $('selectionSummary');
  if (!selectedReceipts.size) { summary.textContent = ''; return; }
  summary.textContent = 'Calculating selected amounts…';
  api('/api/receipts/selection-summary', {method:'POST', body:{ids:[...selectedReceipts]}}).then(data => {
    if (request !== selectionRequest) return;
    summary.textContent = `Available amounts: RM ${data.total} (MYR assumed) · ${data.count} selected` +
      (data.missing_amounts ? ` · ${data.missing_amounts} without a usable amount` : '') +
      (data.not_reviewed ? ` · ${data.not_reviewed} not marked reviewed` : '') +
      (data.duplicate_images ? ` · ${data.duplicate_images} duplicate image(s) included` : '');
  }).catch(error => { if (request === selectionRequest) summary.textContent = `Could not calculate total: ${error.message}`; });
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
    // Selection applies only to the currently shown results.
    const visibleIds = new Set(data.receipts.map(receipt => receipt.public_id));
    for (const id of selectedReceipts) if (!visibleIds.has(id)) selectedReceipts.delete(id);
    $('historyGallery').innerHTML = data.receipts.length ? data.receipts.map(receipt => `<div class="selectable-receipt"><label class="receipt-select"><input type="checkbox" data-select-receipt="${escapeHtml(receipt.public_id)}" ${selectedReceipts.has(receipt.public_id) ? 'checked' : ''} aria-label="Select ${escapeHtml(receipt.merchant || receipt.source_name)}"> Select</label>${historyMode === 'grid' ? receiptCard(receipt) : receiptRow(receipt)}</div>`).join('') : emptyState('history');
    wireReceiptLinks($('historyGallery'));
    $('historyGallery').querySelectorAll('[data-select-receipt]').forEach(box => box.onchange = () => { if (box.checked) selectedReceipts.add(box.dataset.selectReceipt); else selectedReceipts.delete(box.dataset.selectReceipt); updateHistorySelection(); });
    updateHistorySelection();
  } catch (error) { $('historyGallery').innerHTML = `<div class="error-inline">${escapeHtml(error.message)}</div>`; }
}

$('selectVisibleReceipts').onchange = event => {
  $('historyGallery').querySelectorAll('[data-select-receipt]').forEach(box => { box.checked = event.target.checked; if (box.checked) selectedReceipts.add(box.dataset.selectReceipt); else selectedReceipts.delete(box.dataset.selectReceipt); });
  updateHistorySelection();
};
$('deleteSelectedReceipts').onclick = async () => {
  const ids = [...selectedReceipts];
  if (!ids.length || !confirm(`Remove ${ids.length} selected receipt(s) from your history? This cannot be undone from the app.`)) return;
  $('deleteSelectedReceipts').disabled = true;
  let removed = 0;
  for (const id of ids) {
    try { await api(`/api/receipts/${encodeURIComponent(id)}`, {method: 'DELETE'}); selectedReceipts.delete(id); removed++; }
    catch (error) { toast(`Removed ${removed}. ${error.message}`, true); await loadHistory(); return; }
  }
  toast(`${removed} receipt(s) removed.`);
  await loadHistory();
};

$('historySearch').oninput = () => { clearTimeout(searchTimer); searchTimer = setTimeout(loadHistory, 250); };
$('statusFilter').onchange = loadHistory;
$('gridViewButton').onclick = () => { historyMode = 'grid'; $('gridViewButton').classList.add('active'); $('listViewButton').classList.remove('active'); loadHistory(); };
$('listViewButton').onclick = () => { historyMode = 'list'; $('listViewButton').classList.add('active'); $('gridViewButton').classList.remove('active'); loadHistory(); };

function resetScan() {
  selectedFile = null; setValue('fileInput');
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
  const known = new Set(bulkFiles.map(file => `${file.name}:${file.size}:${file.lastModified}`));
  const additions = [...event.target.files].filter(file => !known.has(`${file.name}:${file.size}:${file.lastModified}`));
  bulkFiles = [...bulkFiles, ...additions].slice(0, 100);
  event.target.value = '';
  $('bulkSelection').textContent = bulkFiles.length ? `${bulkFiles.length}/100 receipts queued` : '';
  $('bulkButton').disabled = bulkFiles.length === 0;
};
$('bulkButton').onclick = async () => {
  if (!bulkFiles.length) return;
  const button = $('bulkButton'); button.disabled = true; button.textContent = `Processing ${bulkFiles.length}…`;
  try {
    const pending = [...bulkFiles], failed = [];
    let completed = 0;
    // Send bounded requests so a large selection does not exceed the upload
    // limit or hold a single connection open for the entire batch.
    for (let index = 0; index < pending.length; index++) {
      button.textContent = `Processing ${index + 1}/${pending.length}…`;
      $('bulkSelection').textContent = `${completed} saved; processing ${pending[index].name}`;
      const body = new FormData(); body.append('receipts', pending[index]);
      try {
        const result = await api('/api/bulk-analyze', {method: 'POST', body, timeoutMs: 330000});
        completed += result.summary.completed;
        if (result.summary.failed) {
          failed.push(pending[index]);
          $('bulkSelection').textContent = `${pending[index].name}: ${result.failed?.[0]?.error || 'Processing failed'}`;
          toast($('bulkSelection').textContent, true);
        }
      } catch (error) {
        failed.push(pending[index]);
        toast(`${pending[index].name}: ${error.message}`, true);
      }
      // Yield between expensive receipts so progress, navigation and cancel
      // events remain responsive even for a 100-image queue.
      await new Promise(resolve => requestAnimationFrame(() => resolve()));
    }
    bulkFiles = failed;
    $('bulkSelection').textContent = failed.length ? `${failed.length} receipt(s) remain for retry` : '';
    if (!failed.length) setValue('bulkInput');
    toast(`${completed} receipt(s) saved; ${failed.length} remaining.`, failed.length > 0);
    if (completed && !failed.length) route('history');
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
    // Accept legacy API responses while the server canonicalises old rows.
    if (!currentReceipt.accounting_fields && currentReceipt.accounting_fields_json) {
      try { currentReceipt.accounting_fields = JSON.parse(currentReceipt.accounting_fields_json); }
      catch (_) { currentReceipt.accounting_fields = {}; }
    }
    sessionStorage.setItem('lastReceiptId', id);
    $('detailTitle').textContent = currentReceipt.merchant || currentReceipt.source_name;
    $('detailMeta').textContent = `${formatDate(currentReceipt.created_at)} · ${currentReceipt.public_id}`;
    $('detailStatus').textContent = currentReceipt.review_readiness?.label || statusLabel(currentReceipt.status); $('detailStatus').className = `status-pill ${currentReceipt.review_readiness?.state || currentReceipt.status}`;
    $('originalImage').src = currentReceipt.original_url; $('restoredImage').src = currentReceipt.faithful_url;
    setValue('merchantField', currentReceipt.merchant); setValue('dateField', currentReceipt.transaction_date);
    setValue('numberField', currentReceipt.receipt_number); setValue('totalField', currentReceipt.total_amount);
    setValue('reviewStatus', currentReceipt.status); setValue('tagsField', (currentReceipt.tags || []).join(', ')); setValue('notesField', currentReceipt.notes);
    const accounting = currentReceipt.accounting_fields || {};
    for (const [key,id] of Object.entries({document_type:'documentTypeField',currency:'currencyField',supplier_registration_number:'supplierRegistrationField',supplier_tax_number:'supplierTaxField',supplier_address:'supplierAddressField',supplier_phone:'supplierPhoneField',supplier_email:'supplierEmailField',payment_method:'paymentMethodField',subtotal:'subtotalField',discount:'discountField',tax_amount:'taxAmountField'})) setValue(id, accounting[key]);
    renderLineItems(currentReceipt.line_items || []);
    renderReadiness(currentReceipt);
    const duplicateCount = (currentReceipt.possible_duplicates || []).length;
    $('duplicateWarning').classList.toggle('hidden', !duplicateCount);
    $('duplicateWarning').textContent = duplicateCount ? `Possible duplicate: ${duplicateCount} matching receipt record${duplicateCount === 1 ? '' : 's'} found. Verify before recording it again.` : '';
    const evidence = new Map((currentReceipt.audit?.fields || []).map(field => [field.name, field]));
    for (const [name, elementId] of [['merchant','merchantEvidence'],['transaction_date','dateEvidence'],['receipt_invoice_number','numberEvidence'],['total_amount','totalEvidence']]) {
      const field = evidence.get(name), confidence = field?.ocr_confidence;
      $(elementId).textContent = field?.value ? `${confidence == null ? 'Located in receipt' : `${(Number(confidence) * 100).toFixed(1)}% scanner score (not a guarantee)`} · verify against original` : 'Not read automatically · enter if shown';
    }
    $('runtimeStatus').textContent = `${Number(currentReceipt.runtime_seconds).toFixed(2)} seconds`; $('recordId').textContent = currentReceipt.public_id;
    const integrity = currentReceipt.integrity?.status || 'legacy_unverified';
    $('integrityStatus').textContent = integrity === 'verified' ? 'Evidence files verified' : integrity === 'failed' ? 'Integrity check failed' : 'Legacy record · not hashed';
    $('integrityStatus').className = `status-pill integrity-${integrity}`;
    const missing = ['merchant','transaction_date','receipt_number','total_amount'].filter(key => !currentReceipt[key]);
    const arithmetic = currentReceipt.audit?.arithmetic_consistency?.status;
    const arithmeticMessage = arithmetic === 'inconsistent' ? ' The scanner detected an arithmetic mismatch; compare the line items with the original.' : arithmetic === 'consistent' ? ' The detected line arithmetic is consistent, but still requires comparison with the original.' : ' Line arithmetic could not be checked from the detected content.';
    $('readingMessage').textContent = (missing.length ? `The scanner needs attention on ${missing.length} important ${missing.length === 1 ? 'detail' : 'details'}. Correct only those values before confirming the receipt.` : `${currentReceipt.line_items?.length || 0} line item${currentReceipt.line_items?.length === 1 ? '' : 's'} and all primary details were found. Confirm them against the image.`) + arithmeticMessage;
    const lines = dedupeOcrLines(currentReceipt.audit?.ocr_lines || []);
    const suggestions = new Map((currentReceipt.audit?.ocr_suggestions || []).map(item => [Number(item.line_index), item]));
    $('ocrTable').innerHTML = lines.length ? `<table><thead><tr><th>#</th><th>Scanner reading</th><th>Possible correction</th><th>Confidence</th></tr></thead><tbody>${lines.map((line, index) => { const suggestion = suggestions.get(index); return `<tr><td>${index + 1}</td><td>${escapeHtml(line.text)}</td><td>${suggestion ? `<span class="suggested-reading">${escapeHtml(suggestion.suggested)}</span><small>Verify against original</small>` : '<span class="no-suggestion">—</span>'}</td><td>${line.confidence == null ? '—' : `${(Number(line.confidence) * 100).toFixed(1)}%`}</td></tr>`; }).join('')}</tbody></table>` : '<p class="empty-copy">No text was recognised. Enter the receipt details manually.</p>';
    $('auditDownload').href = `/api/receipts/${encodeURIComponent(id)}/audit.pdf`; $('accountingDownload').href = `/api/receipts/${encodeURIComponent(id)}/ocr.xlsx`; $('technicalDownload').href = `/api/receipts/${encodeURIComponent(id)}/audit.json`; $('documentDownload').href = currentReceipt.restored_url;
    setImageMode('faithful'); route('detail');
  } catch (error) { toast(error.message, true); }
}

function renderLineItems(items) {
  const keys = ['item','description','quantity','unit','unit_price','discount','amount','review_note'];
  const labels = ['Item','Description','Qty','Unit','Unit price','Discount','Amount','Review note'];
  const rows = items.length ? items : [{item:'',description:'',quantity:'',unit:'',unit_price:'',discount:'',amount:'',review_note:''}];
  $('lineItemEditor').innerHTML = `<table><thead><tr>${labels.map(label => `<th>${label}</th>`).join('')}<th></th></tr></thead><tbody>${rows.map((row,index) => `<tr>${keys.map(key => `<td><input data-line-index="${index}" data-line-key="${key}" value="${escapeHtml(row[key] || '')}"></td>`).join('')}<td><button class="button secondary compact remove-line-item" type="button" data-line-index="${index}">Remove</button></td></tr>`).join('')}</tbody></table>`;
  document.querySelectorAll('.remove-line-item').forEach(button => button.onclick = () => { const updated = collectLineItems(); updated.splice(Number(button.dataset.lineIndex),1); renderLineItems(updated); });
}

function collectLineItems() {
  const rows = new Map();
  document.querySelectorAll('#lineItemEditor input[data-line-index]').forEach(input => { const index=Number(input.dataset.lineIndex); if(!rows.has(index)) rows.set(index,{}); rows.get(index)[input.dataset.lineKey]=input.value.trim(); });
  return [...rows.values()].filter(row => Object.values(row).some(Boolean));
}

function renderReadiness(receipt) {
  const a = receipt.accounting_fields || {};
  const rows = [
    ['Document', [a.document_type, receipt.receipt_number].filter(Boolean).join(' · ') || 'Not captured', receipt.receipt_number ? 'Extracted' : 'Needs attention'],
    ['Supplier', a.supplier_name || receipt.merchant || 'Not captured', (a.supplier_registration_number || a.supplier_tax_number) ? 'Identity evidence extracted' : 'Name extracted; registration/tax number not shown or unreadable'],
    ['Transaction', [receipt.transaction_date, a.currency || 'MYR', receipt.total_amount].filter(Boolean).join(' · '), receipt.transaction_date && receipt.total_amount ? 'Extracted' : 'Needs attention'],
    ['Line items', `${receipt.line_items?.length || 0} row(s)`, receipt.line_items?.length ? 'Extracted; confirm uncertain cells' : 'No item row confirmed'],
    ['Tax evidence', [a.subtotal && `subtotal ${a.subtotal}`, a.discount && `discount ${a.discount}`, a.tax_amount && `tax ${a.tax_amount}`].filter(Boolean).join(' · ') || 'Not printed or not detected', 'Used as supporting accounting evidence'],
    ['MyInvois workflow', 'Supporting evidence prepared', 'This app prepares evidence; authorised submission remains separate'],
  ];
  $('myinvoisChecklist').innerHTML = `<table><thead><tr><th>Field group</th><th>Current value</th><th>Status</th></tr></thead><tbody>${rows.map(row => `<tr><td>${escapeHtml(row[0])}</td><td>${escapeHtml(row[1] || '—')}</td><td>${escapeHtml(row[2])}</td></tr>`).join('')}</tbody></table>`;
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
  if (!['faithful','reading','original','compare'].includes(mode)) mode = 'faithful';
  const comparison = $('comparison');
  $('originalImage').src = mode === 'compare' ? (currentReceipt?.faithful_reference_url || currentReceipt?.original_url || '') : (currentReceipt?.original_url || '');
  comparison.classList.remove('improved-mode','original-mode','compare-mode');
  comparison.classList.add(`${mode === 'original' ? 'original' : mode === 'compare' ? 'compare' : 'improved'}-mode`);
  for (const [name,id] of [['faithful','faithfulViewButton'],['reading','readingViewButton'],['original','originalViewButton'],['compare','compareViewButton']]) $(''+id).classList.toggle('active', name === mode);
  if (mode === 'faithful' || mode === 'compare') $('restoredImage').src = currentReceipt?.faithful_url || currentReceipt?.restored_url || '';
  if (mode === 'reading') $('restoredImage').src = currentReceipt?.reading_url || currentReceipt?.faithful_url || '';
  if (mode === 'compare') { setValue('compareSlider', 50); updateComparison(50); }
  const gate = currentReceipt?.audit?.faithfulness_gate;
  const audit = currentReceipt?.audit || {};
  const sourceView = Boolean(audit.release_content_fix || audit.faithful_processing || audit.native_refinement?.applied || audit.display_image === 'faithful_recovery');
  const rejected = gate?.accepted === false && !sourceView;
  $('comparisonViewLabel').textContent = mode === 'faithful' ? (rejected ? 'Original retained' : 'Faithful recovery') : mode === 'reading' ? 'OCR-assisted reconstructed reading' : mode === 'original' ? 'Original upload' : 'Original and faithful recovery';
  $('comparisonNote').textContent = mode === 'reading' ? 'High-confidence scanner readings are typeset in their detected positions. This readability aid is reconstructed content, not the original evidence image.' : mode === 'original' ? 'The original upload is preserved.' : rejected ? 'The recovery was rejected. Review the retained original.' : 'Review text, signatures, logos and faint details against the original. Unreadable characters cannot be confirmed by enhancement.';
  $('comparisonHelp').textContent = mode === 'compare' ? 'Drag the divider' : 'Select Compare for the split view';
}
$('faithfulViewButton').onclick = () => setImageMode('faithful');
$('readingViewButton').onclick = () => setImageMode('reading');
$('originalViewButton').onclick = () => setImageMode('original');
$('compareViewButton').onclick = () => setImageMode('compare');

let imageZoom = 1;
function updateFullImageZoom() {
  $('fullImage').style.width = `${Math.round(imageZoom * 100)}%`;
  $('zoomLevel').textContent = `${Math.round(imageZoom * 100)}%`;
}
function openFullImage() {
  if (!currentReceipt) return;
  $('fullImage').src = $('originalViewButton').classList.contains('active') ? currentReceipt.original_url : $('readingViewButton').classList.contains('active') ? currentReceipt.reading_url : currentReceipt.faithful_url;
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
  const accounting_fields = {...(currentReceipt.accounting_fields || {}),supplier_name:fieldValue('merchantField'),document_number:fieldValue('numberField'),transaction_date:fieldValue('dateField'),payable_amount:fieldValue('totalField'),document_type:fieldValue('documentTypeField'),currency:fieldValue('currencyField'),supplier_registration_number:fieldValue('supplierRegistrationField'),supplier_tax_number:fieldValue('supplierTaxField'),supplier_address:fieldValue('supplierAddressField'),supplier_phone:fieldValue('supplierPhoneField'),supplier_email:fieldValue('supplierEmailField'),payment_method:fieldValue('paymentMethodField'),subtotal:fieldValue('subtotalField'),discount:fieldValue('discountField'),tax_amount:fieldValue('taxAmountField'),line_item_count:collectLineItems().length};
  const payload = {merchant: fieldValue('merchantField'), transaction_date: fieldValue('dateField'), receipt_number: fieldValue('numberField'), total_amount: fieldValue('totalField'), status: fieldValue('reviewStatus'), tags: fieldValue('tagsField').split(',').map(v => v.trim()).filter(Boolean), notes: fieldValue('notesField'), line_items: collectLineItems(), accounting_fields};
  try {
    currentReceipt = await api(`/api/receipts/${encodeURIComponent(currentReceipt.public_id)}`, {method: 'PATCH', body: payload});
    $('detailStatus').textContent = statusLabel(currentReceipt.status); $('detailStatus').className = `status-pill ${currentReceipt.status}`;
    $('detailTitle').textContent = currentReceipt.merchant || currentReceipt.source_name;
    $('saveMessage').textContent = 'Saved successfully.'; toast('Receipt review saved.'); setTimeout(() => $('saveMessage').textContent = '', 2500);
  } catch (error) { $('saveMessage').textContent = [error.message, ...Object.values(error.details?.fields || {})].join(' '); }
};

$('addLineItemButton').onclick = () => { const rows = collectLineItems(); rows.push({item:'',description:'',quantity:'',unit:'',unit_price:'',discount:'',amount:'',review_note:''}); renderLineItems(rows); };

$('copyFieldsButton').onclick = async () => {
  if (!currentReceipt) return;
  const itemText = collectLineItems().map((row,index)=>`Item ${index+1}: ${row.description || '—'} | Qty ${row.quantity || '—'} | Unit price ${row.unit_price || '—'} | Amount ${row.amount || '—'}`).join('\n');
  const text = [`Merchant: ${$('merchantField').value}`, `Supplier registration: ${$('supplierRegistrationField').value}`, `Supplier tax number: ${$('supplierTaxField').value}`, `Transaction date: ${$('dateField').value}`, `Receipt / invoice number: ${$('numberField').value}`, `Total amount (MYR): ${$('totalField').value}`, itemText].filter(Boolean).join('\n');
  try { await navigator.clipboard.writeText(text); toast('Visible details copied. Verify against the original.'); }
  catch (_) { toast('Your browser blocked clipboard access.', true); }
};

$('reextractButton').onclick = async () => {
  if (!currentReceipt) return;
  if (!confirm('Re-reading replaces saved merchant, date, number and total with scanner results and regenerates the recovery image. Unsaved edits will also be lost. Continue?')) return;
  try { await api(`/api/receipts/${encodeURIComponent(currentReceipt.public_id)}/reextract`, {method: 'POST'}); toast('Fields re-read. Verify them against the original.'); await openReceipt(currentReceipt.public_id); }
  catch (error) { toast(error.message, true); }
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
