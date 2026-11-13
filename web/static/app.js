const $ = id => document.getElementById(id);
const setValue = (id, value = '') => {
  const element = $(id);
  if (element) element.value = value ?? '';
  return element;
};
const fieldValue = id => String($(id)?.value ?? '').trim();
const pages = ['overview', 'scan', 'history', 'finance', 'detail'];
const dismissedCapture = new Set();
let csrf = '', user = null, authMode = 'login', selectedFile = null, cameraStream = null;
let currentReceipt = null, historyMode = 'grid', searchTimer = null;
let userSettings = {default_view: 'faithful', show_myinvois: true, open_reading_results: false, pack_images: true};


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

// ------------------------------------------------------------ dialogs
function askConfirm({title = 'Are you sure?', message = '', confirmLabel = 'Continue', danger = false} = {}) {
  return new Promise(resolve => {
    const dialog = $('confirmDialog');
    $('confirmTitle').textContent = title; $('confirmMessage').textContent = message;
    $('confirmOk').textContent = confirmLabel; $('confirmOk').className = `button ${danger ? 'danger-solid' : 'primary'}`;
    const finish = value => { dialog.close(); $('confirmOk').onclick = $('confirmCancel').onclick = dialog.oncancel = null; resolve(value); };
    $('confirmOk').onclick = () => finish(true); $('confirmCancel').onclick = () => finish(false);
    dialog.oncancel = event => { event.preventDefault(); finish(false); };
    dialog.showModal(); $('confirmCancel').focus();
  });
}

const TERMS_VERSION = '1.0';
let legalLang = 'en';
function legalHtml(lang) { return $(lang === 'ms' ? 'legalMs' : 'legalEn').innerHTML; }
function renderTerms() {
  $('termsBody').innerHTML = legalHtml(legalLang);
  document.querySelectorAll('[data-legal-lang]').forEach(b => b.classList.toggle('active', b.dataset.legalLang === legalLang));
  $('termsBody').scrollTop = 0;
}
document.querySelectorAll('[data-legal-lang]').forEach(b => b.onclick = () => { legalLang = b.dataset.legalLang; renderTerms(); });
function openLegal(mode = 'read') {
  const accept = mode === 'accept';
  $('termsEyebrow').textContent = accept ? 'BEFORE YOU START' : 'CLEARRECEIPT';
  $('termsFootAccept').classList.toggle('hidden', !accept); $('termsFootClose').classList.toggle('hidden', accept);
  $('termsCheck').checked = false; $('termsAccept').disabled = true;
  renderTerms(); if (!$('termsDialog').open) $('termsDialog').showModal();
}
$('termsDialog').addEventListener('cancel', event => { if (!$('termsFootAccept').classList.contains('hidden')) event.preventDefault(); });
$('termsClose').onclick = () => $('termsDialog').close();
$('termsCheck').onchange = () => { $('termsAccept').disabled = !$('termsCheck').checked; };
$('termsAccept').onclick = async () => {
  $('termsAccept').disabled = true;
  try {
    await api('/api/terms', {method: 'POST', body: {version: TERMS_VERSION}});
    user.terms_accepted = true; $('termsDialog').close(); showWorkspace();
  } catch (error) { toast(error.message, true); $('termsAccept').disabled = false; }
};
$('termsDecline').onclick = async () => {
  $('termsDialog').close();
  try { await api('/api/logout', {method: 'POST'}); } catch (_) {}
  sessionStorage.setItem('signedOut', '1');
  const state = await api('/api/auth'); csrf = state.csrf_token; accessState = state; showAuth();
  $('authError').textContent = 'ClearReceipt can only be used after accepting the Terms of Use and Privacy Notice.'; $('authError').classList.remove('hidden');
};
document.addEventListener('click', event => { if (event.target.closest('[data-open-legal]')) { event.preventDefault(); openLegal('read'); } });

let accessState = {};
function showAuth() {
  user = null;
  document.body.classList.remove('nav-open');
  document.querySelectorAll('dialog[open]').forEach(d => d.close());
  const open = Boolean(accessState.workspace_available);
  // Downloaded/local copy: receipts live on this computer, so lead with
  // "Open my receipts" and keep accounts behind a link for shared computers.
  $('localEntry').classList.toggle('hidden', !open);
  $('authForm').classList.toggle('local-first', open);
  $('authForm').classList.remove('show-accounts');
  setAuthMode('login');
  $('workspace').classList.add('hidden');
  $('authView').classList.remove('hidden');
}

function showWorkspace() {
  document.body.classList.remove('nav-open');
  if (!user.terms_accepted) { $('authView').classList.add('hidden'); $('workspace').classList.add('hidden'); openLegal('accept'); return; }
  $('authView').classList.add('hidden');
  $('workspace').classList.remove('hidden');
  $('profileName').textContent = user.name;
  $('profileEmail').textContent = user.is_workspace ? 'This computer · no sign-in' : user.is_guest ? 'Guest session · local device' : user.email;
  $('logoutButton').title = $('logoutButton').ariaLabel = accessState.sign_in_required ? 'Sign out' : 'Switch account';
  sessionStorage.removeItem('signedOut');
  $('avatar').textContent = user.name.trim().charAt(0).toUpperCase();
  $('welcomeName').textContent = `Ready to recover a receipt, ${user.name.split(' ')[0]}?`;
  loadUserSettings();
  if (!user.welcome_seen && !$('welcomeDialog').open) $('welcomeDialog').showModal();
  phoneWatchSeen = null; watchPhone();
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
  const localFirst = $('authForm').classList.contains('local-first') && !$('authForm').classList.contains('show-accounts');
  $('authTitle').textContent = localFirst ? 'Welcome back' : register ? 'Create your workspace' : $('authForm').classList.contains('local-first') ? 'Sign in to an account' : 'Welcome back';
  $('authSubtitle').textContent = localFirst ? 'Your receipts are saved on this computer.' : register ? 'Save, organise and review your recovered receipts.' : 'Sign in to access your saved receipts.';
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

$('accountBack').onclick = () => { $('authForm').classList.remove('show-accounts'); setAuthMode('login'); };
$('accountToggle').onclick = () => { $('authForm').classList.add('show-accounts'); setAuthMode('login'); $('authEmail').focus(); };
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
    user = data.user; csrf = data.csrf_token || csrf; showWorkspace();
  } catch (error) {
    showFieldErrors(error.details?.fields);
    $('authError').textContent = error.message; $('authError').classList.remove('hidden');
  } finally {
    button.disabled = false; button.textContent = authMode === 'login' ? 'Sign in' : 'Create account';
  }
};

$('workspaceButton').onclick = async () => {
  const button = $('workspaceButton'); button.disabled = true;
  try {
    const data = await api('/api/workspace', {method: 'POST'});
    user = data.user; sessionStorage.removeItem('lastReceiptId'); history.replaceState(null, '', '#overview'); showWorkspace();
  } catch (error) { $('authError').textContent = error.message; $('authError').classList.remove('hidden'); }
  finally { button.disabled = false; }
};

$('logoutButton').onclick = async () => {
  if (user?.is_guest && !(await askConfirm({title: 'Sign out of the guest session?', message: 'Guest receipts cannot be reopened after signing out. Export what you need first. Files remain on this computer.', confirmLabel: 'Sign out'}))) return;
  try { await api('/api/logout', {method: 'POST'}); } finally {
    selectedReceipts.clear(); sessionStorage.removeItem('lastReceiptId'); currentReceipt = null;
    sessionStorage.setItem('signedOut', '1');
    csrf = ''; const state = await api('/api/auth'); csrf = state.csrf_token; accessState = state; showAuth();
  }
};

const pageCopy = {
  overview: ['Overview', 'Your receipt recovery workspace'],
  scan: ['Scan receipt', 'Recover a faded thermal receipt'],
  history: ['Receipt history', 'Search, review and organise saved receipts'],
  finance: ['Finance reports', 'Track company expenses and spending types'],
  detail: ['Receipt details', 'Compare the image and verify extracted fields']
};

function route(name, updateHash = true) {
  if (!pages.includes(name)) name = 'overview';
  const leaving = pages.find(p => !$(`${p}Page`).classList.contains('hidden'));
  if (name === 'detail' && returnLabels[leaving]) { returnPage = leaving; $('backLabel').textContent = returnLabels[leaving]; }
  pages.forEach(p => $(`${p}Page`).classList.toggle('hidden', p !== name));
  document.querySelectorAll('[data-route]').forEach(el => el.classList.toggle('active', el.dataset.route === name));
  [$('pageTitle').textContent, $('pageSubtitle').textContent] = pageCopy[name];
  if (updateHash) history.replaceState(null, '', `#${name}`);
  document.body.classList.remove('nav-open');
  window.scrollTo({top: 0, behavior: 'smooth'});
  if (name === 'overview') loadDashboard();
  if (name === 'history') loadHistory();
  if (name === 'finance') loadFinance();
  if (name === 'scan') resetScan();
}

document.querySelectorAll('[data-route]').forEach(el => el.onclick = event => { event.preventDefault(); route(el.dataset.route); });
$('menuButton').onclick = () => document.body.classList.toggle('nav-open');
let returnPage = 'history';
const returnLabels = {overview: 'Back to overview', history: 'Back to history', finance: 'Back to finance reports', scan: 'Back to scan'};
$('backButton').onclick = () => route(returnPage);

function formatDate(value) {
  if (!value) return 'Date unavailable';
  const date = new Date(value); return Number.isNaN(date.valueOf()) ? value : date.toLocaleDateString('en-MY', {day: 'numeric', month: 'short', year: 'numeric'});
}
function statusLabel(status) { return ({needs_review: 'To verify', reviewed: 'Reviewed', archived: 'Archived'})[status] || status; }
function money(value, currency = 'MYR') {
  const number = Number(String(value ?? '').replace(/^(RM|MYR)\s*/i, '').replace(/,/g, ''));
  if (value === null || value === undefined || value === '' || Number.isNaN(number)) return value ? String(value) : '';
  const text = number.toLocaleString('en-MY', {minimumFractionDigits: 2, maximumFractionDigits: 2});
  return `${!currency || /^(MYR|RM)$/i.test(currency) ? 'RM' : currency} ${text}`;
}
function monthLabel(value) {
  const match = /^(\d{4})-(\d{2})$/.exec(value || '');
  return match ? new Date(Number(match[1]), Number(match[2]) - 1, 1).toLocaleDateString('en-MY', {month: 'short', year: 'numeric'}) : value;
}
function amountLabel(value) { if (!value) return 'Amount unavailable'; return money(value); }

function emptyState(kind) {
  const history = kind === 'history';
  return `<div class="empty-state"><span>${history ? '<svg class="ico" viewBox="0 0 24 24" aria-hidden="true"><circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/></svg>' : '<svg class="ico" viewBox="0 0 24 24" aria-hidden="true"><path d="M6 3h12v18l-3-2-3 2-3-2-3 2z"/><path d="M9 8h6M9 12h6M9 16h3"/></svg>'}</span><h3>${history ? 'No matching receipts' : 'No saved receipts yet'}</h3><p>${history ? 'Try a different search or status filter.' : 'Your scanned receipts will appear here.'}</p>${history ? '' : '<button class="button primary" data-route="scan">Scan your first receipt</button>'}</div>`;
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

let financeTypes = {};
const financeParams = () => new URLSearchParams({
  from: fieldValue('financeFrom'), to: fieldValue('financeTo'),
  category: fieldValue('financeCategory') || 'all', source: fieldValue('financeSource') || 'all'
});
function financeBars(rows, labelKey) {
  if (!rows.length) return '<p class="empty-copy">No expenses in this period.</p>';
  const maximum = Math.max(...rows.map(row => Number(row.total) || 0), 1);
  return rows.map(row => `<div class="finance-bar"><div><span>${escapeHtml(labelKey === 'month' ? monthLabel(row[labelKey]) : row[labelKey])}</span><b>${escapeHtml(money(row.total))}</b></div><i style="width:${Math.max(2, (Number(row.total) / maximum) * 100)}%"></i></div>`).join('');
}
function financeTypeOptions(selected) {
  return Object.entries(financeTypes).map(([code, label]) => `<option value="${escapeHtml(code)}" ${code === selected ? 'selected' : ''}>${escapeHtml(label)}</option>`).join('');
}
function renderFinanceTable(entries) {
  if (!entries.length) return '<div class="empty-state"><h3>No expenses found</h3><p>Change the filters, scan a receipt, or add a manual expense.</p></div>';
  return `<table class="finance-table"><thead><tr><th>Date</th><th>Supplier / description</th><th>Type</th><th>Source</th><th>Status</th><th class="number">Amount</th><th></th></tr></thead><tbody>${entries.map(item => `<tr><td class="nowrap">${escapeHtml(item.date ? formatDate(item.date) : 'Unknown')}</td><td><b>${escapeHtml(item.supplier)}</b><small>${escapeHtml(item.description)}</small></td><td><select data-finance-type="${escapeHtml(item.id)}" data-finance-source="${escapeHtml(item.source)}">${financeTypeOptions(item.expense_type)}</select></td><td>${item.source === 'receipt' ? 'Scanned receipt' : 'Manual'}</td><td><span class="status-pill ${escapeHtml(item.status)}">${escapeHtml(statusLabel(item.status))}</span></td><td class="number"><b>${item.amount ? escapeHtml(money(item.amount, item.currency)) : 'Missing'}</b></td><td>${item.source === 'receipt' ? `<button class="text-button" data-finance-receipt="${escapeHtml(item.id)}">Open</button>` : `<button class="text-button danger-text" data-delete-expense="${escapeHtml(item.id)}">Delete</button>`}</td></tr>`).join('')}</tbody></table>`;
}
async function loadFinance() {
  $('financeTable').innerHTML = '<div class="loading-state">Preparing expense report…</div>';
  try {
    const data = await api(`/api/finance?${financeParams()}`); financeTypes = data.types;
    if ($('financeCategory').options.length === 1) $('financeCategory').insertAdjacentHTML('beforeend', financeTypeOptions(''));
    if ($('expenseType').options.length === 0) $('expenseType').innerHTML = financeTypeOptions('other');
    $('financeTotal').textContent = money(data.summary.total) || 'RM 0.00'; $('financeCount').textContent = data.summary.count;
    $('financeReview').textContent = data.summary.needs_review; $('financeAverage').textContent = money(data.summary.average) || 'RM 0.00';
    $('financeByType').innerHTML = financeBars(data.by_type, 'label'); $('financeMonthly').innerHTML = financeBars(data.monthly, 'month');
    $('financeNotice').textContent = `${data.summary.missing_amounts} record(s) have no usable amount. ${data.summary.needs_review} scanned receipt(s) still need verification.`;
    $('financeTable').innerHTML = renderFinanceTable(data.entries);
    $('financeTable').querySelectorAll('[data-finance-receipt]').forEach(button => button.onclick = () => openReceipt(button.dataset.financeReceipt));
    $('financeTable').querySelectorAll('[data-finance-type]').forEach(select => select.onchange = async () => {
      try {
        const base = select.dataset.financeSource === 'receipt' ? '/api/finance/receipts/' : '/api/finance/expenses/';
        await api(`${base}${encodeURIComponent(select.dataset.financeType)}`, {method:'PATCH', body:{expense_type:select.value}}); toast('Expense type updated.'); await loadFinance();
      } catch (error) { toast(error.message, true); }
    });
    $('financeTable').querySelectorAll('[data-delete-expense]').forEach(button => button.onclick = async () => {
      if (!(await askConfirm({title: 'Delete this manual expense?', message: 'It will be removed from your finance reports.', confirmLabel: 'Delete', danger: true}))) return;
      try { await api(`/api/finance/expenses/${encodeURIComponent(button.dataset.deleteExpense)}`, {method:'DELETE'}); toast('Expense deleted.'); await loadFinance(); }
      catch (error) { toast(error.message, true); }
    });
  } catch (error) { $('financeTable').innerHTML = `<div class="error-inline">${escapeHtml(error.message)}</div>`; }
}

['financeFrom','financeTo','financeCategory','financeSource'].forEach(id => $(id).onchange = loadFinance);
$('clearFinanceFilters').onclick = () => { ['financeFrom','financeTo'].forEach(id => setValue(id)); setValue('financeCategory','all'); setValue('financeSource','all'); loadFinance(); };
$('addExpenseButton').onclick = () => { $('expenseForm').reset(); setValue('expenseDate', new Date().toISOString().slice(0,10)); setValue('expenseCurrency','MYR'); $('expenseType').innerHTML = financeTypeOptions('other'); $('expenseError').textContent = ''; $('expenseDialog').showModal(); };
$('closeExpenseButton').onclick = () => $('expenseDialog').close();
$('expenseForm').onsubmit = async event => {
  event.preventDefault(); $('expenseError').textContent = '';
  const body = {expense_date:fieldValue('expenseDate'), supplier:fieldValue('expenseSupplier'), description:fieldValue('expenseDescription'), expense_type:fieldValue('expenseType'), amount:fieldValue('expenseAmount'), currency:fieldValue('expenseCurrency'), payment_method:fieldValue('expensePayment'), tax_amount:fieldValue('expenseTax'), notes:fieldValue('expenseNotes')};
  try { await api('/api/finance/expenses', {method:'POST', body}); $('expenseDialog').close(); toast('Expense recorded.'); await loadFinance(); }
  catch (error) { $('expenseError').textContent = error.message + (error.details?.fields ? ` ${Object.values(error.details.fields).join(' ')}` : ''); }
};

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
    summary.textContent = `Available amounts: ${money(data.total) || 'RM 0.00'} (MYR assumed) · ${data.count} selected` +
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
  if (!ids.length || !(await askConfirm({title: `Delete ${ids.length} receipt${ids.length === 1 ? '' : 's'}?`, message: 'They will be removed from your history and reports. This cannot be undone from the app.', confirmLabel: 'Delete', danger: true}))) return;
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

// Fast preset: the fields arrive first and the cleaned image a few seconds later.
let cleanedImageTimer = null;
function waitForCleanedImage(id, tries = 0) {
  clearTimeout(cleanedImageTimer);
  if (tries > 40) return;
  cleanedImageTimer = setTimeout(async () => {
    if (!currentReceipt || currentReceipt.public_id !== id) return;
    try {
      const fresh = await api(`/api/receipts/${encodeURIComponent(id)}`);
      if (fresh.audit?.deferred_display?.status === 'pending') return waitForCleanedImage(id, tries + 1);
      if (currentReceipt?.public_id === id) { currentReceipt.audit = fresh.audit; $('restoredImage').src = `${currentReceipt.faithful_url}?v=${Date.now()}`; }
    } catch (_) { /* keep the interim image */ }
  }, 1500);
}

async function openReceipt(id) {
  try {
    showReceipt(await api(`/api/receipts/${encodeURIComponent(id)}`));
  } catch (error) { toast(error.message, true); }
}

// Fill the review page from a full receipt record. After a save the page is
// refreshed from the server's copy, so it always shows exactly what was stored.
function showReceipt(record, {keepView = false} = {}) {
  {
    currentReceipt = record; const id = record.public_id;
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
    if (currentReceipt.audit?.deferred_display?.status === 'pending') waitForCleanedImage(id);
    setValue('merchantField', currentReceipt.merchant); setValue('dateField', currentReceipt.transaction_date);
    setValue('numberField', currentReceipt.receipt_number); setValue('totalField', currentReceipt.total_amount);
    setValue('reviewStatus', currentReceipt.status); setValue('tagsField', (currentReceipt.tags || []).join(', ')); setValue('notesField', currentReceipt.notes);
    const accounting = currentReceipt.accounting_fields || {};
    setValue('supplierNameField', accounting.supplier_name || currentReceipt.merchant);
    $('supplierNameEvidence').textContent = accounting.brand && accounting.supplier_name && accounting.supplier_name !== currentReceipt.merchant ? `Read from the receipt: the ${accounting.brand} station is run by this business. Use this name for your records.` : '';
    const purpose = [accounting.business_type, accounting.purchase_summary && accounting.purchase_summary !== accounting.business_type ? accounting.purchase_summary : null, accounting.brand && accounting.supplier_name && accounting.supplier_name !== currentReceipt.merchant ? `issued by ${accounting.supplier_name}` : null].filter(Boolean).join(' · ');
    $('purposeEvidence').textContent = purpose ? `Recognised: ${purpose}` : ''; $('purposeEvidence').title = accounting.recognised_from ? `From: ${accounting.recognised_from}` : ''; $('purposeEvidence').classList.toggle('hidden', !purpose);
    for (const [key,id] of Object.entries({document_type:'documentTypeField',currency:'currencyField',supplier_registration_number:'supplierRegistrationField',supplier_tax_number:'supplierTaxField',supplier_address:'supplierAddressField',supplier_phone:'supplierPhoneField',supplier_email:'supplierEmailField',payment_method:'paymentMethodField',subtotal:'subtotalField',discount:'discountField',tax_amount:'taxAmountField'})) setValue(id, accounting[key]);
    renderLineItems(currentReceipt.line_items || []);
    renderReadiness(currentReceipt);
    const duplicateCount = (currentReceipt.possible_duplicates || []).length;
    $('duplicateWarning').classList.toggle('hidden', !duplicateCount);
    const capture = currentReceipt.audit?.capture_check;
    const askRetry = capture?.status === 'retry' && !currentReceipt.audit?.user_saved_at && !dismissedCapture.has(currentReceipt.public_id);
    $('captureWarning').classList.toggle('hidden', !askRetry);
    if (askRetry) {
      $('captureTitle').textContent = capture.issue === 'rotation' ? 'This receipt may be turned the wrong way' : capture.issue === 'crop' ? 'Part of this receipt may be cut off' : 'Very little could be read';
      $('captureMessage').textContent = capture.message || 'Please scan this receipt again.';
    }
    $('duplicateWarning').textContent = duplicateCount ? `Possible duplicate: ${duplicateCount} matching receipt record${duplicateCount === 1 ? '' : 's'} found. Verify before recording it again.` : '';
    const evidence = new Map((currentReceipt.audit?.fields || []).map(field => [field.name, field]));
    for (const [name, elementId] of [['merchant','merchantEvidence'],['transaction_date','dateEvidence'],['receipt_invoice_number','numberEvidence'],['total_amount','totalEvidence']]) {
      const field = evidence.get(name), confidence = field?.ocr_confidence;
      $(elementId).textContent = field?.value ? `${confidence == null ? 'Located in receipt' : `${(Number(confidence) * 100).toFixed(1)}% scanner score (not a guarantee)`} · verify against original` : (field?.suggested ? `Year partly hidden · receipt shows ${field.hint}; only ${field.suggested.slice(0,4)} fits the digits read · check and accept` : field?.hint ? `Not fully read · receipt shows ${field.hint} (year unreadable) · enter the full date` : 'Not read automatically · enter if shown');
      if (name === 'transaction_date') { const btn = $('dateSuggestButton'); btn.classList.toggle('hidden', !(field?.suggested && !$('dateField').value)); btn.dataset.value = field?.suggested || ''; btn.textContent = field?.suggested ? `Use ${field.suggested}` : ''; }
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
    if (!keepView) { setImageMode(userSettings.default_view === 'original' ? 'original' : 'faithful'); route('detail'); }
    applyDisplaySettings();
    loadMyinvois(id);
  }
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
  if ($('myinvoisChecklist')) $('myinvoisChecklist').innerHTML = `<table><thead><tr><th>Field group</th><th>Current value</th><th>Status</th></tr></thead><tbody>${rows.map(row => `<tr><td>${escapeHtml(row[0])}</td><td>${escapeHtml(row[1] || '—')}</td><td>${escapeHtml(row[2])}</td></tr>`).join('')}</tbody></table>`;
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
  if (!['faithful','original','compare'].includes(mode)) mode = 'faithful';
  const comparison = $('comparison');
  $('originalImage').src = mode === 'compare' ? (currentReceipt?.faithful_reference_url || currentReceipt?.original_url || '') : (currentReceipt?.original_url || '');
  comparison.classList.remove('improved-mode','original-mode','compare-mode');
  comparison.classList.add(`${mode === 'original' ? 'original' : mode === 'compare' ? 'compare' : 'improved'}-mode`);
  for (const [name,id] of [['faithful','faithfulViewButton'],['original','originalViewButton'],['compare','compareViewButton']]) $(''+id).classList.toggle('active', name === mode);
  if (mode === 'faithful' || mode === 'compare') $('restoredImage').src = currentReceipt?.faithful_url || currentReceipt?.restored_url || '';
  if (mode === 'compare') { setValue('compareSlider', 50); updateComparison(50); }
  const gate = currentReceipt?.audit?.faithfulness_gate;
  const audit = currentReceipt?.audit || {};
  const sourceView = Boolean(audit.release_content_fix || audit.faithful_processing || audit.native_refinement?.applied || audit.display_image === 'faithful_recovery');
  const rejected = gate?.accepted === false && !sourceView;
  $('comparisonViewLabel').textContent = mode === 'faithful' ? (rejected ? 'Original retained' : 'Faithful recovery') : mode === 'original' ? 'Original upload' : 'Original and faithful recovery';
  $('comparisonNote').textContent = mode === 'original' ? 'The original upload is preserved.' : rejected ? 'The recovery was rejected. Review the retained original.' : 'Review text, signatures, logos and faint details against the original. Unreadable characters cannot be confirmed by enhancement.';
  $('comparisonHelp').textContent = mode === 'compare' ? 'Drag the divider' : 'Select Compare for the split view';
}
$('faithfulViewButton').onclick = () => setImageMode('faithful');
$('originalViewButton').onclick = () => setImageMode('original');
$('compareViewButton').onclick = () => setImageMode('compare');

let imageZoom = 1;
function updateFullImageZoom() {
  $('fullImage').style.width = `${Math.round(imageZoom * 100)}%`;
  $('zoomLevel').textContent = `${Math.round(imageZoom * 100)}%`;
}
function openFullImage() {
  if (!currentReceipt) return;
  $('fullImage').src = $('originalViewButton').classList.contains('active') ? currentReceipt.original_url : currentReceipt.faithful_url;
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
  const accounting_fields = {...(currentReceipt.accounting_fields || {}),supplier_name:fieldValue('supplierNameField') || fieldValue('merchantField'),document_number:fieldValue('numberField'),transaction_date:fieldValue('dateField'),payable_amount:fieldValue('totalField'),document_type:fieldValue('documentTypeField'),currency:fieldValue('currencyField'),supplier_registration_number:fieldValue('supplierRegistrationField'),supplier_tax_number:fieldValue('supplierTaxField'),supplier_address:fieldValue('supplierAddressField'),supplier_phone:fieldValue('supplierPhoneField'),supplier_email:fieldValue('supplierEmailField'),payment_method:fieldValue('paymentMethodField'),subtotal:fieldValue('subtotalField'),discount:fieldValue('discountField'),tax_amount:fieldValue('taxAmountField'),line_item_count:collectLineItems().length};
  const payload = {merchant: fieldValue('merchantField'), transaction_date: fieldValue('dateField'), receipt_number: fieldValue('numberField'), total_amount: fieldValue('totalField'), status: fieldValue('reviewStatus'), tags: fieldValue('tagsField').split(',').map(v => v.trim()).filter(Boolean), notes: fieldValue('notesField'), line_items: collectLineItems(), accounting_fields};
  try {
    const saved = await api(`/api/receipts/${encodeURIComponent(currentReceipt.public_id)}`, {method: 'PATCH', body: payload});
    showReceipt(saved, {keepView: true});
    $('saveMessage').textContent = 'Saved successfully.'; toast('Receipt review saved.'); setTimeout(() => $('saveMessage').textContent = '', 2500);
  } catch (error) { $('saveMessage').textContent = [error.message, ...Object.values(error.details?.fields || {})].join(' '); }
};

$('captureRetry').onclick = async () => {
  if (!currentReceipt) return;
  const id = currentReceipt.public_id;
  if (await askConfirm({title: 'Scan this receipt again?', message: 'This removes the current scan so it is not kept twice. You can then upload a new photo, or use your phone and turn or crop it before sending.', confirmLabel: 'Remove and scan again', danger: true})) {
    try { await api(`/api/receipts/${encodeURIComponent(id)}`, {method: 'DELETE'}); } catch (error) { toast(error.message, true); return; }
    route('scan');
  }
};
$('captureKeep').onclick = () => { if (currentReceipt) dismissedCapture.add(currentReceipt.public_id); $('captureWarning').classList.add('hidden'); };

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
  if (!(await askConfirm({title: 'Read this receipt again?', message: 'This replaces the saved merchant, date, number and total with new scanner results and regenerates the recovered image. Unsaved edits will be lost.', confirmLabel: 'Read again'}))) return;
  try { await api(`/api/receipts/${encodeURIComponent(currentReceipt.public_id)}/reextract`, {method: 'POST'}); toast('Fields re-read. Verify them against the original.'); await openReceipt(currentReceipt.public_id); }
  catch (error) { toast(error.message, true); }
};

$('deleteButton').onclick = async () => {
  if (!currentReceipt || !(await askConfirm({title: 'Delete this receipt?', message: 'It will be removed from your history and reports. This cannot be undone from the app.', confirmLabel: 'Delete', danger: true}))) return;
  try { await api(`/api/receipts/${encodeURIComponent(currentReceipt.public_id)}`, {method: 'DELETE'}); toast('Receipt deleted.'); route(returnPage === 'scan' ? 'history' : returnPage); }
  catch (error) { toast(error.message, true); }
};

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
    const state = await api('/api/auth'); csrf = state.csrf_token; user = state.user; accessState = state;
    // No sign-in needed on this computer: open its receipts straight away,
    // unless the person just chose "Switch account".
    if (!user && state.workspace_available && !sessionStorage.getItem('signedOut')) {
      const data = await api('/api/workspace', {method: 'POST'}); user = data.user;
    }
    if (user) showWorkspace(); else showAuth();
  } catch (_) { showAuth(); $('authError').textContent = 'We could not connect just now. Please refresh the page and try again.'; $('authError').classList.remove('hidden'); }
})();


// ---------------------------------------------------------------- MyInvois
let myinvois = null;   // {draft, validation, schema}
const SOURCE_LABEL = {receipt: 'From receipt', profile: 'Saved profile', default: 'Rule default', consolidated_rule: 'Consolidated rule', user: 'Entered'};
const getPath = (object, path) => path.split('.').reduce((node, key) => (node == null ? undefined : node[key]), object);
function setPath(object, path, value) {
  const parts = path.split('.'); let node = object;
  parts.slice(0, -1).forEach(key => { if (node[key] == null || typeof node[key] !== 'object') node[key] = {}; node = node[key]; });
  node[parts.at(-1)] = value;
}

async function loadMyinvois(id) {
  $('purchaseStatus').textContent = 'Checking'; $('purchaseStatus').className = 'status-pill';
  try { myinvois = await api(`/api/receipts/${encodeURIComponent(id)}/myinvois`); renderPurchase(); }
  catch (error) { $('purchasePanel').textContent = error.message; }
}

function issueMap(validation) {
  const map = new Map();
  for (const issue of validation.issues || []) if (!map.has(issue.path)) map.set(issue.path, issue);
  return map;
}

function sourceBadge(path, value) {
  const origin = myinvois.draft.provenance?.[path];
  if (value === null || value === undefined || String(value).trim() === '') return '<span class="src src-missing">Needed</span>';
  return origin ? `<span class="src src-${escapeHtml(origin)}">${escapeHtml(SOURCE_LABEL[origin] || origin)}</span>` : '<span class="src src-user">Entered</span>';
}

// What an expense record needs: who was paid, when, which document, how much.
const PURCHASE_FIELDS = [
  ['supplier_name', 'Supplier'], ['transaction_date', 'Date'], ['receipt_number', 'Receipt no.'],
  ['total', 'Total (RM)'], ['tax', 'Tax (RM)', true], ['payment_mode', 'Payment', true],
];

function renderPurchase() {
  const p = myinvois.purchase;
  if (!p) { $('purchasePanel').innerHTML = ''; return; }
  const tone = p.status === 'supplier_must_issue' ? 'warn' : (p.status === 'validated_einvoice' ? 'ready' : 'ready');
  $('purchaseStatus').textContent = p.label; $('purchaseStatus').className = `status-pill ${tone}`;
  const req = p.request || {};
  let request = '';
  if (p.status !== 'validated_einvoice') {
    if (req.expired) request = `<p class="purchase-request">The window to ask the supplier for an individual e-invoice has passed${req.rule ? ` (${escapeHtml(req.rule)})` : ''}; the receipt remains your supporting document.</p>`;
    else if (req.deadline && (req.printed_on_receipt || p.status === 'supplier_must_issue')) request = `<p class="purchase-request">Want an e-invoice in your business name? Ask the supplier ${req.printed_on_receipt ? 'as printed on the receipt ' : ''}by <b>${escapeHtml(req.deadline)}</b>${req.rule ? ` (${escapeHtml(req.rule)})` : ''}.</p>`;
  }
  const rows = PURCHASE_FIELDS.filter(([key, , optional]) => !optional || p.essentials?.[key]).map(([key, label]) => {
    const value = p.essentials?.[key];
    return `<div class="purchase-field"><span>${escapeHtml(label)}</span><b>${value ? escapeHtml(value) : '<em>not read — check image</em>'}</b></div>`;
  }).join('');
  const review = (p.needs_review || []).length ? `<p class="purchase-review">Check before use: ${escapeHtml(p.needs_review.join(', '))} could not be read. Correct it in the review form above.</p>` : '';
  $('purchasePanel').innerHTML = `<p class="purchase-action">${escapeHtml(p.action)}</p>${request}${review}<div class="purchase-grid">${rows}</div><p class="purchase-note">For a batch of receipts, use <b>Accountant pack</b> on the history page: a spreadsheet plus every receipt image, no typing needed.</p>`;
}

// The buyer-side workflow only needs the purchase summary above; the
// self-billed e-invoice draft editor was removed from the interface.



// ---------------------------------------------------------------- settings
let settingsSchema = [], settingsPresets = {}, settingsDraft = {}, settingsGroup = 'Processing';

async function loadUserSettings() {
  try {
    const data = await api('/api/settings');
    userSettings = data.values; settingsSchema = data.schema; settingsPresets = data.presets || {};
    applyDisplaySettings();
  } catch (error) { /* defaults stay */ }
}

function applyDisplaySettings() {
  if ($('myinvoisCard')) $('myinvoisCard').classList.toggle('hidden', userSettings.show_myinvois === false);
  if ($('ocrDetails')) $('ocrDetails').open = Boolean(userSettings.open_reading_results);
  if ($('accountantPackLink')) {
    const month = $('packMonth')?.value;
    const base = userSettings.pack_images === false ? '/api/accountant-pack.xlsx' : '/api/accountant-pack.zip';
    $('accountantPackLink').href = month ? `${base}?month=${encodeURIComponent(month)}` : base;
    $('accountantPackLink').textContent = userSettings.pack_images === false ? 'Accountant pack (Excel)' : 'Accountant pack';
  }
}
if ($('packMonth')) $('packMonth').onchange = applyDisplaySettings;

let accessDraft = null, accessInfo = null;
function renderAccessPane() {
  const on = accessDraft.require_sign_in, needsCredentials = on && !accessInfo.has_password && !accessInfo.sign_in_required;
  $('settingsPane').innerHTML = `<h3>Sign-in</h3>
    <label class="settings-row settings-check"><input type="checkbox" id="requireSignIn" ${on ? 'checked' : ''}><span><span class="settings-label">Require sign-in on this computer</span><small>Off: the app opens straight to your receipts, like other desktop software. Receipts never leave this computer. Turn on only if several people share this computer and should not see each other's receipts.</small></span></label>
    ${needsCredentials ? `<div class="settings-credentials"><p>Set an email and password for these receipts, so they stay yours once sign-in is required.</p>
      <label>Name<input id="accessName" autocomplete="name" value="${escapeHtml(accessDraft.name || '')}"><small class="field-error" id="accessNameError"></small></label>
      <label>Email<input id="accessEmail" type="email" autocomplete="email" value="${escapeHtml(accessDraft.email || '')}"><small class="field-error" id="accessEmailError"></small></label>
      <label>Password<input id="accessPassword" type="password" autocomplete="new-password"><small class="field-error" id="accessPasswordError"></small></label>
      <label>Confirm password<input id="accessConfirmation" type="password" autocomplete="new-password"><small class="field-error" id="accessConfirmationError"></small></label></div>` : ''}
    ${!on && accessInfo.sign_in_required ? '<p class="settings-note">Turning this off makes your account open without a password on this computer.</p>' : ''}`;
  $('requireSignIn').onchange = event => { accessDraft.require_sign_in = event.target.checked; renderAccessPane(); };
  for (const [id, key] of [['accessName', 'name'], ['accessEmail', 'email'], ['accessPassword', 'password'], ['accessConfirmation', 'password_confirmation']]) {
    if ($(id)) $(id).oninput = event => { accessDraft[key] = event.target.value; };
  }
}

function renderSettings() {
  const groups = [...new Set(settingsSchema.map(item => item.group))];
  if (accessInfo && !user?.is_guest) groups.push('Sign-in');
  groups.push('About & legal');
  if (!groups.includes(settingsGroup)) settingsGroup = groups[0];
  if (settingsGroup === 'About & legal') {
    $('settingsGroups').innerHTML = groups.map(group => `<button type="button" class="${group === settingsGroup ? 'active' : ''}" data-settings-group="${escapeHtml(group)}">${escapeHtml(group)}</button>`).join('');
    $('settingsPane').innerHTML = `<h3>About ClearReceipt</h3><p class="about-note">Version 1.0.0 · ClearReceipt is the display name of the DSDNet-Receipt system. It recovers faded thermal receipts and keeps the original as evidence, running entirely on this computer.</p><p class="about-note">Support: <a href="mailto:clearreceipt.support@gmail.com">clearreceipt.support@gmail.com</a></p>
      <div class="legal-links"><button type="button" class="button secondary compact" data-open-legal>Terms of Use &amp; Privacy Notice</button></div>
      <p class="about-note">Not affiliated with or endorsed by LHDN or MyInvois. Extracted values are suggestions: check them against the original receipt before use.</p>
      ${$('legalOss').innerHTML}`;
    document.querySelectorAll('[data-settings-group]').forEach(button => button.onclick = () => { settingsGroup = button.dataset.settingsGroup; renderSettings(); });
    return;
  }
  if (settingsGroup === 'Sign-in') {
    $('settingsGroups').innerHTML = groups.map(group => `<button type="button" class="${group === settingsGroup ? 'active' : ''}" data-settings-group="${escapeHtml(group)}">${escapeHtml(group)}</button>`).join('');
    renderAccessPane();
    document.querySelectorAll('[data-settings-group]').forEach(button => button.onclick = () => { settingsGroup = button.dataset.settingsGroup; renderSettings(); });
    return;
  }
  $('settingsGroups').innerHTML = groups.map(group => `<button type="button" class="${group === settingsGroup ? 'active' : ''}" data-settings-group="${escapeHtml(group)}">${escapeHtml(group)}</button>`).join('');
  const presets = settingsGroup === 'Processing' ? `<div class="settings-presets"><span>Quick choice</span>${['fast', 'balanced', 'thorough'].map(name => `<button type="button" class="button secondary compact" data-preset="${name}">${name[0].toUpperCase() + name.slice(1)}</button>`).join('')}<small>Fast skips the optional steps marked “slower”. Restoration and the two main scanner passes always run.</small></div>` : '';
  $('settingsPane').innerHTML = `<h3>${escapeHtml(settingsGroup)}</h3>${presets}` + settingsSchema.filter(item => item.group === settingsGroup).map(item => {
    const value = settingsDraft[item.key];
    const slow = item.slow ? ' <span class="settings-slow">slower</span>' : '';
    if (item.choices) return `<label class="settings-row"><span class="settings-label">${escapeHtml(item.label)}${slow}</span><select data-setting="${item.key}">${Object.entries(item.choices).map(([code, label]) => `<option value="${escapeHtml(code)}" ${value === code ? 'selected' : ''}>${escapeHtml(label)}</option>`).join('')}</select>${item.description ? `<small>${escapeHtml(item.description)}</small>` : ''}</label>`;
    return `<label class="settings-row settings-check"><input type="checkbox" data-setting="${item.key}" ${value ? 'checked' : ''}><span><span class="settings-label">${escapeHtml(item.label)}${slow}</span>${item.description ? `<small>${escapeHtml(item.description)}</small>` : ''}</span></label>`;
  }).join('');
  document.querySelectorAll('[data-settings-group]').forEach(button => button.onclick = () => { settingsGroup = button.dataset.settingsGroup; renderSettings(); });
  document.querySelectorAll('[data-setting]').forEach(control => control.onchange = () => {
    settingsDraft[control.dataset.setting] = control.type === 'checkbox' ? control.checked : control.value;
  });
  document.querySelectorAll('[data-preset]').forEach(button => button.onclick = () => {
    Object.assign(settingsDraft, settingsPresets[button.dataset.preset] || {}); renderSettings();
    $('settingsMessage').textContent = `${button.textContent} settings selected. Press OK to save.`;
  });
}

async function openSettings() {
  await loadUserSettings();
  try { accessInfo = await api('/api/access'); accessDraft = {require_sign_in: accessInfo.sign_in_required, name: accessInfo.name === 'This computer' ? '' : accessInfo.name, email: accessInfo.email}; }
  catch { accessInfo = null; }
  settingsDraft = {...userSettings}; $('settingsMessage').textContent = '';
  renderSettings(); $('settingsDialog').showModal();
}
async function saveSettings() {
  try {
    if (accessInfo && accessDraft && accessDraft.require_sign_in !== accessInfo.sign_in_required) {
      try {
        const result = await api('/api/access', {method: 'PATCH', body: accessDraft});
        accessState = {...accessState, ...result};
        if (result.user) { user = result.user; $('profileName').textContent = user.name; $('profileEmail').textContent = user.is_workspace ? 'This computer · no sign-in' : user.email; }
        $('logoutButton').title = result.sign_in_required ? 'Sign out' : 'Switch account';
        toast(result.sign_in_required ? 'Sign-in is now required on this computer.' : 'This computer now opens without sign-in.');
      } catch (error) {
        settingsGroup = 'Sign-in'; renderSettings();
        Object.entries(error.details?.fields || {}).forEach(([key, message]) => { const node = $(`access${key[0].toUpperCase()}${key.slice(1)}Error`); if (node) node.textContent = message; });
        $('settingsMessage').textContent = error.message; return;
      }
    }
    const data = await api('/api/settings', {method: 'PATCH', body: settingsDraft});
    userSettings = data.values; applyDisplaySettings(); $('settingsDialog').close(); toast('Settings saved. They apply to the next receipt you process.');
  } catch (error) { $('settingsMessage').textContent = error.message; }
}
$('settingsNav').onclick = openSettings;
$('settingsOk').onclick = saveSettings;
$('settingsCancel').onclick = () => $('settingsDialog').close();
$('closeSettingsButton').onclick = () => $('settingsDialog').close();

// ---------------------------------------------------------------- reading copy
function boxBounds(box) {
  if (!Array.isArray(box) || !box.length) return null;
  const xs = box.map(p => Number(p[0])), ys = box.map(p => Number(p[1]));
  return {x0: Math.min(...xs), x1: Math.max(...xs), y0: Math.min(...ys), y1: Math.max(...ys)};
}

async function renderReadingCopy() {
  if (!currentReceipt) return;
  $('readingCopyBody').innerHTML = '<p class="empty-copy">Preparing…</p>';
  let lines = [];
  try { lines = (await api(`/api/receipts/${encodeURIComponent(currentReceipt.public_id)}/reading-copy`)).lines || []; } catch { lines = []; }
  const items = lines.map(line => ({text: line.text, b: {x0: line.x0, y0: line.y0, x1: line.x1, y1: line.y1}})).filter(item => String(item.text).trim());
  const width = Math.max(1, ...items.map(item => item.b.x1)), left = Math.min(0, ...items.map(item => item.b.x0));
  const heights = items.map(item => item.b.y1 - item.b.y0).sort((a, b) => a - b);
  const lineHeight = heights.length ? heights[Math.floor(heights.length / 2)] : 12;
  items.sort((a, b) => (a.b.y0 + a.b.y1) - (b.b.y0 + b.b.y1));
  const rows = [];
  for (const item of items) {
    const centre = (item.b.y0 + item.b.y1) / 2;
    const row = rows.find(r => Math.abs(r.centre - centre) < 0.55 * lineHeight);
    if (row) row.items.push(item); else rows.push({centre, items: [item]});
  }
  const field = (label, value) => `<div><span>${label}</span><b>${escapeHtml(value || '—')}</b></div>`;
  const summary = `<div class="reading-copy-summary">${field('Supplier', $('supplierNameField').value || $('merchantField').value)}${field('For', currentReceipt?.accounting_fields?.purchase_summary || '')}${field('Date', $('dateField').value)}${field('Receipt no.', $('numberField').value)}${field('Total', $('totalField').value)}</div>`;
  const body = rows.map(row => {
    row.items.sort((a, b) => a.b.x0 - b.b.x0);
    const only = row.items.length === 1 ? row.items[0] : null;
    if (only && Math.abs(((only.b.x0 + only.b.x1) / 2 - left) / (width - left) - 0.5) < 0.08) return `<div class="rc-line rc-centre">${escapeHtml(only.text)}</div>`;
    // lay the row out on a fixed character grid so pieces never print over each other
    const columns = 44; let text = '';
    for (const item of row.items) {
      const column = Math.round(columns * (item.b.x0 - left) / (width - left));
      text += (text ? ' ' : '') + ' '.repeat(Math.max(0, column - text.length - (text ? 1 : 0))) + item.text;
    }
    return `<div class="rc-line">${escapeHtml(text)}</div>`;
  }).join('');
  $('readingCopyBody').innerHTML = summary + (body || '<p class="empty-copy">No text was recognised on this receipt.</p>');
}
$('dateSuggestButton').onclick = () => { const b = $('dateSuggestButton'); if (b.dataset.value) { $('dateField').value = b.dataset.value; $('dateField').dispatchEvent(new Event('input', {bubbles: true})); b.classList.add('hidden'); } };
$('readingCopyButton').onclick = () => { $('readingCopyDialog').showModal(); renderReadingCopy(); };
$('closeReadingCopyButton').onclick = () => $('readingCopyDialog').close();


// ---------------------------------------------------------------- first run
async function closeWelcome(next) {
  $('welcomeDialog').close(); if (user) user.welcome_seen = true;
  try { await api('/api/welcome', {method: 'POST'}); } catch (_) { /* shown again next time */ }
  if (next === 'scan') route('scan'); else if (next === 'phone') openPhoneDialog();
}
$('welcomeScan').onclick = () => closeWelcome('scan');
$('welcomePhone').onclick = () => closeWelcome('phone');
$('welcomeSkip').onclick = () => closeWelcome();
$('welcomeDialog').addEventListener('cancel', () => closeWelcome());

// ---------------------------------------------------------------- phone scanner
let phonePoll = null, phoneSeen = 0, phoneWatch = null, phoneWatchSeen = null;
// While a phone is paired, keep an eye on it from any page: show when a photo
// is being read and announce each receipt as it arrives.
function watchPhone() {
  clearInterval(phoneWatch);
  phoneWatch = setInterval(async () => {
    if (!user) { clearInterval(phoneWatch); return; }
    let pairing = null;
    try { pairing = (await api('/api/phone/pair')).pairing; } catch (_) { return; }
    const chip = $('phoneChip');
    if (!pairing) { chip.classList.add('hidden'); clearInterval(phoneWatch); phoneWatchSeen = null; return; }
    const received = pairing.received || [];
    if (phoneWatchSeen === null) phoneWatchSeen = received.length;
    chip.classList.remove('hidden');
    chip.classList.toggle('busy', pairing.processing > 0);
    $('phoneChipText').textContent = pairing.processing > 0
      ? `Reading ${pairing.processing > 1 ? pairing.processing + ' receipts' : 'a receipt'} from your phone…`
      : `Phone connected · ${received.length} received`;
    if (received.length > phoneWatchSeen) {
      const fresh = received.slice(phoneWatchSeen); phoneWatchSeen = received.length;
      const last = fresh[fresh.length - 1];
      toast(`New receipt from your phone: ${last.merchant || 'Receipt'}${last.total ? ' · RM ' + last.total : ''}`);
      chip.dataset.last = last.public_id || '';
      if (typeof loadDashboard === 'function' && location.hash.slice(1) === 'overview') loadDashboard().catch(() => {});
    }
  }, 2500);
}

function renderPhoneStatus(pairing) {
  if (!pairing) { $('phoneExpiry').textContent = 'Pairing ended. Press Scan with phone again for a new code.'; $('phoneQr').classList.add('ended'); return; }
  const minutes = Math.ceil(pairing.expires_in / 60);
  $('phoneExpiry').textContent = `Code works for ${minutes} more minute${minutes === 1 ? '' : 's'}${pairing.processing ? ' · reading a photo now…' : ''}${pairing.failed ? ` · ${pairing.failed} photo(s) could not be read` : ''}.`;
  const received = pairing.received || [];
  $('phoneReceived').innerHTML = received.length ? received.slice().reverse().map(item => `<li><button type="button" class="link-button" data-open-receipt="${escapeHtml(item.public_id)}">${escapeHtml(item.merchant || 'Receipt')}</button><span>${escapeHtml([item.total ? 'RM ' + item.total : '', item.date || ''].filter(Boolean).join(' · '))}</span></li>`).join('') : '<li class="phone-small">Nothing yet.</li>';
  document.querySelectorAll('[data-open-receipt]').forEach(button => button.onclick = () => { $('phoneDialog').close(); openReceipt(button.dataset.openReceipt); });
  if (received.length > phoneSeen) { phoneSeen = received.length; if (typeof loadDashboard === 'function') loadDashboard().catch(() => {}); }
}
async function openPhoneDialog() {
  $('phoneQr').classList.remove('ended'); $('phoneQr').textContent = 'Preparing…'; $('phoneAddress').textContent = ''; $('phoneExpiry').textContent = '';
  $('phoneReceived').innerHTML = '<li class="phone-small">Nothing yet.</li>'; phoneSeen = 0;
  if (!$('phoneDialog').open) $('phoneDialog').showModal();
  try {
    const data = await api('/api/phone/pair', {method: 'POST'});
    $('phoneQr').innerHTML = data.qr_svg;
    $('phoneAddress').textContent = `This computer on the Wi-Fi: ${data.address}` + (data.other_addresses?.length ? ` (if the phone cannot connect, this computer may be on another network: ${data.other_addresses.join(', ')})` : '');
    renderPhoneStatus(data);
    phoneWatchSeen = (data.received || []).length; watchPhone();
    clearInterval(phonePoll);
    phonePoll = setInterval(async () => {
      if (!$('phoneDialog').open) { clearInterval(phonePoll); return; }
      try { renderPhoneStatus((await api('/api/phone/pair')).pairing); } catch (_) { /* keep last view */ }
    }, 3000);
  } catch (error) { $('phoneQr').textContent = error.message; }
}
document.addEventListener('click', event => { const trigger = event.target.closest('[data-phone-scan]'); if (trigger) openPhoneDialog(); });
$('closePhoneButton').onclick = $('donePhoneButton').onclick = () => $('phoneDialog').close();
$('endPhoneButton').onclick = async () => { try { await api('/api/phone/pair', {method: 'DELETE'}); } finally { clearInterval(phonePoll); clearInterval(phoneWatch); $('phoneChip').classList.add('hidden'); renderPhoneStatus(null); $('phoneQr').textContent = 'Pairing ended.'; } };
$('phoneChip').onclick = () => { const last = $('phoneChip').dataset.last; if (last && !$('phoneChip').classList.contains('busy')) openReceipt(last); else openPhoneDialog(); };
$('phoneDialog').addEventListener('close', () => clearInterval(phonePoll));
