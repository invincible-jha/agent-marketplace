/* agent-marketplace dashboard — vanilla JS + Chart.js */
'use strict';

// ---------------------------------------------------------------------------
// Tab navigation
// ---------------------------------------------------------------------------
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const target = btn.dataset.tab;
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + target).classList.add('active');
    refreshActiveTab(target);
  });
});

function refreshActiveTab(tab) {
  if (tab === 'capabilities') loadCapabilities();
  else if (tab === 'agents') loadAgents();
  else if (tab === 'analytics') loadAnalytics();
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------
function fmtTime(ts) {
  if (!ts) return '-';
  return new Date(ts * 1000).toLocaleString();
}
function escHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function trustBadge(score) {
  const pct = Math.round((Number(score) || 0.5) * 100);
  const cls = pct >= 80 ? 'badge-green' : pct >= 50 ? 'badge-yellow' : 'badge-red';
  return `<span class="badge ${cls}">${pct}%</span>`;
}

function categoryBadge(cat) {
  const colors = {
    general: 'badge-blue',
    data: 'badge-green',
    communication: 'badge-yellow',
    security: 'badge-red',
    analytics: 'badge-blue',
  };
  const cls = colors[String(cat).toLowerCase()] || 'badge-blue';
  return `<span class="badge ${cls}">${escHtml(cat)}</span>`;
}

async function apiFetch(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error('HTTP ' + res.status);
  return res.json();
}

function showToast(msg) {
  const c = document.getElementById('toast-container');
  const t = document.createElement('div');
  t.className = 'toast';
  t.textContent = msg;
  c.appendChild(t);
  setTimeout(() => t.remove(), 3000);
}

function setLastRefresh() {
  document.getElementById('last-refresh').textContent =
    'Refreshed ' + new Date().toLocaleTimeString();
}

// ---------------------------------------------------------------------------
// Capabilities Tab
// ---------------------------------------------------------------------------
async function loadCapabilities() {
  try {
    const category = document.getElementById('category-filter').value;
    const provider = document.getElementById('provider-filter').value.trim();

    let url = '/api/capabilities?limit=200';
    if (category) url += '&category=' + encodeURIComponent(category);
    if (provider) url += '&provider=' + encodeURIComponent(provider);

    const [capsData, statsData] = await Promise.all([
      apiFetch(url),
      apiFetch('/api/stats'),
    ]);

    // Stats
    document.getElementById('qs-caps').textContent = statsData.total_capabilities || 0;
    document.getElementById('qs-agents').textContent = statsData.total_agents || 0;
    document.getElementById('qs-usage').textContent = statsData.total_usage_events || 0;
    document.getElementById('qs-categories').textContent =
      Object.keys(statsData.by_category || {}).length || 0;

    // Populate category filter
    const catFilter = document.getElementById('category-filter');
    const existingCats = new Set([...catFilter.options].map(o => o.value).filter(Boolean));
    Object.keys(statsData.by_category || {}).forEach(cat => {
      if (!existingCats.has(cat)) {
        const opt = document.createElement('option');
        opt.value = cat;
        opt.textContent = cat.charAt(0).toUpperCase() + cat.slice(1);
        catFilter.appendChild(opt);
      }
    });

    const tbody = document.getElementById('capabilities-tbody');
    const caps = capsData.capabilities || [];
    if (caps.length === 0) {
      tbody.innerHTML = '<tr><td colspan="6" class="empty-state">No capabilities registered yet.</td></tr>';
      setLastRefresh();
      return;
    }
    tbody.innerHTML = caps.slice().reverse().map(c => `
      <tr>
        <td style="font-weight:600">${escHtml(c.name || '-')}</td>
        <td>${categoryBadge(c.category || 'general')}</td>
        <td>${escHtml(c.provider || '-')}</td>
        <td><span style="font-family:monospace;font-size:12px">${escHtml(c.version || '-')}</span></td>
        <td>${trustBadge(c.trust_score)}</td>
        <td>${fmtTime(c.registered_at)}</td>
      </tr>
    `).join('');
    setLastRefresh();
  } catch (err) {
    showToast('Error loading capabilities: ' + err.message);
  }
}

// ---------------------------------------------------------------------------
// Agent Cards Tab
// ---------------------------------------------------------------------------
async function loadAgents() {
  try {
    const data = await apiFetch('/api/agents?limit=100');
    const grid = document.getElementById('agents-grid');
    const agents = data.agents || [];

    if (agents.length === 0) {
      grid.innerHTML = '<div class="empty-state" style="grid-column:1/-1"><div class="icon">&#129302;</div>No agents registered yet.</div>';
      setLastRefresh();
      return;
    }

    grid.innerHTML = agents.map(agent => {
      const caps = Array.isArray(agent.capabilities) ? agent.capabilities : [];
      const capBadges = caps.slice(0, 5).map(cap =>
        `<span class="badge badge-blue">${escHtml(String(cap).slice(0, 20))}</span>`
      ).join('');
      const moreCount = Math.max(0, caps.length - 5);
      return `
        <div class="agent-card">
          <div class="agent-card-name">&#129302; ${escHtml(agent.name || 'Unnamed Agent')}</div>
          <div class="agent-card-meta">
            ID: <span style="font-family:monospace">${escHtml(String(agent.id || '-').slice(0, 12))}...</span>
          </div>
          ${agent.description ? `<div style="color:var(--text-muted);font-size:12px;margin-top:8px">${escHtml(String(agent.description).slice(0, 120))}</div>` : ''}
          <div class="agent-card-caps">
            ${capBadges}
            ${moreCount > 0 ? `<span style="color:var(--text-muted);font-size:11px">+${moreCount} more</span>` : ''}
          </div>
          <div class="agent-card-meta" style="margin-top:10px">
            Registered: ${fmtTime(agent.registered_at)}
          </div>
        </div>
      `;
    }).join('');
    setLastRefresh();
  } catch (err) {
    showToast('Error loading agents: ' + err.message);
  }
}

// ---------------------------------------------------------------------------
// Analytics Tab
// ---------------------------------------------------------------------------
let usageBarChart = null;
let categoryPieChart = null;

async function loadAnalytics() {
  try {
    const data = await apiFetch('/api/stats');
    const textColor = getComputedStyle(document.documentElement).getPropertyValue('--text').trim();
    const mutedColor = getComputedStyle(document.documentElement).getPropertyValue('--text-muted').trim();
    const surfaceColor = getComputedStyle(document.documentElement).getPropertyValue('--surface').trim();

    // Usage bar chart
    const topCaps = data.top_capabilities || [];
    const usageCtx = document.getElementById('usage-bar-chart').getContext('2d');
    if (usageBarChart) usageBarChart.destroy();
    usageBarChart = new Chart(usageCtx, {
      type: 'bar',
      data: {
        labels: topCaps.map(c => String(c.capability_id).slice(0, 16)),
        datasets: [{
          label: 'Usage Count',
          data: topCaps.map(c => c.usage_count),
          backgroundColor: 'rgba(16,185,129,0.7)',
          borderColor: '#10b981',
          borderWidth: 1,
          borderRadius: 4,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: textColor } } },
        scales: {
          x: {
            ticks: { color: mutedColor, maxRotation: 45 },
            grid: { color: 'rgba(255,255,255,0.05)' },
          },
          y: {
            ticks: { color: mutedColor },
            grid: { color: 'rgba(255,255,255,0.05)' },
          },
        },
      },
    });

    // Category pie chart
    const byCategory = data.by_category || {};
    const catLabels = Object.keys(byCategory);
    const catValues = Object.values(byCategory);
    const catColors = ['#10b981','#3b82f6','#f59e0b','#ef4444','#8b5cf6','#ec4899','#14b8a6','#f97316'];
    const pieCtx = document.getElementById('category-pie-chart').getContext('2d');
    if (categoryPieChart) categoryPieChart.destroy();
    categoryPieChart = new Chart(pieCtx, {
      type: 'pie',
      data: {
        labels: catLabels.length ? catLabels : ['No data'],
        datasets: [{
          data: catValues.length ? catValues : [1],
          backgroundColor: catColors.slice(0, Math.max(catLabels.length, 1)),
          borderColor: surfaceColor,
          borderWidth: 2,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'bottom',
            labels: { color: textColor },
          },
        },
      },
    });
    setLastRefresh();
  } catch (err) {
    showToast('Error loading analytics: ' + err.message);
  }
}

// ---------------------------------------------------------------------------
// Search Tab
// ---------------------------------------------------------------------------
document.getElementById('search-query').addEventListener('keydown', e => {
  if (e.key === 'Enter') runSearch();
});

async function runSearch() {
  const query = document.getElementById('search-query').value.trim();
  if (!query) { showToast('Enter a search query.'); return; }

  const container = document.getElementById('search-results-container');
  container.innerHTML = '<div class="empty-state">Searching...</div>';

  try {
    const data = await apiFetch('/api/capabilities/search?q=' + encodeURIComponent(query));
    const results = data.results || [];

    if (results.length === 0) {
      container.innerHTML = `<div class="empty-state"><div class="icon">&#128269;</div>No capabilities found for "${escHtml(query)}".</div>`;
      return;
    }

    container.innerHTML = `
      <div style="margin-bottom:12px;color:var(--text-muted);font-size:13px">
        Found <strong>${results.length}</strong> result(s) for "<strong>${escHtml(query)}</strong>"
      </div>
      <table>
        <thead>
          <tr><th>Name</th><th>Category</th><th>Provider</th><th>Version</th><th>Trust</th></tr>
        </thead>
        <tbody>
          ${results.map(c => `
            <tr>
              <td style="font-weight:600">${escHtml(c.name || '-')}</td>
              <td>${categoryBadge(c.category || 'general')}</td>
              <td>${escHtml(c.provider || '-')}</td>
              <td style="font-family:monospace;font-size:12px">${escHtml(c.version || '-')}</td>
              <td>${trustBadge(c.trust_score)}</td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    `;
  } catch (err) {
    showToast('Search error: ' + err.message);
  }
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
loadCapabilities();
setInterval(() => {
  const active = document.querySelector('.tab-btn.active');
  if (active) refreshActiveTab(active.dataset.tab);
}, 30000);
