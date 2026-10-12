/**
 * SegmentAI Pro - Frontend Application Logic
 * Coordinates REST API interactions, interactive Plotly 3D/2D projections,
 * Chart.js radar & Pareto curves, customer prediction simulator, and ROI calculator.
 */

// Global State
const state = {
  activeTab: 'dashboard',
  currentDataset: 'online_retail_real',
  currentAlgorithm: 'kmeans',
  k: 4,
  scaling: 'standard',
  eps: 1.5,
  minSamples: 10,
  pcaView: '3d', // '3d' or '2d'
  clusteringData: null,
  radarChartInstance: null,
  paretoChartInstance: null,
  kEvalChartInstance: null,
  simRadarInstance: null,
};

// Color Palette for Segments
const CLUSTER_COLORS = [
  '#06b6d4', // Cyan
  '#6366f1', // Indigo
  '#10b981', // Emerald
  '#f59e0b', // Amber
  '#a855f7', // Purple
  '#f43f5e', // Rose
  '#ec4899', // Pink
  '#3b82f6', // Blue
  '#64748b', // Slate (Noise/Outliers)
];

// Initialize on DOM Ready
document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initControls();
  initUploader();
  initSimulator();
  initRoiCalculator();
  fetchDatasets().then(() => {
    executeClustering();
  });
});

// ==========================================
// Tab Management
// ==========================================
function initTabs() {
  const tabs = document.querySelectorAll('.nav-tab');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      const target = tab.getAttribute('data-tab');
      state.activeTab = target;

      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
      const activePane = document.getElementById(`tab-${target}`);
      if (activePane) activePane.classList.add('active');

      // Trigger redraws if switching to visual tabs
      if (target === 'dashboard' && state.clusteringData) {
        setTimeout(renderPcaPlot, 50);
      }
      if (target === 'workbench' && state.clusteringData) {
        setTimeout(renderKEvalCurve, 50);
      }
    });
  });
}

// ==========================================
// Control Bar Listeners
// ==========================================
function initControls() {
  const datasetSelect = document.getElementById('dataset-select');
  const algoSelect = document.getElementById('algorithm-select');
  const kSlider = document.getElementById('k-slider');
  const kBadge = document.getElementById('k-badge');
  const scalingSelect = document.getElementById('scaling-select');
  const btnRun = document.getElementById('btn-run-clustering');
  const btnExport = document.getElementById('btn-export-csv');

  if (datasetSelect) {
    datasetSelect.addEventListener('change', (e) => {
      state.currentDataset = e.target.value;
      updateControlVisibility();
      executeClustering();
    });
  }

  if (algoSelect) {
    algoSelect.addEventListener('change', (e) => {
      state.currentAlgorithm = e.target.value;
      updateControlVisibility();
    });
  }

  if (kSlider) {
    kSlider.addEventListener('input', (e) => {
      state.k = parseInt(e.target.value, 10);
      if (kBadge) kBadge.textContent = state.k;
    });
  }

  if (scalingSelect) {
    scalingSelect.addEventListener('change', (e) => {
      state.scaling = e.target.value;
    });
  }

  if (btnRun) {
    btnRun.addEventListener('click', () => {
      executeClustering();
    });
  }

  if (btnExport) {
    btnExport.addEventListener('click', () => {
      window.location.href = '/api/download-export';
    });
  }

  // PCA 2D/3D toggle buttons
  const btnPca3d = document.getElementById('btn-pca-3d');
  const btnPca2d = document.getElementById('btn-pca-2d');
  if (btnPca3d && btnPca2d) {
    btnPca3d.addEventListener('click', () => {
      btnPca3d.classList.add('active');
      btnPca2d.classList.remove('active');
      state.pcaView = '3d';
      renderPcaPlot();
    });
    btnPca2d.addEventListener('click', () => {
      btnPca2d.classList.add('active');
      btnPca3d.classList.remove('active');
      state.pcaView = '2d';
      renderPcaPlot();
    });
  }
}

function updateControlVisibility() {
  const kGroup = document.getElementById('k-control-group');
  const dbscanGroup = document.getElementById('dbscan-control-group');
  if (state.currentAlgorithm === 'dbscan') {
    if (kGroup) kGroup.classList.add('hidden');
    if (dbscanGroup) dbscanGroup.classList.remove('hidden');
  } else {
    if (kGroup) kGroup.classList.remove('hidden');
    if (dbscanGroup) dbscanGroup.classList.add('hidden');
  }
}

// ==========================================
// Datasets Ingestion
// ==========================================
async function fetchDatasets() {
  try {
    const res = await fetch('/api/datasets');
    const data = await res.json();
    const select = document.getElementById('dataset-select');
    if (select && data.datasets) {
      select.innerHTML = '';
      data.datasets.forEach(ds => {
        const opt = document.createElement('option');
        opt.value = ds.id;
        opt.textContent = `${ds.name} (${ds.customer_count.toLocaleString()} customers)`;
        if (ds.id === state.currentDataset) opt.selected = true;
        select.appendChild(opt);
      });
    }
  } catch (err) {
    console.error('Failed to fetch datasets:', err);
  }
}

// ==========================================
// Run Clustering Pipeline
// ==========================================
async function executeClustering() {
  const btnRun = document.getElementById('btn-run-clustering');
  const spinner = document.getElementById('run-spinner');
  const btnText = document.getElementById('btn-run-text');

  if (btnRun) btnRun.disabled = true;
  if (spinner) spinner.classList.remove('hidden');
  if (btnText) btnText.textContent = 'Clustering...';

  try {
    const payload = {
      dataset_id: state.currentDataset,
      algorithm: state.currentAlgorithm,
      k: state.k,
      scaling: state.scaling,
      eps: parseFloat(document.getElementById('dbscan-eps')?.value || 1.5),
      min_samples: parseInt(document.getElementById('dbscan-min-samples')?.value || 10, 10),
    };

    const res = await fetch('/api/run-clustering', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Clustering request failed');
    }

    const data = await res.json();
    state.clusteringData = data;

    // Render components
    renderKpiRibbon(data);
    renderPcaPlot();
    renderRadarChart(data);
    renderSegmentCards(data);
    renderWorkbenchSummary(data);
    renderParetoChart(data);
    renderKEvalCurve();
    updateSimulatorDefaults(data);

  } catch (err) {
    console.error('Clustering error:', err);
    alert('Clustering Execution Error: ' + err.message);
  } finally {
    if (btnRun) btnRun.disabled = false;
    if (spinner) spinner.classList.add('hidden');
    if (btnText) btnText.textContent = 'Run Segmentation';
  }
}

// ==========================================
// Render KPI Ribbon
// ==========================================
function renderKpiRibbon(data) {
  const totalCustomers = data.segments.reduce((acc, s) => acc + s.customer_count, 0);
  const totalRev = data.segments.reduce((acc, s) => acc + (s.total_revenue || 0), 0);
  const totalClv = data.segments.reduce((acc, s) => acc + (s.total_clv || 0), 0);
  const currency = data.dataset_info.currency || '£';

  setElText('kpi-active-customers', totalCustomers.toLocaleString());
  setElText('kpi-dataset-meta', `${data.dataset_info.clean_count?.toLocaleString() || totalCustomers.toLocaleString()} transactions`);

  setElText('kpi-silhouette', data.metrics.silhouette_score.toFixed(3));
  setElText('kpi-silhouette-sub', `Davies-Bouldin: ${data.metrics.davies_bouldin_index.toFixed(3)}`);

  setElText('kpi-ari-agreement', `${(data.metrics.ari_vs_kmeans * 100).toFixed(1)}%`);
  setElText('kpi-ari-sub', `Cross-validation vs K-Means`);

  if (totalRev > 0) {
    setElText('kpi-total-revenue', `${currency}${(totalRev / 1e6).toFixed(2)}M`);
    setElText('kpi-revenue-sub', `Projected CLV: ${currency}${(totalClv / 1e6).toFixed(2)}M`);
  } else {
    setElText('kpi-total-revenue', `${data.segments.length} Segments`);
    setElText('kpi-revenue-sub', `${data.pca_variance.total_explained}% PCA Variance`);
  }

  // Active status badge in header
  const statusBadge = document.getElementById('header-status-badge');
  if (statusBadge) {
    statusBadge.textContent = `${data.algorithm.toUpperCase()} | k=${data.chosen_k} | Sil: ${data.metrics.silhouette_score}`;
  }
}

// ==========================================
// Render 2D / 3D PCA Scatter Plot (Plotly)
// ==========================================
function renderPcaPlot() {
  const container = document.getElementById('pca-plot-container');
  if (!container || !state.clusteringData) return;

  const points = state.clusteringData.scatter_points;
  const is3D = (state.pcaView === '3d');
  const clusters = [...new Set(points.map(p => p.cluster))].sort((a, b) => a - b);

  const traces = clusters.map((cid, i) => {
    const clusterPts = points.filter(p => p.cluster === cid);
    const color = (cid === -1) ? '#64748b' : CLUSTER_COLORS[i % CLUSTER_COLORS.length];
    const name = clusterPts[0]?.cluster_name || `Cluster ${cid}`;

    const hoverTexts = clusterPts.map(p => {
      let t = `<b>${name}</b><br>Customer ID: ${p.customer_id}<br>`;
      if (p.recency !== undefined) {
        t += `Recency: ${p.recency}d<br>Orders: ${p.frequency}<br>Spend: £${p.monetary}`;
      } else if (p.income !== undefined) {
        t += `Income: $${p.income}k<br>Spending Score: ${p.spending}/100`;
      }
      return t;
    });

    if (is3D) {
      return {
        x: clusterPts.map(p => p.x),
        y: clusterPts.map(p => p.y),
        z: clusterPts.map(p => p.z),
        mode: 'markers',
        type: 'scatter3d',
        name: name,
        text: hoverTexts,
        hoverinfo: 'text',
        marker: {
          size: 4.5,
          color: color,
          opacity: 0.85,
        }
      };
    } else {
      return {
        x: clusterPts.map(p => p.x),
        y: clusterPts.map(p => p.y),
        mode: 'markers',
        type: 'scatter',
        name: name,
        text: hoverTexts,
        hoverinfo: 'text',
        marker: {
          size: 6.5,
          color: color,
          opacity: 0.85,
        }
      };
    }
  });

  const varInfo = state.clusteringData.pca_variance;
  const layout = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    margin: { l: 20, r: 20, t: 20, b: 20 },
    font: { family: 'Inter, sans-serif', color: '#94a3b8' },
    showlegend: true,
    legend: {
      orientation: 'h',
      y: -0.1,
      font: { size: 11, color: '#f1f5f9' },
    },
    autosize: true,
  };

  if (is3D) {
    layout.scene = {
      xaxis: { title: `PC1 (${varInfo.pc1}%)`, gridcolor: 'rgba(255,255,255,0.08)', zerolinecolor: 'rgba(255,255,255,0.15)' },
      yaxis: { title: `PC2 (${varInfo.pc2}%)`, gridcolor: 'rgba(255,255,255,0.08)', zerolinecolor: 'rgba(255,255,255,0.15)' },
      zaxis: { title: `PC3 (${varInfo.pc3}%)`, gridcolor: 'rgba(255,255,255,0.08)', zerolinecolor: 'rgba(255,255,255,0.15)' },
      bgcolor: 'rgba(0,0,0,0)',
      camera: { eye: { x: 1.4, y: 1.4, z: 1.1 } },
    };
  } else {
    layout.xaxis = { title: `Principal Component 1 (${varInfo.pc1}% variance)`, gridcolor: 'rgba(255,255,255,0.08)', zerolinecolor: 'rgba(255,255,255,0.15)' };
    layout.yaxis = { title: `Principal Component 2 (${varInfo.pc2}% variance)`, gridcolor: 'rgba(255,255,255,0.08)', zerolinecolor: 'rgba(255,255,255,0.15)' };
  }

  const config = { responsive: true, displayModeBar: false };
  Plotly.react('pca-plot-container', traces, layout, config);
}

// ==========================================
// Render Radar Chart (Chart.js)
// ==========================================
function renderRadarChart(data) {
  const ctx = document.getElementById('radar-canvas')?.getContext('2d');
  if (!ctx) return;

  if (state.radarChartInstance) {
    state.radarChartInstance.destroy();
  }

  const datasets = data.radar_data.map((seg, i) => {
    const color = CLUSTER_COLORS[i % CLUSTER_COLORS.length];
    return {
      label: seg.name,
      data: seg.values,
      borderColor: color,
      backgroundColor: hexToRgba(color, 0.15),
      borderWidth: 2,
      pointBackgroundColor: color,
      pointRadius: 3,
    };
  });

  state.radarChartInstance = new Chart(ctx, {
    type: 'radar',
    data: {
      labels: data.radar_features,
      datasets: datasets,
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          grid: { color: 'rgba(255,255,255,0.08)' },
          angleLines: { color: 'rgba(255,255,255,0.08)' },
          pointLabels: {
            font: { family: 'IBM Plex Mono', size: 10 },
            color: '#cbd5e1',
          },
          ticks: { display: false, min: 0, max: 100 },
        }
      },
      plugins: {
        legend: {
          position: 'bottom',
          labels: { font: { size: 11 }, color: '#94a3b8', boxWidth: 12 },
        }
      }
    }
  });
}

// ==========================================
// Render Segment Cards & Playbooks
// ==========================================
function renderSegmentCards(data) {
  const container = document.getElementById('segment-cards-container');
  if (!container) return;

  container.innerHTML = '';
  const currency = data.dataset_info.currency || '£';

  data.segments.forEach((seg, i) => {
    const color = CLUSTER_COLORS[i % CLUSTER_COLORS.length];
    const pb = seg.playbook;

    const card = document.createElement('div');
    card.className = 'segment-card';
    card.style.setProperty('--card-accent', color);

    let metricsHtml = '';
    if (seg.recency_days !== undefined) {
      metricsHtml = `
        <div class="segment-metrics-grid">
          <div class="metric-item">
            <span class="lbl">Avg Recency</span>
            <span class="val">${Math.round(seg.recency_days)}d</span>
          </div>
          <div class="metric-item">
            <span class="lbl">Avg Frequency</span>
            <span class="val">${seg.frequency.toFixed(1)} orders</span>
          </div>
          <div class="metric-item">
            <span class="lbl">Avg Spend</span>
            <span class="val">${currency}${Math.round(seg.monetary_total).toLocaleString()}</span>
          </div>
          <div class="metric-item">
            <span class="lbl">12mo CLV / Cust</span>
            <span class="val" style="color: ${color}">${currency}${Math.round(seg.clv_estimate).toLocaleString()}</span>
          </div>
        </div>
      `;
    } else {
      metricsHtml = `
        <div class="segment-metrics-grid">
          <div class="metric-item">
            <span class="lbl">Avg Income</span>
            <span class="val">$${Math.round(seg.annual_income)}k</span>
          </div>
          <div class="metric-item">
            <span class="lbl">Spending Score</span>
            <span class="val">${Math.round(seg.spending_score)}/100</span>
          </div>
          <div class="metric-item">
            <span class="lbl">Avg Age</span>
            <span class="val">${Math.round(seg.age)} yrs</span>
          </div>
          <div class="metric-item">
            <span class="lbl">Share of Base</span>
            <span class="val" style="color: ${color}">${seg.pct_of_base}%</span>
          </div>
        </div>
      `;
    }

    card.innerHTML = `
      <div>
        <div class="segment-badge-row">
          <span class="segment-pill" style="background: ${hexToRgba(color, 0.15)}; color: ${color}; border: 1px solid ${hexToRgba(color, 0.3)}">
            <span style="width: 8px; height: 8px; border-radius: 50%; background: ${color}"></span>
            Cluster ${seg.cluster_id}
          </span>
          <span class="segment-size-meta">${seg.customer_count.toLocaleString()} customers (${seg.pct_of_base}%)</span>
        </div>
        <div class="segment-name">${seg.name}</div>
        ${metricsHtml}
      </div>
      <div class="playbook-box">
        <div class="playbook-header">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"></path></svg>
          Marketing Playbook
        </div>
        <div class="playbook-strategy">${pb.strategy}</div>
        <div class="playbook-action"><strong>Target Action:</strong> ${pb.action_item}</div>
      </div>
    `;

    container.appendChild(card);
  });
}

// ==========================================
// Render Workbench Summary Table
// ==========================================
function renderWorkbenchSummary(data) {
  const tbody = document.getElementById('workbench-table-body');
  if (!tbody) return;

  tbody.innerHTML = '';
  const currency = data.dataset_info.currency || '£';

  data.segments.forEach((seg, i) => {
    const tr = document.createElement('tr');
    const color = CLUSTER_COLORS[i % CLUSTER_COLORS.length];

    tr.innerHTML = `
      <td><span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:${color}; margin-right:8px;"></span>${seg.name}</td>
      <td>${seg.customer_count.toLocaleString()} (${seg.pct_of_base}%)</td>
      <td>${seg.recency_days !== undefined ? Math.round(seg.recency_days) + 'd' : '-'}</td>
      <td>${seg.frequency !== undefined ? seg.frequency.toFixed(1) : '-'}</td>
      <td>${seg.monetary_total !== undefined ? currency + Math.round(seg.monetary_total).toLocaleString() : '-'}</td>
      <td>${seg.clv_estimate !== undefined ? currency + Math.round(seg.clv_estimate).toLocaleString() : '-'}</td>
      <td><span class="brand-badge">${seg.playbook.churn_risk_level || 'Moderate'}</span></td>
    `;
    tbody.appendChild(tr);
  });

  // Cross-validation agreement readout
  setElText('workbench-model-name', `${data.algorithm.toUpperCase()} (k=${data.chosen_k})`);
  setElText('workbench-sil-score', data.metrics.silhouette_score.toFixed(4));
  setElText('workbench-ari-score', data.metrics.ari_vs_kmeans.toFixed(4));
  setElText('workbench-db-score', data.metrics.davies_bouldin_index.toFixed(4));
  setElText('workbench-ch-score', data.metrics.calinski_harabasz_index.toFixed(1));
}

// ==========================================
// Render Pareto Curve (Chart.js)
// ==========================================
function renderParetoChart(data) {
  const ctx = document.getElementById('pareto-canvas')?.getContext('2d');
  if (!ctx || !data.pareto_data || !data.pareto_data.length) return;

  if (state.paretoChartInstance) {
    state.paretoChartInstance.destroy();
  }

  const labels = data.pareto_data.map(p => `${p.pct_customers}%`);
  const values = data.pareto_data.map(p => p.cum_revenue_pct);

  state.paretoChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: 'Cumulative % Revenue',
        data: values,
        borderColor: '#06b6d4',
        backgroundColor: 'rgba(6, 182, 212, 0.1)',
        fill: true,
        tension: 0.3,
        borderWidth: 2,
        pointBackgroundColor: '#06b6d4',
      }, {
        label: 'Equal Distribution Baseline (45°)',
        data: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        borderColor: 'rgba(255,255,255,0.2)',
        borderDash: [5, 5],
        borderWidth: 1.5,
        fill: false,
        pointRadius: 0,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: { display: true, text: 'Top % of Customer Base', color: '#94a3b8' },
          grid: { color: 'rgba(255,255,255,0.06)' },
          ticks: { color: '#94a3b8' }
        },
        y: {
          title: { display: true, text: 'Cumulative Revenue %', color: '#94a3b8' },
          grid: { color: 'rgba(255,255,255,0.06)' },
          ticks: { color: '#94a3b8' },
          min: 0,
          max: 100
        }
      },
      plugins: {
        legend: { labels: { color: '#cbd5e1' } }
      }
    }
  });
}

// ==========================================
// Render K-Selection Evaluation Curve
// ==========================================
function renderKEvalCurve() {
  const ctx = document.getElementById('k-eval-canvas')?.getContext('2d');
  if (!ctx || !state.clusteringData || !state.clusteringData.k_eval_curve) return;

  if (state.kEvalChartInstance) {
    state.kEvalChartInstance.destroy();
  }

  const curve = state.clusteringData.k_eval_curve;
  const labels = curve.map(c => `k=${c.k}`);
  const inertias = curve.map(c => c.inertia);
  const silhouettes = curve.map(c => c.silhouette);

  state.kEvalChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Silhouette Score (Higher = Better Separation)',
          data: silhouettes,
          borderColor: '#10b981',
          backgroundColor: 'rgba(16, 185, 129, 0.1)',
          yAxisID: 'y1',
          borderWidth: 2.5,
          tension: 0.2,
          pointRadius: 4,
          pointBackgroundColor: '#10b981',
        },
        {
          label: 'Inertia / WCSS (Elbow Curve)',
          data: inertias,
          borderColor: '#6366f1',
          yAxisID: 'y2',
          borderWidth: 2,
          borderDash: [4, 4],
          pointRadius: 3,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y1: {
          type: 'linear',
          position: 'left',
          title: { display: true, text: 'Silhouette Score', color: '#10b981' },
          grid: { color: 'rgba(255,255,255,0.06)' },
          ticks: { color: '#10b981' }
        },
        y2: {
          type: 'linear',
          position: 'right',
          title: { display: true, text: 'Inertia (Within-Cluster Sum of Squares)', color: '#a5b4fc' },
          grid: { drawOnChartArea: false },
          ticks: { color: '#a5b4fc' }
        },
        x: {
          grid: { color: 'rgba(255,255,255,0.06)' },
          ticks: { color: '#94a3b8' }
        }
      },
      plugins: {
        legend: { labels: { color: '#cbd5e1' } }
      }
    }
  });
}

// ==========================================
// Customer Prediction Simulator
// ==========================================
function initSimulator() {
  const btnPredict = document.getElementById('btn-simulate-predict');
  if (!btnPredict) return;

  btnPredict.addEventListener('click', runSimulationPrediction);

  // Auto-recalculate on change
  ['sim-recency', 'sim-frequency', 'sim-monetary', 'sim-aov', 'sim-basket', 'sim-tenure'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('input', debounce(runSimulationPrediction, 300));
  });
}

function updateSimulatorDefaults(data) {
  // If champions exist, set reasonable defaults
  const champ = data.segments[0];
  if (champ && champ.recency_days !== undefined) {
    setVal('sim-recency', Math.round(champ.recency_days * 1.2));
    setVal('sim-frequency', Math.max(1, Math.round(champ.frequency * 0.8)));
    setVal('sim-monetary', Math.round(champ.monetary_total * 0.8));
    setVal('sim-aov', Math.round(champ.avg_order_value));
    setVal('sim-basket', Math.round(champ.avg_basket_size));
    setVal('sim-tenure', Math.round(champ.tenure_days || 180));
    runSimulationPrediction();
  }
}

async function runSimulationPrediction() {
  try {
    const payload = {
      recency_days: parseFloat(document.getElementById('sim-recency')?.value || 30),
      frequency: parseFloat(document.getElementById('sim-frequency')?.value || 4),
      monetary_total: parseFloat(document.getElementById('sim-monetary')?.value || 850),
      avg_order_value: parseFloat(document.getElementById('sim-aov')?.value || 212),
      avg_basket_size: parseFloat(document.getElementById('sim-basket')?.value || 5),
      tenure_days: parseFloat(document.getElementById('sim-tenure')?.value || 120),
    };

    const res = await fetch('/api/predict-customer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) return;
    const pred = await res.json();

    setElText('sim-result-persona', pred.predicted_persona);
    setElText('sim-result-cluster', `Assigned to Cluster ${pred.predicted_cluster}`);
    setElText('sim-result-clv', `£${pred.clv_estimate.toLocaleString()}`);
    setElText('sim-result-dist', `${pred.distance_to_centroid} (centroid dist)`);
    setElText('sim-result-risk', pred.playbook.churn_risk_level);
    setElText('sim-result-strategy', pred.playbook.strategy);
    setElText('sim-result-action', pred.playbook.action_item);
    setElText('sim-result-channels', pred.playbook.key_channels.join(', '));
    setElText('sim-result-discount', pred.playbook.discount_sensitivity);

  } catch (err) {
    console.error('Simulation error:', err);
  }
}

// ==========================================
// Marketing ROI & Decision Simulator
// ==========================================
function initRoiCalculator() {
  const btnCalc = document.getElementById('btn-calc-roi');
  if (!btnCalc) return;

  btnCalc.addEventListener('click', runRoiCalculation);
  ['roi-budget', 'roi-winback-rate', 'roi-loyalty-rate', 'roi-cost-per-cust'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('input', debounce(runRoiCalculation, 300));
  });
}

async function runRoiCalculation() {
  try {
    const payload = {
      marketing_budget: parseFloat(document.getElementById('roi-budget')?.value || 10000),
      winback_conversion_rate: parseFloat(document.getElementById('roi-winback-rate')?.value || 18),
      loyalty_upsell_rate: parseFloat(document.getElementById('roi-loyalty-rate')?.value || 22),
      campaign_cost_per_customer: parseFloat(document.getElementById('roi-cost-per-cust')?.value || 2.50),
    };

    const res = await fetch('/api/simulate-roi', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) return;
    const roi = await res.json();

    setElText('roi-targeted-count', roi.targeted_customers.toLocaleString());
    setElText('roi-gross-lift', `£${Math.round(roi.gross_revenue_lift).toLocaleString()}`);
    setElText('roi-net-profit', `£${Math.round(roi.net_profit).toLocaleString()}`);
    setElText('roi-pct-val', `${roi.roi_percent}%`);

    setElText('roi-vip-upsell', `£${Math.round(roi.breakdown?.vip_upsell_revenue || 0).toLocaleString()}`);
    setElText('roi-atrisk-saved', `£${Math.round(roi.breakdown?.recovered_at_risk_revenue || 0).toLocaleString()}`);
    setElText('roi-actual-cost', `£${Math.round(roi.budget_allocated).toLocaleString()}`);

  } catch (err) {
    console.error('ROI calculation error:', err);
  }
}

// ==========================================
// Custom Dataset Uploader
// ==========================================
function initUploader() {
  const fileInput = document.getElementById('file-upload-input');
  const dropzone = document.getElementById('upload-dropzone');
  const uploadStatus = document.getElementById('upload-status');

  if (!dropzone || !fileInput) return;

  dropzone.addEventListener('click', () => fileInput.click());

  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.style.borderColor = 'var(--accent-cyan)';
  });

  dropzone.addEventListener('dragleave', () => {
    dropzone.style.borderColor = 'rgba(99, 102, 241, 0.4)';
  });

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.style.borderColor = 'rgba(99, 102, 241, 0.4)';
    if (e.dataTransfer.files.length) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
      handleFileUpload(e.target.files[0]);
    }
  });
}

async function handleFileUpload(file) {
  const uploadStatus = document.getElementById('upload-status');
  if (uploadStatus) {
    uploadStatus.innerHTML = `<span class="spinner" style="display:inline-block; vertical-align:middle; margin-right:8px;"></span> Ingesting and cleaning ${file.name}...`;
    uploadStatus.classList.remove('hidden');
  }

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/api/upload-dataset', {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Upload failed');
    }

    const data = await res.json();
    if (uploadStatus) {
      uploadStatus.innerHTML = `✓ Successfully loaded <strong>${data.name}</strong> (${data.clean_rows.toLocaleString()} clean rows, ${data.customers.toLocaleString()} customers).`;
      uploadStatus.style.color = 'var(--accent-emerald)';
    }

    await fetchDatasets();
    state.currentDataset = data.dataset_id;
    const datasetSelect = document.getElementById('dataset-select');
    if (datasetSelect) datasetSelect.value = data.dataset_id;

    executeClustering();

  } catch (err) {
    if (uploadStatus) {
      uploadStatus.innerHTML = `✗ Error: ${err.message}`;
      uploadStatus.style.color = 'var(--accent-rose)';
    }
  }
}

// ==========================================
// Utility Helpers
// ==========================================
function setElText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function setVal(id, val) {
  const el = document.getElementById(id);
  if (el) el.value = val;
}

function hexToRgba(hex, alpha) {
  hex = hex.replace('#', '');
  if (hex.length === 3) {
    hex = hex.split('').map(c => c + c).join('');
  }
  const r = parseInt(hex.substring(0, 2), 16);
  const g = parseInt(hex.substring(2, 4), 16);
  const b = parseInt(hex.substring(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function debounce(fn, ms) {
  let timer;
  return function(...args) {
    clearTimeout(timer);
    timer = setTimeout(() => fn.apply(this, args), ms);
  };
}
