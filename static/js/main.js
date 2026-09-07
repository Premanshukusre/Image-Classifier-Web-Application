/* ==========================================================================
   ADVANCED AI DIAGNOSTIC DASHBOARD - APPLE DISEASE CLASSIFIER JAVASCRIPT
   ========================================================================== */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const btnSelectFile = document.getElementById("btnSelectFile");
  const btnWebcam = document.getElementById("btnWebcam");
  const btnThemeToggle = document.getElementById("btnThemeToggle");
  const btnExportPDF = document.getElementById("btnExportPDF");

  const cameraModal = document.getElementById("cameraModal");
  const cameraVideo = document.getElementById("cameraVideo");
  const btnSnapPhoto = document.getElementById("btnSnapPhoto");
  const btnCloseCamera = document.getElementById("btnCloseCamera");

  const loadingOverlay = document.getElementById("loadingOverlay");
  const loadingStatusText = document.getElementById("loadingStatusText");

  const emptyStateCard = document.getElementById("emptyStateCard");
  const activeResultSection = document.getElementById("activeResultSection");

  const resDiseaseName = document.getElementById("resDiseaseName");
  const resSeverityBadge = document.getElementById("resSeverityBadge");
  const resConfidenceVal = document.getElementById("resConfidenceVal");
  const resPreviewImg = document.getElementById("resPreviewImg");
  const probBarsContainer = document.getElementById("probBarsContainer");

  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabPanels = document.querySelectorAll(".tab-content-panel");

  const leafCanvas = document.getElementById("leafInspectorCanvas");
  const filterBtns = document.querySelectorAll(".filter-btn");
  const historyGrid = document.getElementById("historyGrid");

  let currentImageObject = null;
  let activeStream = null;

  // Initialize Theme
  initTheme();

  // Load Scan History
  renderScanHistory();

  // ==========================================================================
  // THEME SWITCHER
  // ==========================================================================

  function initTheme() {
    const savedTheme = localStorage.getItem("theme") || "light";
    document.documentElement.setAttribute("data-theme", savedTheme);
    updateThemeIcon(savedTheme);
  }

  btnThemeToggle.addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme");
    const next = current === "light" ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("theme", next);
    updateThemeIcon(next);
  });

  function updateThemeIcon(theme) {
    btnThemeToggle.innerHTML = theme === "light" ? "🌙" : "☀️";
    btnThemeToggle.title = `Switch to ${theme === "light" ? "Dark" : "Light"} Mode`;
  }

  // ==========================================================================
  // FILE UPLOAD & DRAG DROP
  // ==========================================================================

  btnSelectFile.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
      handleFileSelected(files[0]);
    }
  });

  function handleFileSelected(file) {
    if (!file.type.startsWith("image/")) {
      alert("Please select a valid image file (JPG, PNG, WEBP).");
      return;
    }
    const formData = new FormData();
    formData.append("image", file);
    sendPredictionRequest(formData);
  }

  // Preset Sample Pickers
  document.querySelectorAll(".sample-card").forEach((card) => {
    card.addEventListener("click", () => {
      const sampleFile = card.getAttribute("data-sample");
      if (sampleFile) {
        sendPredictionRequest({ sample: sampleFile }, true);
      }
    });
  });

  // ==========================================================================
  // WEBCAM CAMERA MODAL
  // ==========================================================================

  btnWebcam.addEventListener("click", async () => {
    try {
      activeStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
      });
      cameraVideo.srcObject = activeStream;
      cameraModal.classList.add("active");
    } catch (err) {
      alert("Unable to access camera: " + err.message);
    }
  });

  btnCloseCamera.addEventListener("click", closeCameraModal);

  function closeCameraModal() {
    if (activeStream) {
      activeStream.getTracks().forEach((track) => track.stop());
      activeStream = null;
    }
    cameraModal.classList.remove("active");
  }

  btnSnapPhoto.addEventListener("click", () => {
    if (!cameraVideo.srcObject) return;
    const canvas = document.createElement("canvas");
    canvas.width = cameraVideo.videoWidth || 640;
    canvas.height = cameraVideo.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(cameraVideo, 0, 0, canvas.width, canvas.height);

    canvas.toBlob((blob) => {
      const file = new File([blob], `camera_snap_${Date.now()}.jpg`, { type: "image/jpeg" });
      closeCameraModal();
      handleFileSelected(file);
    }, "image/jpeg", 0.92);
  });

  // ==========================================================================
  // AJAX PREDICTION API REQUEST
  // ==========================================================================

  async function sendPredictionRequest(payload, isJson = false) {
    showLoadingOverlay();

    const steps = [
      "Uploading leaf image...",
      "Initializing Neural Network model...",
      "Extracting lesion and chlorosis patterns...",
      "Computing confidence probabilities...",
      "Synthesizing botanical recommendations..."
    ];

    let stepIdx = 0;
    const interval = setInterval(() => {
      if (stepIdx < steps.length) {
        loadingStatusText.textContent = steps[stepIdx];
        stepIdx++;
      }
    }, 300);

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 60000);

      let options = { method: "POST", signal: controller.signal };
      if (isJson) {
        options.headers = { "Content-Type": "application/json" };
        options.body = JSON.stringify(payload);
      } else {
        options.body = payload;
      }

      const resp = await fetch("/api/predict", options);
      clearTimeout(timeoutId);
      const data = await resp.json();

      clearInterval(interval);
      hideLoadingOverlay();

      if (data.success) {
        renderDiagnosticDashboard(data);
        saveScanToHistory(data);
      } else {
        alert("Diagnostic Notice: " + (data.error || "The server could not process this image. Please try again."));
      }
    } catch (err) {
      clearInterval(interval);
      hideLoadingOverlay();
      if (err.name === "AbortError") {
        alert("Request timed out. The server took longer than 60 seconds to respond. Please retry.");
      } else {
        alert("Network or Server error: " + err.message);
      }
    }
  }

  function showLoadingOverlay() {
    loadingOverlay.classList.add("active");
  }

  function hideLoadingOverlay() {
    loadingOverlay.classList.remove("active");
  }

  // ==========================================================================
  // DIAGNOSTIC DASHBOARD RENDERER
  // ==========================================================================

  function renderDiagnosticDashboard(data) {
    emptyStateCard.style.display = "none";
    activeResultSection.style.display = "block";

    // 1. Title & Badge
    resDiseaseName.textContent = data.prediction;
    resPreviewImg.src = data.image_url;

    const info = data.disease_info || {};
    const status = info.status || "healthy";
    const sevLabel = info.severity_label || "Normal";

    resSeverityBadge.className = `severity-badge ${status}`;
    resSeverityBadge.innerHTML = `<span>●</span> ${sevLabel}`;

    // 2. Confidence Counter Animation
    animateConfidenceCounter(data.confidence);

    // 3. Probability Distribution Bars
    renderProbabilityBars(data.all_probabilities, data.prediction);

    // 4. Tabbed Knowledge Base
    document.getElementById("infoSummary").textContent = info.summary || "No summary available.";
    renderBulletList("infoSymptoms", info.symptoms || []);
    renderBulletList("infoOrganic", info.organic_remedies || []);
    renderBulletList("infoChemical", info.chemical_controls || []);
    renderBulletList("infoPrevention", info.prevention_tips || []);

    // 5. Canvas Inspector Setup
    setupCanvasInspector(data.image_url);

    // Scroll smoothly to diagnostic panel
    activeResultSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function animateConfidenceCounter(targetVal) {
    let curr = 0;
    const duration = 800; // ms
    const stepTime = 20;
    const steps = duration / stepTime;
    const increment = targetVal / steps;

    const timer = setInterval(() => {
      curr += increment;
      if (curr >= targetVal) {
        curr = targetVal;
        clearInterval(timer);
      }
      resConfidenceVal.textContent = curr.toFixed(1) + "%";
    }, stepTime);
  }

  function renderProbabilityBars(allProbs, topPrediction) {
    probBarsContainer.innerHTML = "";
    if (!allProbs) return;

    Object.entries(allProbs).forEach(([name, prob]) => {
      const isTop = name === topPrediction;
      const row = document.createElement("div");
      row.className = "prob-row";
      row.innerHTML = `
        <div class="prob-labels">
          <span style="${isTop ? 'color: var(--accent-emerald); font-weight: 700;' : ''}">${name}</span>
          <span>${prob}%</span>
        </div>
        <div class="prob-track">
          <div class="prob-fill ${isTop ? 'highlight' : ''}" style="width: 0%;"></div>
        </div>
      `;
      probBarsContainer.appendChild(row);

      // Trigger transition animation after DOM insertion
      setTimeout(() => {
        row.querySelector(".prob-fill").style.width = `${prob}%`;
      }, 50);
    });
  }

  function renderBulletList(elementId, items) {
    const el = document.getElementById(elementId);
    if (!el) return;
    el.innerHTML = items.map((item) => `<li>${item}</li>`).join("");
  }

  // Tab Switcher Logic
  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      tabBtns.forEach((b) => b.classList.remove("active"));
      tabPanels.forEach((p) => p.classList.remove("active"));

      btn.classList.add("active");
      const targetTab = btn.getAttribute("data-tab");
      document.getElementById(`tab-${targetTab}`).classList.add("active");
    });
  });

  // ==========================================================================
  // LEAF CANVAS VISUAL INSPECTOR
  // ==========================================================================

  function setupCanvasInspector(imageUrl) {
    const ctx = leafCanvas.getContext("2d");
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.src = imageUrl;

    img.onload = () => {
      currentImageObject = img;
      leafCanvas.width = img.width;
      leafCanvas.height = img.height;
      applyCanvasFilter("normal");
    };

    filterBtns.forEach((btn) => {
      btn.onclick = () => {
        filterBtns.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        const filterType = btn.getAttribute("data-filter");
        applyCanvasFilter(filterType);
      };
    });
  }

  function applyCanvasFilter(filterType) {
    if (!currentImageObject) return;
    const ctx = leafCanvas.getContext("2d");
    ctx.filter = "none";
    ctx.drawImage(currentImageObject, 0, 0);

    if (filterType === "contrast") {
      ctx.filter = "contrast(180%) saturate(140%)";
      ctx.drawImage(currentImageObject, 0, 0);
    } else if (filterType === "edge") {
      // Basic edge detection matrix
      const imgData = ctx.getImageData(0, 0, leafCanvas.width, leafCanvas.height);
      const d = imgData.data;
      const w = leafCanvas.width;
      const h = leafCanvas.height;

      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const i = (y * w + x) * 4;
          const next = (y * w + (x + 1)) * 4;
          const diff = Math.abs(d[i] - d[next]) + Math.abs(d[i + 1] - d[next + 1]);
          const val = diff > 40 ? 255 : 0;
          d[i] = 0;
          d[i + 1] = val;
          d[i + 2] = val > 0 ? 150 : 0;
        }
      }
      ctx.putImageData(imgData, 0, 0);
    } else if (filterType === "thermal") {
      // False color thermal simulation for damaged plant tissue
      const imgData = ctx.getImageData(0, 0, leafCanvas.width, leafCanvas.height);
      const d = imgData.data;
      for (let i = 0; i < d.length; i += 4) {
        const avg = (d[i] + d[i + 1] + d[i + 2]) / 3;
        if (avg < 80) {
          d[i] = 239; d[i + 1] = 68; d[i + 2] = 68; // Damaged spots -> Red
        } else if (avg < 140) {
          d[i] = 245; d[i + 1] = 158; d[i + 2] = 11; // Chlorosis -> Yellow
        } else {
          d[i] = 16; d[i + 1] = 185; d[i + 2] = 129; // Healthy green
        }
      }
      ctx.putImageData(imgData, 0, 0);
    }
  }

  // ==========================================================================
  // SESSION SCAN HISTORY (LOCAL STORAGE)
  // ==========================================================================

  function saveScanToHistory(data) {
    try {
      let history = JSON.parse(localStorage.getItem("apple_scans") || "[]");
      history.unshift({
        prediction: data.prediction,
        confidence: data.confidence,
        imageUrl: data.image_url,
        timestamp: data.timestamp,
        fullData: data,
      });
      if (history.length > 8) history = history.slice(0, 8);
      localStorage.setItem("apple_scans", JSON.stringify(history));
      renderScanHistory();
    } catch (e) {
      console.warn("Unable to save scan to localStorage:", e);
    }
  }

  function renderScanHistory() {
    if (!historyGrid) return;
    try {
      const history = JSON.parse(localStorage.getItem("apple_scans") || "[]");
      if (history.length === 0) {
        historyGrid.innerHTML = `<p style="color: var(--text-muted); font-size: 0.82rem;">No previous scans recorded in this session.</p>`;
        return;
      }
      historyGrid.innerHTML = history
        .map(
          (item, idx) => `
        <div class="history-card" data-idx="${idx}">
          <img src="${item.imageUrl}" class="history-thumb" alt="${item.prediction}">
          <div class="history-name">${item.prediction}</div>
          <div class="history-date">${item.confidence}% • ${item.timestamp}</div>
        </div>
      `
        )
        .join("");

      historyGrid.querySelectorAll(".history-card").forEach((card) => {
        card.addEventListener("click", () => {
          const idx = parseInt(card.getAttribute("data-idx"));
          if (history[idx]) {
            renderDiagnosticDashboard(history[idx].fullData);
          }
        });
      });
    } catch (e) {
      console.warn("Error rendering history:", e);
    }
  }

  // Export / Print PDF Report
  if (btnExportPDF) {
    btnExportPDF.addEventListener("click", () => {
      window.print();
    });
  }
});
