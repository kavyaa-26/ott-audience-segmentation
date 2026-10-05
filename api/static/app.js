// OTT Audience Intelligence & Personalization Service - Frontend Application Logic

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const navTabs = document.querySelectorAll(".nav-tab");
  const viewPanels = document.querySelectorAll(".view-panel");
  const statusDot = document.getElementById("system-status-dot");
  const statusText = document.getElementById("system-status-text");

  // Form & Inputs
  const viewerForm = document.getElementById("viewer-form");
  const inputUserId = document.getElementById("input-user-id");
  const inputWatchTime = document.getElementById("input-watch-time");
  const inputAvgSession = document.getElementById("input-avg-session");
  const inputVisitsWeek = document.getElementById("input-visits-week");
  const genreChips = document.querySelectorAll(".genre-chip");
  const selectedGenresDisplay = document.getElementById("selected-genres-display");
  const btnSubmit = document.getElementById("btn-submit-analyze");
  const submitSpinner = document.getElementById("submit-spinner");
  const demoButtonsContainer = document.getElementById("demo-buttons-container");

  // Error Banner
  const errorBanner = document.getElementById("error-banner");
  const errorBannerMessage = document.getElementById("error-banner-message");
  const errorBannerClose = document.getElementById("error-banner-close");

  // Results View
  const resultsEmptyState = document.getElementById("results-empty-state");
  const resultsDisplay = document.getElementById("results-display");
  const resSegmentName = document.getElementById("res-segment-name");
  const resSegmentId = document.getElementById("res-segment-id");
  const resCentroidDist = document.getElementById("res-centroid-dist");
  const resUserId = document.getElementById("res-user-id");
  const resWatchTime = document.getElementById("res-watch-time");
  const resSession = document.getElementById("res-session");
  const resVisits = document.getElementById("res-visits");
  const resGenres = document.getElementById("res-genres");
  const resExplanation = document.getElementById("res-explanation-text");
  const recommendationsContainer = document.getElementById("recommendations-container");

  // Segments View
  const segmentsCardsContainer = document.getElementById("segments-cards-container");
  const segmentsMetaSummary = document.getElementById("segments-meta-summary");

  // Status View
  const statusModelName = document.getElementById("status-model-name");
  const statusPreprocessor = document.getElementById("status-preprocessor");
  const statusKClusters = document.getElementById("status-k-clusters");
  const statusSilhouette = document.getElementById("status-silhouette");
  const statusModelStatus = document.getElementById("status-model-status");
  const statusApiStatus = document.getElementById("status-api-status");
  const kEvaluationTbody = document.getElementById("k-evaluation-tbody");

  // Modal
  const modalContainer = document.getElementById("modal-container");
  const modalTitle = document.getElementById("modal-title");
  const modalBody = document.getElementById("modal-body");
  const modalClose = document.getElementById("modal-close");
  const modalDismiss = document.getElementById("modal-dismiss");
  const linkPrivacy = document.getElementById("link-privacy");
  const linkTerms = document.getElementById("link-terms");

  // State
  let selectedGenres = new Set();
  let demoUsersList = [];
  let segmentsMetadataCache = null;

  // Initialize
  initNavigation();
  initGenreChips();
  initDemoUsers();
  initErrorBanner();
  initModals();
  fetchSystemStatus();
  fetchSegmentsMetadata();

  // Navigation Tabs
  function initNavigation() {
    navTabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        const targetTab = tab.getAttribute("data-tab");
        navTabs.forEach((t) => {
          t.classList.remove("active");
          t.setAttribute("aria-selected", "false");
        });
        viewPanels.forEach((p) => {
          p.classList.remove("active");
          p.hidden = true;
        });

        tab.classList.add("active");
        tab.setAttribute("aria-selected", "true");
        const activePanel = document.getElementById(`view-${targetTab}`);
        if (activePanel) {
          activePanel.classList.add("active");
          activePanel.hidden = false;
        }

        if (targetTab === "segments" && !segmentsMetadataCache) {
          fetchSegmentsMetadata();
        } else if (targetTab === "status") {
          fetchSystemStatus();
        }
      });
    });
  }

  // Genre Selection Chips
  function initGenreChips() {
    genreChips.forEach((chip) => {
      chip.addEventListener("click", () => {
        const genre = chip.getAttribute("data-genre");
        if (selectedGenres.has(genre)) {
          selectedGenres.delete(genre);
          chip.classList.remove("selected");
        } else {
          selectedGenres.add(genre);
          chip.classList.add("selected");
        }
        updateGenreDisplay();
      });
    });
  }

  function updateGenreDisplay() {
    if (selectedGenres.size === 0) {
      selectedGenresDisplay.textContent = "Selected: None";
    } else {
      selectedGenresDisplay.textContent = `Selected: ${Array.from(selectedGenres).join(", ")}`;
    }
  }

  function setSelectedGenres(genres) {
    selectedGenres.clear();
    const cleanList = (genres || []).map((g) => g.trim().toLowerCase());
    genreChips.forEach((chip) => {
      const gName = chip.getAttribute("data-genre");
      if (cleanList.includes(gName.toLowerCase())) {
        selectedGenres.add(gName);
        chip.classList.add("selected");
      } else {
        chip.classList.remove("selected");
      }
    });
    updateGenreDisplay();
  }

  // Error Banner handling
  function initErrorBanner() {
    errorBannerClose.addEventListener("click", () => {
      errorBanner.classList.add("hidden");
    });
  }

  function showError(msg) {
    errorBannerMessage.textContent = msg;
    errorBanner.classList.remove("hidden");
    errorBanner.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function clearError() {
    errorBanner.classList.add("hidden");
    errorBannerMessage.textContent = "";
  }

  // Option A: Seeded Demo Users
  async function initDemoUsers() {
    try {
      const res = await fetch("/api/demo-users");
      if (res.ok) {
        demoUsersList = await res.json();
      } else {
        demoUsersList = getFallbackDemoUsers();
      }
    } catch (e) {
      demoUsersList = getFallbackDemoUsers();
    }
    renderDemoButtons();
  }

  function getFallbackDemoUsers() {
    return [
      { user_id: "DEMO-ACTION-001", name: "High Engagement Action", watch_time_hours: 42, avg_session_mins: 105, visits_per_week: 28, top_genres: ["Action", "Thriller"] },
      { user_id: "DEMO-CASUAL-001", name: "Casual Viewer", watch_time_hours: 5, avg_session_mins: 18, visits_per_week: 5, top_genres: ["Comedy", "Family"] },
      { user_id: "DEMO-FOCUSED-001", name: "Genre-Focused", watch_time_hours: 24, avg_session_mins: 60, visits_per_week: 17, top_genres: ["Drama", "Drama"] },
      { user_id: "DEMO-LOW-001", name: "Low Activity", watch_time_hours: 2, avg_session_mins: 12, visits_per_week: 2, top_genres: ["Family"] },
      { user_id: "DEMO-EXPLORER-001", name: "Genre Explorer", watch_time_hours: 30, avg_session_mins: 55, visits_per_week: 22, top_genres: ["Action", "Comedy", "Drama", "Sci-Fi"] },
      { user_id: "DEMO-DRAMA-001", name: "Drama Viewer", watch_time_hours: 21, avg_session_mins: 65, visits_per_week: 15, top_genres: ["Romance", "Drama"] },
      { user_id: "DEMO-THRILLER-001", name: "Thriller Viewer", watch_time_hours: 29, avg_session_mins: 72, visits_per_week: 19, top_genres: ["Thriller", "Mystery"] },
      { user_id: "DEMO-FAMILY-001", name: "Family Viewer", watch_time_hours: 10, avg_session_mins: 35, visits_per_week: 8, top_genres: ["Family", "Animation"] }
    ];
  }

  function renderDemoButtons() {
    demoButtonsContainer.innerHTML = "";
    demoUsersList.forEach((user) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "btn-demo";
      btn.setAttribute("aria-label", `Select demo user ${user.name}`);
      btn.innerHTML = `
        <span class="demo-btn-title">${escapeHtml(user.name)}</span>
        <span class="demo-btn-genres">${escapeHtml(user.top_genres.join(", "))}</span>
      `;
      btn.addEventListener("click", () => {
        document.querySelectorAll(".btn-demo").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        populateForm(user);
        clearError();
      });
      demoButtonsContainer.appendChild(btn);
    });
  }

  function populateForm(user) {
    inputUserId.value = user.user_id;
    inputWatchTime.value = user.watch_time_hours;
    inputAvgSession.value = user.avg_session_mins;
    inputVisitsWeek.value = user.visits_per_week;
    setSelectedGenres(user.top_genres);
  }

  // Form Submit -> POST /recommend
  viewerForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    clearError();

    const userId = inputUserId.value.trim();
    const watchTime = parseFloat(inputWatchTime.value);
    const avgSession = parseFloat(inputAvgSession.value);
    const visits = parseFloat(inputVisitsWeek.value);
    const genres = Array.from(selectedGenres);

    // Client-side validations
    if (!userId) {
      showError("User ID is required.");
      return;
    }
    if (isNaN(watchTime) || watchTime < 0) {
      showError("Watch Time must be a valid non-negative number.");
      return;
    }
    if (isNaN(avgSession) || avgSession < 0) {
      showError("Average Session duration must be a valid non-negative number.");
      return;
    }
    if (isNaN(visits) || visits < 0) {
      showError("Visits per Week must be a valid non-negative number.");
      return;
    }
    if (genres.length === 0) {
      showError("Please select at least one preferred genre.");
      return;
    }

    const payload = {
      user_id: userId,
      watch_time_hours: watchTime,
      avg_session_mins: avgSession,
      visits_per_week: visits,
      top_genres: genres
    };

    setLoadingState(true);

    try {
      const response = await fetch("/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Error obtaining recommendations from API.");
      }

      renderResults(data, payload);
    } catch (err) {
      showError(err.message || "An unexpected error occurred while communicating with the API.");
    } finally {
      setLoadingState(false);
    }
  });

  function setLoadingState(isLoading) {
    btnSubmit.disabled = isLoading;
    if (isLoading) {
      submitSpinner.classList.remove("hidden");
    } else {
      submitSpinner.classList.add("hidden");
    }
  }

  // Render Results View
  function renderResults(data, submittedPayload) {
    resultsEmptyState.classList.add("hidden");
    resultsDisplay.classList.remove("hidden");

    resSegmentName.textContent = data.segment_name;
    resSegmentId.textContent = `Cluster ID: ${data.segment_id}`;
    resCentroidDist.textContent = `${Number(data.distance_to_centroid).toFixed(2)}`;

    resUserId.textContent = data.user_id;
    resWatchTime.textContent = `${submittedPayload.watch_time_hours} hrs`;
    resSession.textContent = `${submittedPayload.avg_session_mins} mins`;
    resVisits.textContent = `${submittedPayload.visits_per_week} / wk`;
    resGenres.textContent = submittedPayload.top_genres.join(", ");

    resExplanation.textContent = `Recommendations are tailored specifically to match your preferred genre portfolio (${submittedPayload.top_genres.join(", ")}) aligned with the viewing characteristics of the "${data.segment_name}" audience cluster.`;

    recommendationsContainer.innerHTML = "";
    if (data.recommendations && data.recommendations.length > 0) {
      data.recommendations.forEach((movie) => {
        const card = document.createElement("article");
        card.className = "poster-card";
        card.innerHTML = `
          <div class="poster-header">
            <h5 class="poster-title">${escapeHtml(movie.title)}</h5>
            <div class="poster-badges">
              <span class="badge-genre">${escapeHtml(movie.genre)}</span>
            </div>
          </div>
          <p class="poster-reason">${escapeHtml(movie.reason)}</p>
        `;
        recommendationsContainer.appendChild(card);
      });
    } else {
      recommendationsContainer.innerHTML = `<p class="section-desc">No recommendations returned.</p>`;
    }

    resultsDisplay.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  // Fetch System Status (/api/status & /health)
  async function fetchSystemStatus() {
    try {
      const healthRes = await fetch("/health");
      const healthData = await healthRes.json();
      if (healthRes.ok && healthData.model_loaded) {
        statusDot.className = "status-indicator-dot online";
        statusText.textContent = "API Ready";
        statusApiStatus.textContent = "Operational";
        statusModelStatus.textContent = "Loaded & Ready";
      } else {
        statusDot.className = "status-indicator-dot offline";
        statusText.textContent = "Model Initializing";
        statusApiStatus.textContent = "Awaiting Model";
        statusModelStatus.textContent = "Not Ready";
      }

      const statusRes = await fetch("/api/status");
      if (statusRes.ok) {
        const s = await statusRes.json();
        statusModelName.textContent = s.model || "KMeans";
        statusPreprocessor.textContent = s.preprocessing || "StandardScaler";
        statusKClusters.textContent = s.clusters !== null ? s.clusters : "—";
        statusSilhouette.textContent = s.silhouette_score !== null ? s.silhouette_score : "—";
      }
    } catch (e) {
      statusDot.className = "status-indicator-dot offline";
      statusText.textContent = "Offline";
      statusApiStatus.textContent = "Error";
      statusModelStatus.textContent = "Unavailable";
    }
  }

  // Fetch Audience Segments Metadata (/api/segments)
  async function fetchSegmentsMetadata() {
    try {
      const res = await fetch("/api/segments");
      if (!res.ok) throw new Error("Metadata unavailable");
      const meta = await res.json();
      segmentsMetadataCache = meta;
      renderSegmentsView(meta);
      renderKEvaluationTable(meta);
    } catch (e) {
      segmentsCardsContainer.innerHTML = `
        <div class="empty-state">
          <p class="empty-state-desc">Model metadata is currently loading or unavailable. Run the trainer service to produce cluster profiles.</p>
        </div>
      `;
    }
  }

  function renderSegmentsView(meta) {
    // Meta Summary Pills
    segmentsMetaSummary.innerHTML = `
      <div class="summary-pill">Total Clusters: <strong>${meta.num_clusters}</strong></div>
      <div class="summary-pill">Silhouette Score: <strong>${meta.silhouette_score}</strong></div>
      <div class="summary-pill">Clustering Inertia: <strong>${meta.inertia}</strong></div>
      <div class="summary-pill">Preprocessing: <strong>${escapeHtml(meta.preprocessor)}</strong></div>
    `;

    // Cards Grid
    segmentsCardsContainer.innerHTML = "";
    (meta.clusters || []).forEach((c) => {
      const card = document.createElement("div");
      card.className = "segment-card";

      const genresHtml = (c.dominant_genres || [])
        .map((g) => `<span class="genre-tag">${escapeHtml(g)}</span>`)
        .join("");

      card.innerHTML = `
        <div class="segment-card-header">
          <h4 class="segment-card-title">${escapeHtml(c.segment_name)}</h4>
          <span class="segment-card-id">Cluster ${c.cluster_id}</span>
        </div>
        <div class="segment-card-body">
          <div class="card-stat-row">
            <span class="stat-label">Segment Size:</span>
            <span class="stat-numeric">${c.cluster_size} viewers</span>
          </div>
          <div class="card-stat-row">
            <span class="stat-label">Average Watch Time:</span>
            <span class="stat-numeric">${c.avg_watch_time} hrs</span>
          </div>
          <div class="card-stat-row">
            <span class="stat-label">Average Session:</span>
            <span class="stat-numeric">${c.avg_session_mins} mins</span>
          </div>
          <div class="card-stat-row">
            <span class="stat-label">Visit Frequency:</span>
            <span class="stat-numeric">${c.avg_visit_frequency} / week</span>
          </div>
          <div>
            <span class="stat-label">Dominant Genres:</span>
            <div class="dominant-genres-list">
              ${genresHtml || '<span class="genre-tag">Diverse</span>'}
            </div>
          </div>
        </div>
      `;
      segmentsCardsContainer.appendChild(card);
    });
  }

  function renderKEvaluationTable(meta) {
    if (!meta.candidate_evaluations || meta.candidate_evaluations.length === 0) return;

    kEvaluationTbody.innerHTML = "";
    const chosenK = meta.num_clusters;

    meta.candidate_evaluations.forEach((evalItem) => {
      const tr = document.createElement("tr");
      const isSelected = evalItem.k === chosenK;
      if (isSelected) {
        tr.className = "selected-k";
      }
      tr.innerHTML = `
        <td>K = ${evalItem.k}</td>
        <td>${evalItem.silhouette.toFixed(4)}</td>
        <td>${evalItem.inertia.toFixed(2)}</td>
        <td>
          <span class="decision-badge ${isSelected ? "selected" : "candidate"}">
            ${isSelected ? "Selected (Optimal Silhouette)" : "Evaluated"}
          </span>
        </td>
      `;
      kEvaluationTbody.appendChild(tr);
    });
  }

  // Modals for Privacy & Terms
  function initModals() {
    linkPrivacy.addEventListener("click", (e) => {
      e.preventDefault();
      showModal("Privacy Policy", getPrivacyText());
    });

    linkTerms.addEventListener("click", (e) => {
      e.preventDefault();
      showModal("Terms & Conditions", getTermsText());
    });

    modalClose.addEventListener("click", closeModal);
    modalDismiss.addEventListener("click", closeModal);
    modalContainer.addEventListener("click", (e) => {
      if (e.target === modalContainer) {
        closeModal();
      }
    });

    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && !modalContainer.classList.contains("hidden")) {
        closeModal();
      }
    });
  }

  function showModal(title, contentHtml) {
    modalTitle.textContent = title;
    modalBody.innerHTML = contentHtml;
    modalContainer.classList.remove("hidden");
  }

  function closeModal() {
    modalContainer.classList.add("hidden");
  }

  function getPrivacyText() {
    return `
      <h4>1. Prototype Nature &amp; Data Processing</h4>
      <p>This application is an educational engineering hackathon prototype designed to demonstrate unsupervised machine learning (KMeans clustering) for audience segmentation and content personalization.</p>
      
      <h4>2. Synthetic Development Data</h4>
      <p>All training data and default demonstration profiles are synthetically generated using fixed random seeds. No personally identifiable information (PII) or real consumer viewing histories are processed, collected, or stored.</p>
      
      <h4>3. Client Input Handling</h4>
      <p>Inputs submitted via the "Analyze Viewer" interface (such as User ID and watch metrics) are processed strictly in-memory by the backend API to calculate cluster distances and catalog matches. Submitted parameters are neither persisted to permanent storage nor shared with external entities.</p>
      
      <h4>4. External Services</h4>
      <p>This service operates entirely locally or within containerized boundaries. No external paid AI, cloud analytics, or third-party tracking services are invoked.</p>
    `;
  }

  function getTermsText() {
    return `
      <h4>1. Demonstration Purpose</h4>
      <p>This software is provided solely for demonstration and evaluation purposes as part of a student hackathon arena showcase. It is not licensed for production commercial deployment.</p>
      
      <h4>2. Explainable Recommendation Engine</h4>
      <p>Content recommendations are computed via lightweight heuristic scoring matching predicted cluster characteristics and catalog attributes. Recommendations do not represent commercial endorsements, professional ratings, or user review aggregates.</p>
      
      <h4>3. Warranty &amp; Reliability</h4>
      <p>The service is provided "as is" without warranty of any kind. Operational readiness depends on containerized pipeline execution and persisted model artifacts.</p>
    `;
  }

  function escapeHtml(str) {
    if (str === null || str === undefined) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
});
