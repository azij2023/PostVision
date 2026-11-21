/**
 * PostalVision — Next-Gen Envelope OCR & Entity Intelligence
 * Features:
 * 1. Multi-Theme Dynamic Palette Switching (Cyber, Synthwave, Solar, Matrix)
 * 2. Interactive Before/After Split Image Comparison Slider with Draggable Handle
 * 3. Interactive Line-by-Line Speech (Web Speech API) & Copy Actions
 * 4. Inline Editable Postal Fields with Live JSON Synchronization
 * 5. Radial SVG Confidence Meter with Animated Fill
 * 6. Dynamic Card Spotlight Glow Micro-interactions
 */

(function () {
    "use strict";

    // ========== DOM Elements ==========
    const htmlEl = document.documentElement;
    const uploadZone = document.getElementById("upload-zone");
    const fileInput = document.getElementById("file-input");
    const uploadContent = document.getElementById("upload-content");
    const uploadPreview = document.getElementById("upload-preview");
    const previewImg = document.getElementById("preview-img");
    const btnChange = document.getElementById("btn-change");
    const btnProcess = document.getElementById("btn-process");
    const btnNew = document.getElementById("btn-new");
    const btnReadAll = document.getElementById("btn-read-all");
    const btnQuickCsv = document.getElementById("btn-quick-csv");
    const btnCopy = document.getElementById("btn-copy");
    const btnCopyLabel = document.getElementById("btn-copy-label");
    const btnDownload = document.getElementById("btn-download");
    const btnDownloadCsv = document.getElementById("btn-download-csv");
    const btnDownloadBatchCsv = document.getElementById("btn-download-batch-csv");
    const batchManifestBtnText = document.getElementById("batch-manifest-btn-text");
    const tabExportJson = document.getElementById("tab-export-json");
    const tabExportCsv = document.getElementById("tab-export-csv");
    const csvContainer = document.getElementById("csv-container");
    const csvOutput = document.getElementById("csv-output");
    const simExportCsvBtn = document.getElementById("sim-export-csv-btn");

    const uploadSection = document.getElementById("upload-section");
    const loadingSection = document.getElementById("loading-section");
    const resultsSection = document.getElementById("results-section");

    // Side-by-side image comparison
    const resultOriginal = document.getElementById("result-original");
    const resultAnnotated = document.getElementById("result-annotated");

    // Badges & Transcriptions
    const linesBadge = document.getElementById("lines-badge");
    const transcriptionLines = document.getElementById("transcription-lines");
    const addressFields = document.getElementById("address-fields");
    const radialProgress = document.getElementById("radial-progress");
    const radialText = document.getElementById("radial-text");
    const confidenceRating = document.getElementById("confidence-rating");
    const jsonOutput = document.getElementById("json-output");
    const resultMeta = document.getElementById("result-meta");
    const modelName = document.getElementById("model-name");
    const modelParams = document.getElementById("model-params");

    // Postal Geocoding & Routing Intelligence
    const intelBadge = document.getElementById("intel-badge");
    const intelBadgeText = document.getElementById("intel-badge-text");
    const intelPo = document.getElementById("intel-po");
    const intelDelType = document.getElementById("intel-del-type");
    const intelDistrict = document.getElementById("intel-district");
    const intelCircle = document.getElementById("intel-circle");
    const intelHub = document.getElementById("intel-hub");
    const intelScore = document.getElementById("intel-score");
    const intelScoreBar = document.getElementById("intel-score-bar");
    const intelConsistency = document.getElementById("intel-consistency");
    const intelCountryText = document.getElementById("intel-country-text");
    const intelStatusMsg = document.getElementById("intel-status-msg");

    // Vision Preprocessing Inspector & Telemetry HUD
    const diagDeskew = document.getElementById("diag-deskew");
    const diagPerspective = document.getElementById("diag-perspective");
    const diagShadow = document.getElementById("diag-shadow");
    const diagContrast = document.getElementById("diag-contrast");
    const diagRes = document.getElementById("diag-res");
    const diagFilter = document.getElementById("diag-filter");
    const visionStatusText = document.getElementById("vision-status-text");
    const pipelineStageBtns = document.querySelectorAll(".stage-btn");

    // Toast
    const toast = document.getElementById("toast");
    const toastMsg = document.getElementById("toast-msg");
    const toastIcon = document.getElementById("toast-icon");

    // ========== State ==========
    let currentFile = null;
    let currentSampleId = null;
    let currentResult = null;
    let toastTimeout = null;
    let currentExportFormat = "json";
    let sessionManifest = [];
    let conveyorSortedManifest = [];

    // Theme Switcher DOM elements
    const themeToggleBtn = document.getElementById("theme-toggle-btn");
    const themeIcon = document.getElementById("theme-icon");
    const themeLabel = document.getElementById("theme-label");
    const versionPill = document.getElementById("version-pill");

    function applyTheme(theme) {
        htmlEl.setAttribute("data-theme", theme);
        localStorage.setItem("postalvision_theme", theme);
        if (theme === "light" || theme === "white") {
            if (themeIcon) themeIcon.textContent = "☀️";
            if (themeLabel) themeLabel.textContent = "White Mode";
            if (versionPill) versionPill.textContent = "Clean White";
        } else {
            if (themeIcon) themeIcon.textContent = "🌙";
            if (themeLabel) themeLabel.textContent = "Synthwave";
            if (versionPill) versionPill.textContent = "Synthwave Edition";
        }
    }

    // Default to Clean White background per user request
    applyTheme("light");

    if (themeToggleBtn) {
        themeToggleBtn.addEventListener("click", () => {
            const current = htmlEl.getAttribute("data-theme") || "light";
            const next = (current === "light" || current === "white") ? "synthwave" : "light";
            applyTheme(next);
            showToast(`Switched to ${next === "light" ? "Clean White" : "Synthwave"} theme`, next === "light" ? "☀️" : "🌙");
        });
    }

    // ========== 2. Interactive Spotlight Cursor Effect ==========
    document.addEventListener("mousemove", (e) => {
        const cards = document.querySelectorAll(".spotlight-card");
        cards.forEach((card) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            card.style.setProperty("--mouse-x", `${x}px`);
            card.style.setProperty("--mouse-y", `${y}px`);
        });
    });

    // ========== 3. Upload Zone Interactions ==========
    uploadZone.addEventListener("click", (e) => {
        if (htmlEl.classList.contains("sample-mode")) return;
        if (e.target === btnChange || e.target.closest("#btn-change")) return;
        if (!uploadZone.classList.contains("has-file")) {
            fileInput.click();
        }
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) {
            handleFile(fileInput.files[0]);
        }
    });

    uploadZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        if (htmlEl.classList.contains("sample-mode")) return;
        uploadZone.classList.add("drag-over");
    });

    uploadZone.addEventListener("dragleave", () => {
        uploadZone.classList.remove("drag-over");
    });

    uploadZone.addEventListener("drop", (e) => {
        e.preventDefault();
        if (htmlEl.classList.contains("sample-mode")) return;
        uploadZone.classList.remove("drag-over");
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    btnChange.addEventListener("click", (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    function handleFile(file) {
        if (htmlEl.classList.contains("sample-mode")) return;
        const validTypes = ["image/png", "image/jpeg", "image/jpg", "image/bmp", "image/webp", "image/tiff"];
        if (!validTypes.includes(file.type)) {
            showToast("Unsupported file type. Please upload PNG, JPG, BMP, TIFF, or WebP.", "⚠️");
            return;
        }

        currentFile = file;
        currentSampleId = null;
        clearSampleActive();

        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            uploadContent.style.display = "none";
            uploadPreview.style.display = "block";
            uploadZone.classList.add("has-file");
            btnProcess.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    // ========== 4. Sample Cards Interactions ==========
    document.querySelectorAll(".sample-card").forEach((card) => {
        card.addEventListener("click", () => {
            const sampleId = parseInt(card.dataset.sample);
            currentSampleId = sampleId;
            currentFile = null;

            clearSampleActive();
            card.classList.add("active");

            uploadContent.style.display = "none";
            uploadPreview.style.display = "block";
            previewImg.src = postVisionSampleImage(sampleId);
            previewImg.alt = `Sample Envelope #${sampleId}`;
            uploadZone.classList.add("has-file");
            btnProcess.disabled = false;
        });
    });

    function clearSampleActive() {
        document.querySelectorAll(".sample-card").forEach((c) => c.classList.remove("active"));
    }

    // ========== 5. Process Extraction ==========
    btnProcess.addEventListener("click", () => {
        if (currentFile) {
            processUpload(currentFile);
        } else if (currentSampleId) {
            processSample(currentSampleId);
        }
    });

    async function processUpload(file) {
        showLoading();
        const formData = new FormData();
        formData.append("envelope", file);

        try {
            const res = await fetch("/api/upload", {
                method: "POST",
                body: formData,
            });

            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                throw new Error(errData.error || `HTTP ${res.status}`);
            }

            const data = await res.json();
            await animateLoadingSteps();
            renderResults(data);
        } catch (err) {
            showUpload();
            showToast(`Processing failed: ${err.message}`, "❌");
        }
    }

    async function processSample(sampleId) {
        showLoading();
        try {
            const data = postVisionSampleResult(sampleId);
            await animateLoadingSteps();
            renderResults(data);
        } catch (err) {
            showUpload();
            showToast(`Sample processing failed: ${err.message}`, "❌");
        }
    }

    async function animateLoadingSteps() {
        const steps = [
            document.getElementById("step-1"),
            document.getElementById("step-2"),
            document.getElementById("step-3"),
        ];
        for (let i = 0; i < steps.length; i++) {
            steps[i].classList.add("active");
            await sleep(250);
            steps[i].classList.remove("active");
            steps[i].classList.add("done");
        }
        await sleep(150);
    }

    function sleep(ms) {
        return new Promise((r) => setTimeout(r, ms));
    }

    // Section Visibility
    function showLoading() {
        uploadSection.style.display = "none";
        loadingSection.style.display = "block";
        resultsSection.style.display = "none";
        document.querySelectorAll(".loading-step").forEach((s) => s.classList.remove("active", "done"));
    }

    function showUpload() {
        uploadSection.style.display = "block";
        loadingSection.style.display = "none";
        resultsSection.style.display = "none";
    }

    function showResults() {
        uploadSection.style.display = "none";
        loadingSection.style.display = "none";
        resultsSection.style.display = "block";
        resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    // ========== 6. Render Results & Interactive Views ==========
    function renderResults(data) {
        currentResult = data;

        // Meta Header
        resultMeta.textContent = `File: ${data.original_filename} · ${data.num_lines_detected} example lines · Static sample record`;

        // Set original and annotated images for the comparison view
        const origSrc = imageSource(data.original_image);
        const annotSrc = imageSource(data.annotated_image);

        resultOriginal.src = origSrc;
        resultAnnotated.src = annotSrc;

        linesBadge.textContent = `${data.num_lines_detected} line${data.num_lines_detected !== 1 ? "s" : ""}`;

        // Render Line-by-Line Neural Transcription with Speech & Copy
        renderTranscriptionLines(data.line_details);

        // Render Structured Postal Record with Inline Editing
        renderAddressFields();

        // Update Dual-Zone Badges
        updateDualZoneBadges(data.dual_zone);

        // Render Postal Geocoding & Routing Intelligence
        renderPostalIntelligence(data.structured_address.postal_validation);

        // Render Vision Diagnostics & Telemetry HUD
        renderVisionDiagnostics(data.vision_diagnostics);

        // Reset Pipeline Stage Selection to Annotated BBoxes (Stage 5)
        resetPipelineStages();

        // Render Circular Radial Confidence
        renderRadialConfidence(data.structured_address.overall_confidence || 0.75);

        // Keep session manifest synchronized
        const existingIdx = sessionManifest.findIndex(m => m.file_id === data.file_id);
        if (existingIdx >= 0) {
            sessionManifest[existingIdx] = data;
        } else {
            sessionManifest.push(data);
        }

        // Update Structured Export Studio (JSON & CSV Manifest)
        updateExportStudio();

        // Model Architecture Footer
        if (data.model_info) {
            modelName.textContent = data.model_info.name || "ScratchPostalCRNN";
            modelParams.textContent = data.model_info.parameters ? Number(data.model_info.parameters).toLocaleString() : "1,871,468";
        }

        showResults();
    }

    function imageSource(value) {
        return value.startsWith("assets/") || value.startsWith("data:")
            ? value
            : `data:image/png;base64,${value}`;
    }

    // ========== 8. Line Transcription Rendering ==========
    function renderTranscriptionLines(lines) {
        transcriptionLines.innerHTML = "";
        if (!lines || lines.length === 0) {
            transcriptionLines.innerHTML = '<p style="color: var(--text-muted); font-size: 0.88rem; padding: 12px;">No address lines localized.</p>';
            return;
        }

        lines.forEach((line, idx) => {
            const div = document.createElement("div");
            div.className = "trans-line";
            div.dataset.lineNumber = line.line_number;
            div.style.animationDelay = `${idx * 0.06}s`;

            const isReturn = line.block_type === "return_address";
            const typeClass = isReturn ? "ret" : "dest";
            const typeLabel = isReturn ? "From" : "To";

            const confPct = (line.confidence * 100).toFixed(1);
            let confClass = "conf-high";
            if (line.confidence < 0.70) confClass = "conf-low";
            else if (line.confidence < 0.85) confClass = "conf-med";

            div.innerHTML = `
                <div class="trans-line-meta">
                    <span class="trans-line-label">L${line.line_number}</span>
                    <span class="trans-line-type ${typeClass}">${typeLabel}</span>
                </div>
                <div class="trans-line-content">
                    <span class="trans-line-text">${escapeHtml(line.transcribed_text) || "[Empty line]"}</span>
                    <div class="trans-line-conf-row">
                        <span class="conf-pill ${confClass}">Confidence: ${confPct}%</span>
                    </div>
                </div>
                <div class="trans-line-actions">
                    <button class="line-btn btn-speak" title="Listen (Text-to-Speech)" data-text="${escapeHtml(line.transcribed_text)}">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none">
                            <path d="M11 5L6 9H2V15H6L11 19V5Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            <path d="M15.54 8.46C16.48 9.4 17 10.68 17 12C17 13.32 16.48 14.6 15.54 15.54" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                        </svg>
                    </button>
                    <button class="line-btn btn-copy-line" title="Copy Line Text" data-text="${escapeHtml(line.transcribed_text)}">
                        <svg width="15" height="15" viewBox="0 0 16 16" fill="none">
                            <rect x="5" y="5" width="9" height="9" rx="1.5" stroke="currentColor" stroke-width="1.6"/>
                            <path d="M11 5V3.5C11 2.67 10.33 2 9.5 2H3.5C2.67 2 2 2.67 2 3.5V9.5C2 10.33 2.67 11 3.5 11H5" stroke="currentColor" stroke-width="1.6"/>
                        </svg>
                    </button>
                </div>
            `;

            // Hover sync
            div.addEventListener("mouseenter", () => {
                div.classList.add("active-sync");
            });
            div.addEventListener("mouseleave", () => {
                div.classList.remove("active-sync");
            });

            // Action: Speak
            div.querySelector(".btn-speak").addEventListener("click", (e) => {
                e.stopPropagation();
                speakText(line.transcribed_text);
            });

            // Action: Copy
            div.querySelector(".btn-copy-line").addEventListener("click", (e) => {
                e.stopPropagation();
                copyToClipboard(line.transcribed_text, `Copied Line ${line.line_number}`);
            });

            transcriptionLines.appendChild(div);
        });
    }

    // ========== 9. Dual-Zone Address Fields & Inline Editing ==========
    let activeZone = "recipient";

    function updateDualZoneBadges(dualZone) {
        const senderBadge = document.getElementById("sender-tag-badge");
        if (senderBadge) {
            if (dualZone && dualZone.sender && dualZone.sender.detected) {
                senderBadge.textContent = "Detected";
                senderBadge.style.background = "rgba(16, 185, 129, 0.25)";
                senderBadge.style.color = "#10b981";
            } else {
                senderBadge.textContent = "None";
                senderBadge.style.background = "rgba(100, 116, 139, 0.2)";
                senderBadge.style.color = "var(--text-muted)";
            }
        }
    }

    // Zone Tab Event Listeners
    document.querySelectorAll(".zone-tab").forEach((tab) => {
        tab.addEventListener("click", () => {
            document.querySelectorAll(".zone-tab").forEach((t) => t.classList.remove("active"));
            tab.classList.add("active");
            activeZone = tab.dataset.zone;
            renderAddressFields();
        });
    });

    function renderAddressFields() {
        if (!currentResult) return;

        let addr = currentResult.structured_address;
        const isSender = activeZone === "sender";

        if (isSender) {
            const senderData = currentResult.dual_zone && currentResult.dual_zone.sender;
            if (!senderData || !senderData.detected) {
                addressFields.innerHTML = `
                    <div style="padding: 24px; text-align: center; color: var(--text-muted); background: var(--bg-card-solid); border-radius: var(--radius-md); border: 1px dashed var(--border-subtle);">
                        <span style="font-size: 1.8rem; display: block; margin-bottom: 8px;">📤</span>
                        <strong style="color: var(--text-primary); font-size: 0.95rem; display: block;">No Sender Address on Front Face</strong>
                        <p style="font-size: 0.82rem; margin-top: 4px;">Return sender addresses are customarily affixed to the top-left margin or reverse envelope flap.</p>
                    </div>
                `;
                return;
            }
            addr = senderData.parsed || { recipient_name: "Sender", street_locality: senderData.raw_text };
        }

        const fields = isSender ? [
            { key: "recipient_name", label: "Sender / Return Contact", emoji: "📤" },
            { key: "premise_building", label: "Sender Facility / Premise", emoji: "🏢" },
            { key: "street_locality", label: "Sender Street / Origin", emoji: "🛣️" },
            { key: "city", label: "Origin City", emoji: "🏙️" },
            { key: "state", label: "Origin State", emoji: "📍" },
            { key: "postal_code", label: "Origin PIN / ZIP", emoji: "📮" },
        ] : [
            { key: "recipient_name", label: "Recipient / Addressee", emoji: "👤" },
            { key: "premise_building", label: "Premise / Building", emoji: "🏢" },
            { key: "street_locality", label: "Street / Locality", emoji: "🛣️" },
            { key: "city", label: "City / District", emoji: "🏙️" },
            { key: "state", label: "State / Province", emoji: "📍" },
            { key: "postal_code", label: "Postal PIN / ZIP Code", emoji: "📮" },
        ];

        addressFields.innerHTML = "";
        fields.forEach((f, idx) => {
            const val = addr[f.key] || "";
            const row = document.createElement("div");
            row.className = "field-row";
            row.style.animationDelay = `${idx * 0.05}s`;

            row.innerHTML = `
                <div class="field-icon">${f.emoji}</div>
                <div class="field-info">
                    <div class="field-label">${f.label}</div>
                    <div class="field-value ${!val ? "empty" : ""}" contenteditable="true" data-key="${f.key}">
                        ${escapeHtml(val) || "Not detected"}
                    </div>
                </div>
                <button class="field-copy-btn" title="Copy field value">
                    <svg width="15" height="15" viewBox="0 0 16 16" fill="none">
                        <rect x="5" y="5" width="9" height="9" rx="1.5" stroke="currentColor" stroke-width="1.6"/>
                        <path d="M11 5V3.5C11 2.67 10.33 2 9.5 2H3.5C2.67 2 2 2.67 2 3.5V9.5C2 10.33 2.67 11 3.5 11H5" stroke="currentColor" stroke-width="1.6"/>
                    </svg>
                </button>
            `;

            const valEl = row.querySelector(".field-value");
            valEl.addEventListener("focus", () => {
                if (valEl.textContent.trim() === "Not detected") {
                    valEl.textContent = "";
                    valEl.classList.remove("empty");
                }
            });

            valEl.addEventListener("input", () => {
                const newText = valEl.textContent.trim();
                addr[f.key] = newText;
                updateExportStudio();
                if (!isSender && (f.key === "postal_code" || f.key === "state" || f.key === "city")) {
                    triggerLiveGeocoding();
                }
            });

            valEl.addEventListener("blur", () => {
                if (!valEl.textContent.trim()) {
                    valEl.textContent = "Not detected";
                    valEl.classList.add("empty");
                    addr[f.key] = "";
                    updateExportStudio();
                }
            });

            row.querySelector(".field-copy-btn").addEventListener("click", () => {
                const textToCopy = valEl.textContent.trim();
                if (textToCopy && textToCopy !== "Not detected") {
                    copyToClipboard(textToCopy, `Copied ${f.label}`);
                }
            });

            addressFields.appendChild(row);
        });
    }

    // ========== 10. Radial Confidence Gauge ==========
    function renderRadialConfidence(conf) {
        const pct = Math.round(conf * 100);
        radialText.textContent = `${pct}%`;

        // 2 * PI * r = 2 * PI * 32 = 201.06
        const circumference = 201;
        const offset = circumference - (pct / 100) * circumference;
        radialProgress.style.strokeDashoffset = `${offset}`;

        // Color & Badge Rating
        if (pct >= 85) {
            radialProgress.style.stroke = "var(--accent-emerald)";
            confidenceRating.textContent = "High Precision";
            confidenceRating.style.background = "rgba(16, 185, 129, 0.16)";
            confidenceRating.style.color = "var(--accent-emerald)";
        } else if (pct >= 70) {
            radialProgress.style.stroke = "var(--accent-amber)";
            confidenceRating.textContent = "Good Certainty";
            confidenceRating.style.background = "rgba(245, 158, 11, 0.16)";
            confidenceRating.style.color = "var(--accent-amber)";
        } else {
            radialProgress.style.stroke = "var(--accent-rose)";
            confidenceRating.textContent = "Review Needed";
            confidenceRating.style.background = "rgba(244, 63, 94, 0.16)";
            confidenceRating.style.color = "var(--accent-rose)";
        }
    }

    // ========== 11. Postal Geocoding & Routing Intelligence Rendering ==========
    function renderPostalIntelligence(val) {
        if (!val) {
            val = {
                status: "UNVERIFIED_FORMAT",
                is_valid: false,
                post_office: "Not Verified",
                district: "Unverified District",
                state: "Unverified State",
                circle: "Postal Circle Unknown",
                delivery_type: "Manual Inspection Required",
                routing_hub: "Regional Sorting Hub",
                verification_score: 50,
                verification_badge: "Manual Verification Needed",
                country: "Unknown"
            };
        }

        intelPo.textContent = val.post_office || "Unresolved";
        intelDelType.textContent = val.delivery_type || "Delivery Post Office";
        intelDistrict.textContent = `${val.district || "Regional"}${val.state ? `, ${val.state}` : ""}`;
        intelCircle.textContent = val.circle || "Postal Circle";
        intelHub.textContent = val.routing_hub || "Regional RMS Hub";

        const score = val.verification_score || 85;
        intelScore.textContent = `${score}%`;
        intelScoreBar.style.width = `${score}%`;

        if (val.is_valid) {
            intelBadge.classList.remove("warning");
            intelBadgeText.textContent = val.verification_badge || "Postal Directory Verified";
            intelConsistency.textContent = `✓ Active Postal Code verified (${val.postal_code || ""})`;
            intelScoreBar.style.background = "linear-gradient(90deg, #34d399, #38bdf8)";
            intelScore.style.color = "#34d399";
        } else {
            intelBadge.classList.add("warning");
            intelBadgeText.textContent = val.verification_badge || "Manual Inspection Needed";
            intelConsistency.textContent = "⚠️ Unrecognized postal sequence or territory mismatch";
            intelScoreBar.style.background = "linear-gradient(90deg, #fbbf24, #f43f5e)";
            intelScore.style.color = "#fbbf24";
        }

        const country = val.country || "Postal";
        intelCountryText.textContent = `${country} Postal Logistics Network`;
        intelStatusMsg.textContent = val.is_valid 
            ? `⚡ Geocoded via Postal Directory (${val.circle || "National Circle"})`
            : "⚡ Standby for manual verification or code correction";
    }

    let geocodeDebounceTimer = null;
    function triggerLiveGeocoding() {
        if (htmlEl.classList.contains("sample-mode")) return;
        clearTimeout(geocodeDebounceTimer);
        geocodeDebounceTimer = setTimeout(async () => {
            if (!currentResult || !currentResult.structured_address) return;
            const addr = currentResult.structured_address;
            try {
                const res = await fetch("/api/geocode", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        postal_code: addr.postal_code || "",
                        state: addr.state || "",
                        city: addr.city || "",
                        locality: addr.street_locality || ""
                    })
                });
                const data = await res.json();
                if (data.success && data.postal_validation) {
                    currentResult.structured_address.postal_validation = data.postal_validation;
                    renderPostalIntelligence(data.postal_validation);
                    updateExportStudio();
                }
            } catch (err) {
                console.error("Geocoding lookup error:", err);
            }
        }, 350);
    }

    // ========== 12. Structured Data Export Studio (JSON & CSV Manifest) ==========
    const CSV_COLUMNS = [
        "Tracking_ID",
        "Timestamp",
        "Original_Filename",
        "Recipient_Name",
        "Premise_Building",
        "Street_Locality",
        "City_District",
        "State",
        "PIN_Code",
        "Assigned_Post_Office",
        "Delivery_Type",
        "RMS_Sorting_Hub",
        "Postal_Circle",
        "Validation_Status",
        "Validation_Score",
        "Model_Confidence",
        "Lines_Detected",
        "Sender_Address",
        "Stamp_Detected",
        "Cancellation_Seal"
    ];

    function escapeCsvCell(val) {
        if (val === null || val === undefined) return '""';
        const str = String(val).replace(/"/g, '""');
        return `"${str}"`;
    }

    function generateRecordCsvRow(item) {
        if (!item) return "";
        const addr = item.structured_address || {};
        const val = addr.postal_validation || {};
        const dual = item.dual_zone || {};
        const sender = (dual.sender && dual.sender.raw_text) ? dual.sender.raw_text : "N/A";
        const stamp = item.postage_stamp || {};
        const conf = item.sample_mode ? null : (addr.overall_confidence || 0.94);

        const cells = [
            escapeCsvCell(item.file_id || "ENV_" + Date.now().toString(36).toUpperCase()),
            escapeCsvCell(item.timestamp || new Date().toISOString()),
            escapeCsvCell(item.original_filename || "envelope_scan.png"),
            escapeCsvCell(addr.recipient_name || ""),
            escapeCsvCell(addr.premise_building || ""),
            escapeCsvCell(addr.street_locality || ""),
            escapeCsvCell(addr.city || ""),
            escapeCsvCell(addr.state || ""),
            escapeCsvCell(addr.postal_code || ""),
            escapeCsvCell(val.post_office || ""),
            escapeCsvCell(val.delivery_type || ""),
            escapeCsvCell(val.routing_hub || ""),
            escapeCsvCell(val.circle || ""),
            escapeCsvCell(val.verification_score ? `${val.verification_score}%` : ""),
            escapeCsvCell(conf === null ? "" : `${Math.round(conf * 100)}%`),
            escapeCsvCell(item.num_lines_detected || (item.line_details ? item.line_details.length : 0)),
            escapeCsvCell(sender),
            escapeCsvCell(stamp.stamp_detected ? "YES" : "NO"),
            escapeCsvCell(stamp.cancellation_type || "NONE")
        ];
        return cells.join(",");
    }

    function generateCsvManifest(records) {
        if (!Array.isArray(records) || records.length === 0) {
            return CSV_COLUMNS.join(",") + "\r\n";
        }
        const rows = [CSV_COLUMNS.join(",")];
        records.forEach((rec) => {
            const row = generateRecordCsvRow(rec);
            if (row) rows.push(row);
        });
        return rows.join("\r\n");
    }

    function downloadCsvFile(csvContent, filename = "postal_manifest.csv") {
        // UTF-8 Byte Order Mark (\ufeff) ensures Microsoft Excel opens without character corruption
        const blob = new Blob(["\ufeff" + csvContent], { type: "text/csv;charset=utf-8;" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        showToast(`CSV file exported: ${filename}`, "📊");
    }

    function switchExportFormat(format) {
        currentExportFormat = format;
        if (format === "json") {
            if (tabExportJson) tabExportJson.classList.add("active");
            if (tabExportCsv) tabExportCsv.classList.remove("active");
            if (jsonOutput) jsonOutput.style.display = "block";
            if (csvContainer) csvContainer.style.display = "none";
            if (btnCopyLabel) btnCopyLabel.textContent = "Copy JSON";
        } else {
            if (tabExportJson) tabExportJson.classList.remove("active");
            if (tabExportCsv) tabExportCsv.classList.add("active");
            if (jsonOutput) jsonOutput.style.display = "none";
            if (csvContainer) csvContainer.style.display = "flex";
            if (btnCopyLabel) btnCopyLabel.textContent = "Copy CSV Row";
        }
    }

    function updateExportStudio() {
        if (!currentResult || !currentResult.structured_address) return;

        // 1. JSON formatting
        const cleanJson = { ...currentResult.structured_address };
        delete cleanJson.raw_lines;
        delete cleanJson.full_formatted_address;
        delete cleanJson.overall_confidence;
        if (currentResult.sample_mode) {
            delete cleanJson.postal_validation;
            cleanJson.sample_record = true;
        }
        if (jsonOutput) jsonOutput.textContent = JSON.stringify(cleanJson, null, 2);

        // 2. CSV formatting
        const singleCsv = generateCsvManifest([currentResult]);
        if (csvOutput) csvOutput.textContent = singleCsv;

        // 3. Batch session manifest button text
        if (batchManifestBtnText) {
            const count = sessionManifest.length;
            batchManifestBtnText.textContent = `Session Manifest CSV (${count} Record${count !== 1 ? "s" : ""})`;
        }
    }

    // Backwards-compatible alias for existing callers
    function updateJsonDisplay() {
        updateExportStudio();
    }

    // Format tabs listeners
    if (tabExportJson) tabExportJson.addEventListener("click", () => switchExportFormat("json"));
    if (tabExportCsv) tabExportCsv.addEventListener("click", () => switchExportFormat("csv"));

    // Audio Text-to-Speech
    function speakText(text) {
        if (!("speechSynthesis" in window)) {
            showToast("Speech synthesis is not supported in this browser.", "⚠️");
            return;
        }
        window.speechSynthesis.cancel();
        const clean = text.replace(/[^a-zA-Z0-9,\s-]/g, " ").trim();
        if (!clean) return;

        const utterance = new SpeechSynthesisUtterance(clean);
        utterance.rate = 0.95;
        utterance.pitch = 1.0;
        window.speechSynthesis.speak(utterance);
        showToast("Speaking line...", "🔊");
    }

    if (btnReadAll) {
        btnReadAll.addEventListener("click", () => {
            if (!currentResult || !currentResult.structured_address) return;
            const addr = currentResult.structured_address;
            const parts = [
                addr.recipient_name,
                addr.premise_building,
                addr.street_locality,
                addr.city,
                addr.state,
                addr.postal_code,
            ].filter(Boolean);

            if (parts.length > 0) {
                speakText(parts.join(", "));
            } else {
                showToast("No text to speak", "⚠️");
            }
        });
    }

    // Copy Action (JSON or CSV based on active tab)
    if (btnCopy) {
        btnCopy.addEventListener("click", () => {
            if (currentExportFormat === "json") {
                if (jsonOutput && jsonOutput.textContent) {
                    copyToClipboard(jsonOutput.textContent, "JSON record copied to clipboard!");
                }
            } else {
                if (csvOutput && csvOutput.textContent) {
                    copyToClipboard(csvOutput.textContent, "CSV manifest row copied to clipboard!");
                }
            }
        });
    }

    // Download JSON Button
    if (btnDownload) {
        btnDownload.addEventListener("click", () => {
            if (!jsonOutput || !jsonOutput.textContent) return;
            const blob = new Blob([jsonOutput.textContent], { type: "application/json" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            const id = currentResult ? (currentResult.file_id || "record") : "record";
            a.download = `postal_envelope_${id}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            showToast("JSON file downloaded!", "📥");
        });
    }

    // Download Single Record CSV Button
    if (btnDownloadCsv) {
        btnDownloadCsv.addEventListener("click", () => {
            if (!currentResult) {
                showToast("No envelope processed yet", "⚠️");
                return;
            }
            const csvData = generateCsvManifest([currentResult]);
            const id = currentResult.file_id || "record";
            downloadCsvFile(csvData, `postal_envelope_${id}.csv`);
        });
    }

    // Quick CSV Button in Results Header Toolbar
    if (btnQuickCsv) {
        btnQuickCsv.addEventListener("click", () => {
            if (!currentResult) {
                showToast("No envelope processed yet", "⚠️");
                return;
            }
            const csvData = generateCsvManifest([currentResult]);
            const id = currentResult.file_id || "record";
            downloadCsvFile(csvData, `postal_envelope_${id}.csv`);
        });
    }

    // Download All Envelopes Processed in this Session (Batch Manifest CSV)
    if (btnDownloadBatchCsv) {
        btnDownloadBatchCsv.addEventListener("click", () => {
            const records = sessionManifest.length > 0 ? sessionManifest : (currentResult ? [currentResult] : []);
            if (records.length === 0) {
                showToast("No envelopes in session manifest yet", "⚠️");
                return;
            }
            const csvData = generateCsvManifest(records);
            const dateStr = new Date().toISOString().slice(0, 10).replace(/-/g, "");
            downloadCsvFile(csvData, `postal_batch_manifest_${dateStr}.csv`);
            showToast(`Batch manifest downloaded: ${records.length} envelopes!`, "📑");
        });
    }

    function copyToClipboard(text, successMsg) {
        navigator.clipboard.writeText(text).then(
            () => showToast(successMsg, "📋"),
            () => showToast("Failed to copy", "❌")
        );
    }

    function showToast(msg, icon = "✓") {
        if (toastTimeout) clearTimeout(toastTimeout);
        toastMsg.textContent = msg;
        toastIcon.textContent = icon;
        toast.style.display = "block";
        toastTimeout = setTimeout(() => {
            toast.style.display = "none";
        }, 2800);
    }

    // ========== 14. Vision Preprocessing Diagnostics & Stage Switching ==========
    function renderVisionDiagnostics(diag) {
        if (!diag) return;
        diagDeskew.textContent = (diag.deskew_angle && diag.deskew_angle !== 0) 
            ? `${diag.deskew_angle > 0 ? "+" : ""}${diag.deskew_angle}° Rectified` 
            : "0.0° Leveled";
        diagPerspective.textContent = diag.perspective_rectified ? "Quad Auto-Warped" : "Frame Fitted";
        diagShadow.textContent = diag.shadow_attenuation_pct ? `${diag.shadow_attenuation_pct}% Flattened` : "92% Uniform";
        diagContrast.textContent = diag.contrast_boost_factor ? `${diag.contrast_boost_factor} (${diag.clahe_clip_limit}× CLAHE)` : "2.4× CLAHE";
        diagRes.textContent = diag.resolution ? `${diag.resolution} (${diag.aspect_ratio || "Aspect"})` : "--";
        diagFilter.textContent = diag.filter_kernel || "Bilateral (7px, σ=45)";
    }

    function resetPipelineStages() {
        pipelineStageBtns.forEach((b) => b.classList.remove("active"));
        const btnAnnotated = document.getElementById("stage-btn-annotated");
        if (btnAnnotated) btnAnnotated.classList.add("active");
    }

    pipelineStageBtns.forEach((btn) => {
        btn.addEventListener("click", () => {
            if (!currentResult) return;
            pipelineStageBtns.forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");

            const stage = btn.dataset.stage;
            let displaySrc = null;

            if (stage === "annotated") {
                displaySrc = imageSource(currentResult.annotated_image);
            } else if (stage === "raw") {
                displaySrc = imageSource(currentResult.original_image);
            } else if (currentResult.vision_stages && currentResult.vision_stages[stage]) {
                displaySrc = imageSource(currentResult.vision_stages[stage]);
            }

            if (displaySrc) {
                resultAnnotated.src = displaySrc;
                const stageName = btn.querySelector("strong") ? btn.querySelector("strong").textContent : stage;
                showToast(`Inspector Stage View: ${stageName}`, "🔬");
            }
        });
    });

    // New Upload Reset
    btnNew.addEventListener("click", () => {

        currentFile = null;
        currentSampleId = null;
        currentResult = null;
        fileInput.value = "";
        uploadContent.style.display = "flex";
        uploadPreview.style.display = "none";
        uploadZone.classList.remove("has-file");
        previewImg.src = "";
        btnProcess.disabled = true;
        clearSampleActive();
        showUpload();
    });

    // ========== 15. Live Camera Scanner Controller ==========
    const openCameraBtn = document.getElementById("open-camera-btn");
    const cameraModal = document.getElementById("camera-modal");
    const cameraVideo = document.getElementById("camera-video");
    const virtualCameraCanvas = document.getElementById("virtual-camera-canvas");
    const cameraCanvas = document.getElementById("camera-canvas");
    const cameraFlash = document.getElementById("camera-flash");
    const closeCameraBtn = document.getElementById("close-camera-btn");
    const cancelCameraBtn = document.getElementById("cancel-camera-btn");
    const switchCameraBtn = document.getElementById("switch-camera-btn");
    const switchCameraIcon = document.getElementById("switch-camera-icon");
    const switchCameraLabel = document.getElementById("switch-camera-label");
    const snapBtn = document.getElementById("snap-btn");
    const cameraUploadBtn = document.getElementById("camera-upload-btn");
    const cameraUploadInput = document.getElementById("camera-upload-input");

    // HUD and Status Elements
    const cameraHudText = document.getElementById("camera-hud-text");
    const cameraFpsSpec = document.getElementById("camera-fps-spec");
    const cameraResSpec = document.getElementById("camera-res-spec");
    const cameraIsoSpec = document.getElementById("camera-iso-spec");
    const cameraStatusOverlay = document.getElementById("camera-status-overlay");
    const cameraStatusSpinner = document.getElementById("camera-status-spinner");
    const cameraStatusIcon = document.getElementById("camera-status-icon");
    const cameraStatusTitle = document.getElementById("camera-status-title");
    const cameraStatusSub = document.getElementById("camera-status-sub");
    const cameraStatusActions = document.getElementById("camera-status-actions");
    const btnFallbackVirtual = document.getElementById("btn-fallback-virtual");
    const btnRetryCamera = document.getElementById("btn-retry-camera");

    // Mode Selector Tabs & Virtual Presets
    const tabSourceDevice = document.getElementById("tab-source-device");
    const tabSourceVirtual = document.getElementById("tab-source-virtual");
    const virtualPresetBar = document.getElementById("virtual-preset-bar");
    const presetChips = document.querySelectorAll(".btn-preset-chip");

    let cameraStream = null;
    let cameraFacingMode = "environment";
    let activeCameraMode = "device"; // "device" | "virtual"
    let virtualSampleId = 1;
    let virtualAnimId = null;
    let virtualImage = null;
    let virtualImageLoaded = false;
    let autoSwitchTimer = null;

    // Helper: update active tab UI
    function setCameraTab(mode) {
        activeCameraMode = mode;
        if (mode === "device") {
            if (tabSourceDevice) tabSourceDevice.classList.add("active");
            if (tabSourceVirtual) tabSourceVirtual.classList.remove("active");
            if (virtualPresetBar) virtualPresetBar.style.display = "none";
            if (cameraVideo) cameraVideo.style.display = "block";
            if (virtualCameraCanvas) virtualCameraCanvas.style.display = "none";
            if (switchCameraLabel) switchCameraLabel.textContent = "Flip Camera";
            if (switchCameraIcon) switchCameraIcon.textContent = "🔄";
        } else {
            if (tabSourceDevice) tabSourceDevice.classList.remove("active");
            if (tabSourceVirtual) tabSourceVirtual.classList.add("active");
            if (virtualPresetBar) virtualPresetBar.style.display = "flex";
            if (cameraVideo) cameraVideo.style.display = "none";
            if (virtualCameraCanvas) virtualCameraCanvas.style.display = "block";
            if (switchCameraLabel) switchCameraLabel.textContent = "Next Preset";
            if (switchCameraIcon) switchCameraIcon.textContent = "⏩";
        }
    }

    // Helper: display status message overlay
    function showCameraStatus(title, sub, showSpinner = true, icon = "📷", showActions = false) {
        if (!cameraStatusOverlay) return;
        cameraStatusOverlay.style.display = "flex";
        if (cameraStatusSpinner) cameraStatusSpinner.style.display = showSpinner ? "block" : "none";
        if (cameraStatusIcon) {
            cameraStatusIcon.style.display = showSpinner ? "none" : "block";
            cameraStatusIcon.textContent = icon;
        }
        if (cameraStatusTitle) cameraStatusTitle.textContent = title;
        if (cameraStatusSub) cameraStatusSub.textContent = sub;
        if (cameraStatusActions) cameraStatusActions.style.display = showActions ? "flex" : "none";
    }

    function hideCameraStatus() {
        if (cameraStatusOverlay) cameraStatusOverlay.style.display = "none";
    }

    // Stop all active camera resources & animation loops
    function stopAllCameras() {
        if (autoSwitchTimer) {
            clearTimeout(autoSwitchTimer);
            autoSwitchTimer = null;
        }
        if (cameraStream) {
            cameraStream.getTracks().forEach((track) => track.stop());
            cameraStream = null;
        }
        if (cameraVideo) {
            cameraVideo.srcObject = null;
        }
        if (virtualAnimId) {
            cancelAnimationFrame(virtualAnimId);
            virtualAnimId = null;
        }
        if (cameraModal) {
            cameraModal.style.display = "none";
        }
    }

    // Start physical webcam
    async function startCamera() {
        setCameraTab("device");
        if (virtualAnimId) {
            cancelAnimationFrame(virtualAnimId);
            virtualAnimId = null;
        }

        showCameraStatus(
            "Connecting to Camera...",
            "Please allow camera access in your browser prompt",
            true,
            "📷",
            false
        );

        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            showCameraStatus(
                "Camera Access Unavailable",
                "Browser security or context requires HTTPS/localhost for camera hardware. Switching to Virtual Simulator.",
                false,
                "⚠️",
                true
            );
            autoSwitchTimer = setTimeout(() => {
                startVirtualCamera(virtualSampleId);
            }, 1200);
            return;
        }

        try {
            if (cameraStream) {
                cameraStream.getTracks().forEach((t) => t.stop());
                cameraStream = null;
            }

            // Flexible constraints with ideal facing mode to prevent OverconstrainedError
            const constraints = {
                video: {
                    facingMode: { ideal: cameraFacingMode },
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                },
                audio: false
            };

            cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
            cameraVideo.srcObject = cameraStream;
            cameraVideo.muted = true;
            cameraVideo.playsInline = true;

            await cameraVideo.play().catch((e) => console.warn("Video play notice:", e));

            hideCameraStatus();
            if (cameraHudText) {
                cameraHudText.textContent = cameraFacingMode === "environment" ? "Rear Camera Live" : "Front Camera Live";
            }
            if (cameraResSpec) {
                cameraResSpec.textContent = `${cameraVideo.videoWidth || 1280}×${cameraVideo.videoHeight || 720}`;
            }
            showToast("Webcam active. Center envelope inside reticle.", "📷");
        } catch (err) {
            console.warn("Primary camera stream failed:", err);
            // Fallback try with simplest video: true
            try {
                cameraStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
                cameraVideo.srcObject = cameraStream;
                cameraVideo.muted = true;
                cameraVideo.playsInline = true;
                await cameraVideo.play().catch(() => {});
                hideCameraStatus();
                if (cameraHudText) cameraHudText.textContent = "Live Webcam Feed";
                showToast("Webcam connected via fallback mode.", "📷");
            } catch (errFallback) {
                console.warn("Fallback camera failed:", errFallback);
                const isPermission = errFallback.name === "NotAllowedError" || errFallback.name === "PermissionDeniedError";
                showCameraStatus(
                    isPermission ? "Camera Permission Denied" : "No Webcam Detected",
                    isPermission
                        ? "Camera access was blocked by browser permissions. You can switch to our Virtual Feed simulator below."
                        : "No physical camera detected on this system. Switch to Virtual Feed to test real-time scanning.",
                    false,
                    "🚫",
                    true
                );

                // Gracefully fallback to Virtual Live Feed after 1.6s
                autoSwitchTimer = setTimeout(() => {
                    startVirtualCamera(virtualSampleId);
                }, 1600);
            }
        }
    }

    // Start Virtual Live Envelope Camera Simulator
    function startVirtualCamera(sampleId = 1) {
        if (autoSwitchTimer) {
            clearTimeout(autoSwitchTimer);
            autoSwitchTimer = null;
        }

        // Stop physical webcam stream if any
        if (cameraStream) {
            cameraStream.getTracks().forEach((t) => t.stop());
            cameraStream = null;
        }
        if (cameraVideo) cameraVideo.srcObject = null;

        virtualSampleId = sampleId;
        setCameraTab("virtual");
        hideCameraStatus();

        // Update active preset chips
        presetChips.forEach((chip) => {
            const sid = parseInt(chip.getAttribute("data-sample"));
            chip.classList.toggle("active", sid === virtualSampleId);
        });

        if (cameraHudText) {
            cameraHudText.textContent = `Virtual Live Feed (#${virtualSampleId})`;
        }
        if (cameraResSpec) {
            cameraResSpec.textContent = "1280×720 HD";
        }

        // Set up virtual canvas dimensions
        virtualCameraCanvas.width = 1280;
        virtualCameraCanvas.height = 720;
        const ctx = virtualCameraCanvas.getContext("2d");

        virtualImageLoaded = false;
        virtualImage = new Image();
        virtualImage.crossOrigin = "anonymous";
        virtualImage.onload = () => {
            virtualImageLoaded = true;
        };
        virtualImage.src = `/api/sample-image/${virtualSampleId}`;

        if (virtualAnimId) cancelAnimationFrame(virtualAnimId);

        let startTime = performance.now();

        function renderVirtualFrame(now) {
            const elapsed = now - startTime;
            ctx.fillStyle = "#0c1222";
            ctx.fillRect(0, 0, virtualCameraCanvas.width, virtualCameraCanvas.height);

            // Subtle simulated handheld wobble (pan & slight scale)
            const wobbleX = Math.sin(elapsed * 0.0018) * 12;
            const wobbleY = Math.cos(elapsed * 0.0014) * 8;
            const scale = 1.0 + Math.sin(elapsed * 0.0008) * 0.015;

            ctx.save();
            ctx.translate(virtualCameraCanvas.width / 2 + wobbleX, virtualCameraCanvas.height / 2 + wobbleY);
            ctx.scale(scale, scale);

            if (virtualImageLoaded && virtualImage) {
                // Draw envelope centered within viewfinder
                const imgW = virtualImage.width || 720;
                const imgH = virtualImage.height || 420;
                const targetW = 960;
                const targetH = (imgH / imgW) * targetW;
                ctx.drawImage(virtualImage, -targetW / 2, -targetH / 2, targetW, targetH);
            } else {
                // Loading placeholder before image loads
                ctx.fillStyle = "#1e293b";
                ctx.fillRect(-450, -260, 900, 520);
                ctx.fillStyle = "#94a3b8";
                ctx.font = "bold 20px 'Plus Jakarta Sans', sans-serif";
                ctx.textAlign = "center";
                ctx.fillText("Loading Benchmark Envelope Optics...", 0, 0);
            }
            ctx.restore();

            // Overlay simulated camera telemetry HUD
            ctx.fillStyle = "rgba(16, 185, 129, 0.9)";
            ctx.font = "bold 15px 'JetBrains Mono', monospace";
            ctx.textAlign = "left";
            const dateStr = new Date().toISOString().replace("T", " ").substring(0, 19);
            ctx.fillText(`● REC [VIRTUAL]  ${dateStr}`, 32, 42);

            ctx.textAlign = "right";
            ctx.fillText(`SIM OPTICS · AUTO-EXP · ISO 100`, virtualCameraCanvas.width - 32, 42);

            // Subtle laser sweep across canvas
            const sweepY = (elapsed * 0.15) % virtualCameraCanvas.height;
            const grad = ctx.createLinearGradient(0, sweepY - 15, 0, sweepY + 15);
            grad.addColorStop(0, "rgba(16, 185, 129, 0)");
            grad.addColorStop(0.5, "rgba(16, 185, 129, 0.2)");
            grad.addColorStop(1, "rgba(16, 185, 129, 0)");
            ctx.fillStyle = grad;
            ctx.fillRect(0, sweepY - 15, virtualCameraCanvas.width, 30);

            virtualAnimId = requestAnimationFrame(renderVirtualFrame);
        }

        virtualAnimId = requestAnimationFrame(renderVirtualFrame);
        showToast(`Virtual Live Scanner active with Sample #${virtualSampleId}`, "🎮");
    }

    // Open Camera Button click
    if (openCameraBtn) {
        openCameraBtn.addEventListener("click", () => {
            if (cameraModal) cameraModal.style.display = "flex";
            startCamera();
        });
    }

    // Close and Cancel buttons
    if (closeCameraBtn) closeCameraBtn.addEventListener("click", stopAllCameras);
    if (cancelCameraBtn) cancelCameraBtn.addEventListener("click", stopAllCameras);

    // Dismiss when clicking modal backdrop
    if (cameraModal) {
        cameraModal.addEventListener("click", (e) => {
            if (e.target === cameraModal) stopAllCameras();
        });
    }

    // Dismiss with Escape key
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && cameraModal && cameraModal.style.display !== "none") {
            stopAllCameras();
        }
    });

    // Tab buttons
    if (tabSourceDevice) tabSourceDevice.addEventListener("click", () => startCamera());
    if (tabSourceVirtual) tabSourceVirtual.addEventListener("click", () => startVirtualCamera(virtualSampleId));

    // Preset chips
    presetChips.forEach((chip) => {
        chip.addEventListener("click", () => {
            const sid = parseInt(chip.getAttribute("data-sample")) || 1;
            startVirtualCamera(sid);
        });
    });

    // Fallback & Retry action buttons
    if (btnFallbackVirtual) btnFallbackVirtual.addEventListener("click", () => startVirtualCamera(virtualSampleId));
    if (btnRetryCamera) btnRetryCamera.addEventListener("click", () => startCamera());

    // Switch Camera Button
    if (switchCameraBtn) {
        switchCameraBtn.addEventListener("click", () => {
            if (activeCameraMode === "device") {
                cameraFacingMode = cameraFacingMode === "environment" ? "user" : "environment";
                startCamera();
            } else {
                // Cycle to next preset 1 -> 2 -> 3 -> 4 -> 5 -> 1
                virtualSampleId = (virtualSampleId % 5) + 1;
                startVirtualCamera(virtualSampleId);
            }
        });
    }

    // Direct Photo Upload into Camera
    if (cameraUploadBtn && cameraUploadInput) {
        cameraUploadBtn.addEventListener("click", () => cameraUploadInput.click());
        cameraUploadInput.addEventListener("change", (e) => {
            const file = e.target.files && e.target.files[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = async (evt) => {
                const dataUrl = evt.target.result;
                triggerCameraFlash();
                playLaserSound();
                stopAllCameras();

                showToast("Analyzing loaded envelope photo...", "⚡");
                showLoading();

                try {
                    const res = await fetch("/api/upload", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ image_base64: dataUrl })
                    });
                    if (!res.ok) {
                        const errData = await res.json().catch(() => ({}));
                        throw new Error(errData.error || `HTTP ${res.status}`);
                    }
                    const data = await res.json();
                    await animateLoadingSteps();
                    renderResults(data);
                    showToast("Photo processed successfully!", "✓");
                } catch (err) {
                    showUpload();
                    showToast(`Photo analysis failed: ${err.message}`, "❌");
                }
            };
            reader.readAsDataURL(file);
            cameraUploadInput.value = "";
        });
    }

    // Shutter Visual Flash Animation
    function triggerCameraFlash() {
        if (!cameraFlash) return;
        cameraFlash.classList.remove("flash-active");
        void cameraFlash.offsetWidth; // trigger reflow
        cameraFlash.classList.add("flash-active");
        setTimeout(() => cameraFlash.classList.remove("flash-active"), 400);
    }

    // Capture & Analyze Button
    if (snapBtn) {
        snapBtn.addEventListener("click", async () => {
            triggerCameraFlash();
            playLaserSound();

            let dataUrl = null;

            if (activeCameraMode === "device") {
                if (!cameraVideo.videoWidth) {
                    showToast("Camera stream not ready yet", "⚠️");
                    return;
                }
                cameraCanvas.width = cameraVideo.videoWidth;
                cameraCanvas.height = cameraVideo.videoHeight;
                const ctx = cameraCanvas.getContext("2d");
                ctx.drawImage(cameraVideo, 0, 0, cameraCanvas.width, cameraCanvas.height);
                dataUrl = cameraCanvas.toDataURL("image/jpeg", 0.95);
            } else {
                // Capture from Virtual Simulator Canvas
                cameraCanvas.width = virtualCameraCanvas.width;
                cameraCanvas.height = virtualCameraCanvas.height;
                const ctx = cameraCanvas.getContext("2d");
                ctx.drawImage(virtualCameraCanvas, 0, 0, cameraCanvas.width, cameraCanvas.height);
                dataUrl = cameraCanvas.toDataURL("image/jpeg", 0.95);
            }

            stopAllCameras();

            showToast("Processing captured envelope snapshot...", "⚡");
            showLoading();

            try {
                const res = await fetch("/api/upload", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ image_base64: dataUrl })
                });

                if (!res.ok) {
                    const errData = await res.json().catch(() => ({}));
                    throw new Error(errData.error || `HTTP ${res.status}`);
                }

                const data = await res.json();
                await animateLoadingSteps();
                renderResults(data);
                showToast("Captured envelope processed successfully!", "✓");
            } catch (err) {
                showUpload();
                showToast(`Analysis failed: ${err.message}`, "❌");
            }
        });
    }

    // ========== 16. Industrial Conveyor Belt Auto-Sort Simulator ==========
    const openConveyorBtn = document.getElementById("open-conveyor-btn");
    const conveyorSection = document.getElementById("conveyor-section");
    const simRunBtn = document.getElementById("sim-run-btn");
    const simBatchBtn = document.getElementById("sim-batch-btn");
    const simSpeedBtn = document.getElementById("sim-speed-btn");
    const simResetBtn = document.getElementById("sim-reset-btn");
    const travelingEnvelope = document.getElementById("traveling-envelope");
    const travelPinTag = document.getElementById("travel-pin-tag");
    const opticalTunnel = document.getElementById("optical-tunnel");
    const sensorIndicator = document.getElementById("sensor-indicator");
    const simTotalCount = document.getElementById("sim-total-count");
    const simLastChute = document.getElementById("sim-last-chute");
    const conveyorStripes = document.getElementById("conveyor-stripes");

    let simSpeedMultiplier = 1;
    let totalSortedLetters = 0;
    let isSimulating = false;

    if (openConveyorBtn) {
        openConveyorBtn.addEventListener("click", () => {
            conveyorSection.scrollIntoView({ behavior: "smooth", block: "center" });
            showToast("Conveyor Belt Simulator ready", "⚙️");
        });
    }

    if (simSpeedBtn) {
        simSpeedBtn.addEventListener("click", () => {
            if (simSpeedMultiplier === 1) simSpeedMultiplier = 2;
            else if (simSpeedMultiplier === 2) simSpeedMultiplier = 4;
            else simSpeedMultiplier = 1;
            simSpeedBtn.textContent = `⚡ ${simSpeedMultiplier}× Speed`;
            if (conveyorStripes) conveyorStripes.style.animationDuration = `${1.2 / simSpeedMultiplier}s`;
            showToast(`Conveyor speed: ${simSpeedMultiplier}×`, "⚡");
        });
    }

    if (simResetBtn) {
        simResetBtn.addEventListener("click", () => {
            totalSortedLetters = 0;
            simTotalCount.textContent = "0";
            simLastChute.textContent = "Standby";
            for (let i = 1; i <= 5; i++) {
                const el = document.getElementById(`bin-count-${i}`);
                if (el) el.textContent = "0";
            }
            showToast("Sorting bins reset to 0", "🔄");
        });
    }

    function playLaserSound() {
        try {
            const ctx = new (window.AudioContext || window.webkitAudioContext)();
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = "sawtooth";
            osc.frequency.setValueAtTime(880, ctx.currentTime);
            osc.frequency.exponentialRampToValueAtTime(220, ctx.currentTime + 0.15);
            gain.gain.setValueAtTime(0.15, ctx.currentTime);
            gain.gain.linearRampToValueAtTime(0.01, ctx.currentTime + 0.15);
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.start();
            osc.stop(ctx.currentTime + 0.15);
        } catch (e) {}
    }

    function playDropChime() {
        try {
            const ctx = new (window.AudioContext || window.webkitAudioContext)();
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = "sine";
            osc.frequency.setValueAtTime(523.25, ctx.currentTime);
            osc.frequency.setValueAtTime(659.25, ctx.currentTime + 0.08);
            gain.gain.setValueAtTime(0.2, ctx.currentTime);
            gain.gain.linearRampToValueAtTime(0.01, ctx.currentTime + 0.25);
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.start();
            osc.stop(ctx.currentTime + 0.25);
        } catch (e) {}
    }

    function resolveTargetBin(pin) {
        const p = String(pin || "").trim();
        if (/^400\d{3}$/.test(p) || p.startsWith("40")) return { binId: 1, hub: "BOM-GPO", name: "Mumbai Hub (West)" };
        if (/^110\d{3}$/.test(p) || p.startsWith("11")) return { binId: 2, hub: "DEL-NDHO", name: "Delhi Hub (North)" };
        if (/^560\d{3}$/.test(p) || p.startsWith("56")) return { binId: 3, hub: "BLR-GPO", name: "Bengaluru (South)" };
        if (/^700\d{3}$/.test(p) || p.startsWith("70") || p.startsWith("22")) return { binId: 4, hub: "CCU-GPO", name: "Kolkata Hub (East)" };
        return { binId: 5, hub: "REJECT", name: "Manual Review" };
    }

    async function simulateEnvelopeSort(pin, labelText = "") {
        if (!travelingEnvelope) return;

        const target = resolveTargetBin(pin);
        const durationFactor = 1 / simSpeedMultiplier;

        travelingEnvelope.style.display = "flex";
        travelingEnvelope.style.transition = `left ${0.6 * durationFactor}s ease-in, top ${0.5 * durationFactor}s ease-in, transform ${0.5 * durationFactor}s ease, opacity 0.3s`;
        travelingEnvelope.style.left = "20px";
        travelingEnvelope.style.top = "56px";
        travelingEnvelope.style.opacity = "1";
        travelingEnvelope.style.transform = "scale(1)";
        travelPinTag.textContent = pin || "UNKNOWN";
        sensorIndicator.textContent = "DETECTING";
        sensorIndicator.style.color = "#fbbf24";

        await sleep(100 * durationFactor);

        travelingEnvelope.style.left = "150px";
        await sleep(600 * durationFactor);

        playLaserSound();
        sensorIndicator.textContent = target.hub;
        sensorIndicator.style.color = target.hub === "REJECT" ? "#ef4444" : "#34d399";
        opticalTunnel.style.boxShadow = `0 0 26px ${target.hub === "REJECT" ? "#ef4444" : "#3b82f6"}`;

        await sleep(350 * durationFactor);
        opticalTunnel.style.boxShadow = "0 0 20px rgba(59, 130, 246, 0.4)";

        const rigEl = document.getElementById("conveyor-rig");
        const rigWidth = (rigEl ? rigEl.offsetWidth : 600) || 600;
        const binOffsets = [0.10, 0.30, 0.50, 0.70, 0.90];
        const targetX = Math.round(rigWidth * binOffsets[target.binId - 1] - 32);

        travelingEnvelope.style.left = `${Math.max(20, targetX)}px`;
        await sleep(650 * durationFactor);

        const diverterEl = document.getElementById(`diverter-${target.binId}`);
        const binBoxEl = document.getElementById(`bin-${target.binId}`);
        if (diverterEl) diverterEl.classList.add("active");
        if (binBoxEl) binBoxEl.classList.add("highlight-bin");

        travelingEnvelope.style.top = "200px";
        travelingEnvelope.style.opacity = "0";
        travelingEnvelope.style.transform = "scale(0.8) rotate(15deg)";
        playDropChime();

        await sleep(400 * durationFactor);

        const binCountEl = document.getElementById(`bin-count-${target.binId}`);
        if (binCountEl) {
            const cur = parseInt(binCountEl.textContent || "0", 10) + 1;
            binCountEl.textContent = cur;
        }

        totalSortedLetters++;
        simTotalCount.textContent = totalSortedLetters;
        simLastChute.textContent = `${target.hub} (${target.name})`;

        // Record into conveyor sorted manifest
        conveyorSortedManifest.push({
            sortingId: `SORT_${Date.now().toString(36).toUpperCase()}_${totalSortedLetters}`,
            timestamp: new Date().toISOString(),
            pin: pin || "400001",
            binId: target.binId,
            chute: `Chute #${target.binId}`,
            hub: target.hub,
            destinationName: target.name
        });

        if (diverterEl) diverterEl.classList.remove("active");
        if (binBoxEl) binBoxEl.classList.remove("highlight-bin");
        sensorIndicator.textContent = "IDLE";
        sensorIndicator.style.color = "#10b981";

        travelingEnvelope.style.display = "none";
        await sleep(150 * durationFactor);
    }

    if (simRunBtn) {
        simRunBtn.addEventListener("click", async () => {
            if (isSimulating) return;
            isSimulating = true;
            simRunBtn.disabled = true;

            const pin = currentResult && currentResult.structured_address 
                ? currentResult.structured_address.postal_code 
                : "400001";

            showToast(`Sorting envelope for PIN ${pin}...`, "⚙️");
            await simulateEnvelopeSort(pin, "Current Envelope");
            isSimulating = false;
            simRunBtn.disabled = false;
        });
    }

    if (simBatchBtn) {
        simBatchBtn.addEventListener("click", async () => {
            if (isSimulating) return;
            isSimulating = true;
            simBatchBtn.disabled = true;

            const batchPins = [
                "400001", "110001", "560001", "700001", "400050",
                "110025", "560068", "999999", "400018", "700019"
            ];

            showToast("Streaming high-speed batch through conveyor...", "🚀");

            for (let i = 0; i < batchPins.length; i++) {
                await simulateEnvelopeSort(batchPins[i]);
            }

            showToast(`Batch completed: ${batchPins.length} envelopes auto-sorted!`, "✓");
            isSimulating = false;
            simBatchBtn.disabled = false;
        });
    }

    // Export Conveyor Sort Manifest CSV
    if (simExportCsvBtn) {
        simExportCsvBtn.addEventListener("click", () => {
            if (conveyorSortedManifest.length === 0) {
                if (sessionManifest.length > 0) {
                    const csvContent = generateCsvManifest(sessionManifest);
                    downloadCsvFile(csvContent, `postal_session_manifest_${Date.now()}.csv`);
                } else {
                    showToast("No envelopes sorted yet. Click 'Stream 10 Envelopes' or 'Sort' first!", "ℹ️");
                }
                return;
            }

            const headers = [
                "Sorting_ID",
                "Timestamp",
                "PIN_Code",
                "Diverted_Chute",
                "RMS_Sorting_Hub",
                "Destination_Terminal",
                "Sort_Bin_Number",
                "Automated_Action"
            ].join(",");

            const rows = conveyorSortedManifest.map((m) => [
                `"${m.sortingId}"`,
                `"${m.timestamp}"`,
                `"${m.pin}"`,
                `"${m.chute}"`,
                `"${m.hub}"`,
                `"${m.destinationName}"`,
                `"Bin ${m.binId}"`,
                `"AUTO_DIVERTED_PASS"`
            ].join(","));

            const csvData = [headers, ...rows].join("\r\n");
            const dateStr = new Date().toISOString().slice(0, 10).replace(/-/g, "");
            downloadCsvFile(csvData, `conveyor_sorting_manifest_${dateStr}.csv`);
            showToast(`Conveyor manifest exported (${conveyorSortedManifest.length} sorted items)`, "📊");
        });
    }

    function escapeHtml(str) {
        if (!str) return "";
        return str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
})();
