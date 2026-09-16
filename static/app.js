document.addEventListener("DOMContentLoaded", () => {
    // State
    let selectedFile = null;
    let currentReport = null;
    let currentOriginalImage = null;
    let currentAnnotatedImage = null;
    let currentViewMode = "annotated";

    // DOM Elements
    const uploadZone = document.getElementById("uploadZone");
    const fileInput = document.getElementById("fileInput");
    const uploadPrompt = document.getElementById("uploadPrompt");
    const previewWrapper = document.getElementById("previewWrapper");
    const imagePreview = document.getElementById("imagePreview");
    const removeImageBtn = document.getElementById("removeImageBtn");

    const confSlider = document.getElementById("confSlider");
    const confValue = document.getElementById("confValue");
    const groupDuplicatesToggle = document.getElementById("groupDuplicatesToggle");
    const detectBtn = document.getElementById("detectBtn");
    const resetBtn = document.getElementById("resetBtn");

    const resultsCard = document.getElementById("resultsCard");
    const emptyState = document.getElementById("emptyState");
    const resultsContent = document.getElementById("resultsContent");
    const loadingOverlay = document.getElementById("loadingOverlay");

    const statTotalDetections = document.getElementById("statTotalDetections");
    const statCategoriesCount = document.getElementById("statCategoriesCount");
    const statPeakConfidence = document.getElementById("statPeakConfidence");
    const statLatency = document.getElementById("statLatency");

    const mainResultImage = document.getElementById("mainResultImage");
    const splitOriginalImage = document.getElementById("splitOriginalImage");
    const splitAnnotatedImage = document.getElementById("splitAnnotatedImage");
    const singleView = document.getElementById("singleView");
    const splitView = document.getElementById("splitView");
    const viewToggleBtns = document.querySelectorAll(".btn-toggle");

    const breakdownTableBody = document.getElementById("breakdownTableBody");
    const detectionsGrid = document.getElementById("detectionsGrid");

    const viewFullReportBtn = document.getElementById("viewFullReportBtn");
    const downloadPdfBtn = document.getElementById("downloadPdfBtn");
    const downloadPdfBtn2 = document.getElementById("downloadPdfBtn2");
    const exportJsonBtn = document.getElementById("exportJsonBtn");
    const reportNavTab = document.getElementById("reportNavTab");

    const navTabs = document.querySelectorAll(".nav-tab");
    const tabContents = document.querySelectorAll(".tab-content");

    // Initialize System Status
    fetchSystemStatus();

    // Tab Switching
    navTabs.forEach(tab => {
        tab.addEventListener("click", () => {
            if (tab.disabled) return;
            const target = tab.getAttribute("data-tab");
            
            navTabs.forEach(t => t.classList.remove("active"));
            tabContents.forEach(c => c.classList.remove("active"));

            tab.classList.add("active");
            const content = document.getElementById(target);
            if (content) content.classList.add("active");
        });
    });

    // Slider
    confSlider.addEventListener("input", (e) => {
        confValue.textContent = parseFloat(e.target.value).toFixed(2);
    });

    // Upload zone interactions
    uploadZone.addEventListener("click", (e) => {
        if (e.target !== removeImageBtn && !previewWrapper.contains(e.target)) {
            fileInput.click();
        }
    });

    uploadZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        uploadZone.classList.add("dragover");
    });

    uploadZone.addEventListener("dragleave", () => {
        uploadZone.classList.remove("dragover");
    });

    uploadZone.addEventListener("drop", (e) => {
        e.preventDefault();
        uploadZone.classList.remove("dragover");
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    removeImageBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        resetUpload();
    });

    resetBtn.addEventListener("click", () => {
        resetUpload();
        resetResults();
    });

    function handleFileSelect(file) {
        if (!file.type.startsWith("image/")) {
            alert("Please upload a valid image file (JPEG, PNG, WebP).");
            return;
        }
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (event) => {
            imagePreview.src = event.target.result;
            uploadPrompt.style.display = "none";
            previewWrapper.style.display = "flex";
            detectBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    function resetUpload() {
        selectedFile = null;
        fileInput.value = "";
        imagePreview.src = "";
        uploadPrompt.style.display = "block";
        previewWrapper.style.display = "none";
        detectBtn.disabled = true;
    }

    function resetResults() {
        currentReport = null;
        currentOriginalImage = null;
        currentAnnotatedImage = null;
        emptyState.style.display = "block";
        resultsContent.style.display = "none";
        reportNavTab.disabled = true;
        // Switch back to detect tab if on report tab
        if (reportNavTab.classList.contains("active")) {
            document.querySelector('[data-tab="detectTab"]').click();
        }
    }

    // View Toggles (Annotated / Original / Side-by-Side)
    viewToggleBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            viewToggleBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            currentViewMode = btn.getAttribute("data-view");
            updateImageViewer();
        });
    });

    function updateImageViewer() {
        if (!currentAnnotatedImage || !currentOriginalImage) return;

        if (currentViewMode === "split") {
            singleView.style.display = "none";
            splitView.style.display = "grid";
            splitOriginalImage.src = currentOriginalImage;
            splitAnnotatedImage.src = currentAnnotatedImage;
        } else {
            splitView.style.display = "none";
            singleView.style.display = "flex";
            if (currentViewMode === "annotated") {
                mainResultImage.src = currentAnnotatedImage;
            } else {
                mainResultImage.src = currentOriginalImage;
            }
        }
    }

    // Run Detection
    detectBtn.addEventListener("click", async () => {
        if (!selectedFile) return;

        loadingOverlay.style.display = "flex";
        
        try {
            const formData = new FormData();
            formData.append("file", selectedFile);
            formData.append("confidence_threshold", confSlider.value);
            formData.append("group_duplicates", groupDuplicatesToggle.checked);

            const response = await fetch("/api/predict", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || "Inference failed.");
            }

            const data = await response.json();
            if (data.success) {
                currentReport = data.report;
                currentOriginalImage = data.original_image;
                currentAnnotatedImage = data.annotated_image;

                renderResults(data.report);
                renderFormalReport(data.report);
                reportNavTab.disabled = false;
            }
        } catch (error) {
            console.error("Prediction error:", error);
            alert("Error during detection: " + error.message);
        } finally {
            loadingOverlay.style.display = "none";
        }
    });

    function renderResults(report) {
        emptyState.style.display = "none";
        resultsContent.style.display = "block";

        // Metrics
        statTotalDetections.textContent = report.summary.total_detections;
        statCategoriesCount.textContent = report.summary.damage_categories_count;
        statPeakConfidence.textContent = report.summary.highest_confidence_pct;
        statLatency.textContent = (report.inference_latency_ms || 0) + " ms";

        // Images
        currentViewMode = "annotated";
        viewToggleBtns.forEach(b => b.classList.remove("active"));
        document.querySelector('[data-view="annotated"]').classList.add("active");
        updateImageViewer();

        // Breakdown Table
        breakdownTableBody.innerHTML = "";
        if (report.breakdown.length === 0) {
            breakdownTableBody.innerHTML = `<tr><td colspan="4" class="text-muted" style="text-align:center;">No damage detected above threshold (${confSlider.value}).</td></tr>`;
        } else {
            report.breakdown.forEach(item => {
                const row = document.createElement("tr");
                const regions = (item.regions_present || []).map(r => `<span class="tag-location">${r}</span>`).join(" ");
                row.innerHTML = `
                    <td><span class="tag-badge tag-${item.class_name}">${item.display_name}</span></td>
                    <td><strong>${item.count}</strong></td>
                    <td><strong class="param-val">${item.highest_confidence_pct}</strong></td>
                    <td>${regions || "center"}</td>
                `;
                breakdownTableBody.appendChild(row);
            });
        }

        // Detailed Detections Grid
        detectionsGrid.innerHTML = "";
        if (report.detections.length === 0) {
            detectionsGrid.innerHTML = `<p class="text-muted">No bounding boxes to display.</p>`;
        } else {
            report.detections.forEach(det => {
                const card = document.createElement("div");
                card.className = "detection-card";
                const bbox = det.bbox;
                const bboxText = `x1:${bbox.x1} y1:${bbox.y1} x2:${bbox.x2} y2:${bbox.y2} (${bbox.width}x${bbox.height}px)`;
                
                card.innerHTML = `
                    <div class="det-header">
                        <span class="tag-badge tag-${det.class_name}">${det.display_name}</span>
                        <span class="det-conf">${det.confidence_pct}</span>
                    </div>
                    <div class="det-loc">${det.location_description}</div>
                    <div class="det-bbox">BBox: ${bboxText}</div>
                `;
                detectionsGrid.appendChild(card);
            });
        }
    }

    function renderFormalReport(report) {
        document.getElementById("repId").textContent = report.report_id;
        document.getElementById("repDate").textContent = report.created_at.substring(0, 19).replace("T", " ") + " UTC";
        document.getElementById("repExecutiveSummary").textContent = report.executive_summary;

        document.getElementById("repOriginalImg").src = currentOriginalImage;
        document.getElementById("repAnnotatedImg").src = currentAnnotatedImage;

        // Breakdown in formal report
        const repBreakdownBody = document.getElementById("repBreakdownBody");
        repBreakdownBody.innerHTML = "";
        if (report.breakdown.length === 0) {
            repBreakdownBody.innerHTML = `<tr><td colspan="4">No damage regions identified.</td></tr>`;
        } else {
            report.breakdown.forEach(item => {
                const row = document.createElement("tr");
                const regions = (item.regions_present || []).join(", ") || "center";
                row.innerHTML = `
                    <td><strong>${item.display_name}</strong></td>
                    <td>${item.count}</td>
                    <td>${item.highest_confidence_pct}</td>
                    <td>${regions}</td>
                `;
                repBreakdownBody.appendChild(row);
            });
        }

        // Inventory table in formal report
        const repInventoryBody = document.getElementById("repInventoryBody");
        repInventoryBody.innerHTML = "";
        if (report.detections.length === 0) {
            repInventoryBody.innerHTML = `<tr><td colspan="6">No detected damage entries.</td></tr>`;
        } else {
            report.detections.forEach(det => {
                const row = document.createElement("tr");
                const bbox = det.bbox;
                const bboxStr = `[${bbox.x1}, ${bbox.y1}, ${bbox.x2}, ${bbox.y2}]`;
                row.innerHTML = `
                    <td>${det.id}</td>
                    <td><strong>${det.display_name}</strong></td>
                    <td>${det.confidence_pct}</td>
                    <td><span class="tag-location">${det.location}</span></td>
                    <td><code>${bboxStr}</code></td>
                    <td>${det.severity}</td>
                `;
                repInventoryBody.appendChild(row);
            });
        }

        // Limitations & Recommendations
        const repLimitationsList = document.getElementById("repLimitationsList");
        repLimitationsList.innerHTML = (report.limitations || []).map(l => `<li>${l}</li>`).join("");

        const repRecommendationsList = document.getElementById("repRecommendationsList");
        repRecommendationsList.innerHTML = (report.recommendations || []).map(r => `<li>${r}</li>`).join("");

        // Model Provenance
        const provenance = report.model_info || {};
        document.getElementById("repModelProvenance").innerHTML = `
            <strong>Architecture:</strong> ${provenance.architecture || "RTDetrForObjectDetection"} &nbsp;|&nbsp;
            <strong>Base:</strong> ${provenance.base_checkpoint || "rtdetr_r50vd"} &nbsp;|&nbsp;
            <strong>Resolution:</strong> ${provenance.input_resolution || "640x640"} &nbsp;|&nbsp;
            <strong>Device:</strong> ${provenance.device || "CPU"} &nbsp;|&nbsp;
            <strong>Framework:</strong> ${provenance.framework || "PyTorch"}
        `;
    }

    // View Structured Report Button
    viewFullReportBtn.addEventListener("click", () => {
        reportNavTab.disabled = false;
        reportNavTab.click();
    });

    // PDF Download Handlers
    async function handlePdfDownload() {
        if (!selectedFile) return;

        loadingOverlay.style.display = "flex";
        try {
            const formData = new FormData();
            formData.append("file", selectedFile);
            formData.append("confidence_threshold", confSlider.value);
            formData.append("group_duplicates", groupDuplicatesToggle.checked);

            const response = await fetch("/api/report/pdf", {
                method: "POST",
                body: formData
            });

            if (!response.ok) throw new Error("Failed to generate PDF report.");

            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `Vehicle_Damage_Report_${currentReport ? currentReport.report_id : "audit"}.pdf`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        } catch (error) {
            console.error("PDF download error:", error);
            alert("Error downloading PDF: " + error.message);
        } finally {
            loadingOverlay.style.display = "none";
        }
    }

    downloadPdfBtn.addEventListener("click", handlePdfDownload);
    downloadPdfBtn2.addEventListener("click", handlePdfDownload);

    // Export JSON
    exportJsonBtn.addEventListener("click", () => {
        if (!currentReport) return;
        const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(currentReport, null, 2));
        const a = document.createElement("a");
        a.href = dataStr;
        a.download = `Damage_Report_${currentReport.report_id}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    });

    // Fetch System & Model Info
    async function fetchSystemStatus() {
        try {
            const resHealth = await fetch("/api/health");
            if (resHealth.ok) {
                const health = await resHealth.json();
                const deviceBadge = document.getElementById("deviceBadge");
                deviceBadge.textContent = health.device.toUpperCase();
            }

            const resInfo = await fetch("/api/model-info");
            if (resInfo.ok) {
                const infoRes = await resInfo.json();
                if (infoRes.success) {
                    const info = infoRes.info;
                    document.getElementById("infoArch").textContent = info.architecture;
                    document.getElementById("infoCheckpoint").textContent = info.base_checkpoint.split("/").pop();
                    document.getElementById("infoEpochs").textContent = info.trained_epochs || "21";
                    document.getElementById("infoMap").textContent = (info.test_mAP_50 ? (info.test_mAP_50 * 100).toFixed(1) + "%" : "58.9%");
                }
            }
        } catch (e) {
            console.warn("Could not fetch initial health status:", e);
        }
    }
});
