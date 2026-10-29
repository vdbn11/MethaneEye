const SITE = [25.9587, 49.6775];
const POLL_FALLBACK_MS = 1200;

const SECTORS = [
  { key: "N", letter: "N", name: "North", color: "#3aa0ff" },
  { key: "E", letter: "E", name: "East", color: "#2ecc71" },
  { key: "S", letter: "S", name: "South", color: "#ffb066" },
  { key: "W", letter: "W", name: "West", color: "#ff5a4d" },
];
const SECTOR_BY_KEY = Object.fromEntries(SECTORS.map((sector) => [sector.key, sector]));
const PHASE_NAMES = { idle: "STANDBY", transit: "TRANSIT", inspect: "INSPECTION", return: "RETURN", done: "COMPLETE", hold: "HOLD" };

let state = null;
let socket = null;
let revealTruth = false;
let map;
let pipelineLayer;
let routeLines = {};
let trailLines = {};
let droneMarkers = {};
let waypointMarkers = new Map();
let leakMarkers = new Map();
let detectionMarkers = new Map();
let refinedMarkers = new Map();
let scanDots = new Map();
let leakStateMarkers = new Map();
let vehicleMarkers = new Map();
let lastAutoMode = null;
let pipelineReady = false;
let gaChart = null;
let gaSignature = "";
let lastOrdersSignature = "";
let lastLogsSignature = "";
let lastFleetHtml = "";
let lastPhase = null;
let selectedDrone = "N";

const $ = (id) => document.getElementById(id);
const fmt = (value, digits = 1) => (value === null || value === undefined || Number.isNaN(value) ? "--" : Number(value).toFixed(digits));

function initMap() {
  if (!window.L) {
    $("map").innerHTML = '<div style="padding:30px;color:#ff8a73;font-family:Consolas">Leaflet could not load. Check the network connection.</div>';
    return;
  }
  map = L.map("map", { zoomControl: true, attributionControl: true }).setView(SITE, 13);
  L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
    maxZoom: 19,
    attribution: "Imagery &copy; Esri, Maxar, Earthstar Geographics",
  }).addTo(map);
  L.control.scale({ imperial: false, position: "bottomright" }).addTo(map);

  pipelineLayer = L.layerGroup().addTo(map);
  window.waypointLayer = L.layerGroup().addTo(map);
  window.leakLayer = L.layerGroup().addTo(map);
  window.detectionLayer = L.layerGroup().addTo(map);
  window.refinedLayer = L.layerGroup().addTo(map);
  window.scanLayer = L.layerGroup().addTo(map);
  window.leakStateLayer = L.layerGroup().addTo(map);
  window.vehicleLayer = L.layerGroup().addTo(map);
  window.routeLayer = L.layerGroup().addTo(map);
  window.trailLayer = L.layerGroup().addTo(map);
  window.droneLayer = L.layerGroup().addTo(map);

  SECTORS.forEach((sector) => {
    routeLines[sector.key] = L.polyline([], {
      color: sector.color,
      weight: 2,
      opacity: 0.85,
      dashArray: "7 9",
      lineCap: "round",
    }).addTo(window.routeLayer);
    trailLines[sector.key] = L.polyline([], {
      color: sector.color,
      weight: 2.4,
      opacity: 0.55,
      lineCap: "round",
    }).addTo(window.trailLayer);
    droneMarkers[sector.key] = L.marker(SITE, {
      icon: L.divIcon({
        className: "drone-marker",
        html: `<div class="drone-pin" style="--c:${sector.color}"><div class="drone-body"><i></i><i></i><i></i><i></i><b></b></div><em>${sector.letter}</em></div>`,
        iconSize: [34, 34],
        iconAnchor: [17, 17],
      }),
      zIndexOffset: 1000,
      opacity: 0.7,
    }).addTo(window.droneLayer);
  });
}

function drawPipeline(pipeline) {
  if (pipelineReady || !pipeline) return;
  pipelineLayer.clearLayers();
  const bounds = [];
  for (const edge of pipeline.edges) {
    const latlngs = [edge.from, edge.to];
    const pipeColor = edge.color || "#3ba0a5";
    L.polyline(latlngs, { color: "#061b20", weight: 6, opacity: 0.85 }).addTo(pipelineLayer);
    L.polyline(latlngs, { color: pipeColor, weight: 2.0, opacity: 0.95 }).addTo(pipelineLayer);
    bounds.push(edge.from, edge.to);
  }
  if (bounds.length) map.fitBounds(L.latLngBounds(bounds).pad(0.14));
  pipelineReady = true;
  $("network-length").textContent = `${fmt(pipeline.total_km)} km`;
  $("waypoint-count").textContent = `${state ? state.waypoints.length : "--"}`;
}

function updateWaypoints(waypoints) {
  if (!window.waypointLayer) return;
  waypoints.forEach((point) => {
    const sector = SECTOR_BY_KEY[point.drone];
    const sectorColor = sector ? sector.color : "#6f8a90";
    const markerKey = `${point.drone}-${point.index}`;
    let marker = waypointMarkers.get(markerKey);
    if (!marker) {
      marker = L.circleMarker([point.lat, point.lon], {
        radius: 4,
        color: sectorColor,
        weight: 1,
        fillColor: "#061216",
        fillOpacity: 1,
      }).addTo(window.waypointLayer);
      marker.bindTooltip(`Point ${point.index + 1} · ${sector ? sector.name : "?"} network`, { direction: "top", opacity: 0.8 });
      waypointMarkers.set(markerKey, marker);
    }
    const risk = point.risk > 300;
    marker.setStyle({
      radius: point.covered ? 5 : risk ? 5 : 3.5,
      color: point.covered ? sectorColor : risk ? "#ffb020" : sectorColor,
      fillColor: point.covered ? sectorColor : "#061216",
      fillOpacity: point.covered ? 0.9 : 1,
      opacity: point.covered ? 1 : 0.75,
    });
    marker.setTooltipContent(`Point ${point.index + 1} · ${sector ? sector.name : "?"} network`);
  });
}

function updateLeaks(leaks) {
  if (!window.leakLayer) return;
  leaks.forEach((leak) => {
    const sector = SECTOR_BY_KEY[leak.drone];
    let truth = leakMarkers.get(leak.id);
    if (!truth) {
      truth = L.marker([leak.lat, leak.lon], {
        icon: L.divIcon({ className: "truth-marker-purple", html: "&#9679;", iconSize: [18, 18], iconAnchor: [9, 9] }),
        opacity: 0,
      }).addTo(window.leakLayer);
      leakMarkers.set(leak.id, truth);
    }
    truth.setOpacity(revealTruth || leak.detected ? 0.95 : 0);
    const popupHtml = `Ground-truth leak #${leak.id}<br/>${fmt(leak.severity_kg_h, 0)} kg/h<br/>${sector ? sector.name + " network" : ""}`;
    if (truth.getPopup()) {
      truth.setPopupContent(popupHtml);
    } else {
      truth.bindPopup(popupHtml);
    }
  });
}

function updateDetections(detections) {
  if (!window.detectionLayer) return;
  detections.forEach((detection) => {
    if (detectionMarkers.has(detection.id)) return;
    const sector = SECTOR_BY_KEY[detection.drone];
    const marker = L.marker([detection.lat, detection.lon], {
      icon: L.divIcon({ className: "pulse-marker-purple", html: "", iconSize: [16, 16], iconAnchor: [8, 8] }),
      zIndexOffset: 700,
    }).addTo(window.detectionLayer);

    const imgHtml = detection.frame_url
      ? `<br/><img src="${detection.frame_url}" style="width:100%;max-width:240px;border-radius:4px;margin-top:6px;border:1px solid #9b59b6;" />`
      : "";

    marker.bindPopup(
      `<strong>Leak #${detection.leak_id || "?"} (Detected)</strong><br/>` +
        `Spotted by <b style="color:${sector ? sector.color : "#9b59b6"}">${detection.letter || "?"} drone</b><br/>` +
        `Estimated ${fmt(detection.severity_kg_h, 0)} kg/h · Conf ${fmt(detection.confidence * 100, 1)}%` +
        imgHtml
    );
    detectionMarkers.set(detection.id, marker);

    if (detection.refined) {
      const refined = L.marker([detection.refined.lat, detection.refined.lon], {
        icon: L.divIcon({ className: "refined-marker", html: "", iconSize: [14, 14], iconAnchor: [7, 7] }),
        zIndexOffset: 650,
      }).addTo(window.refinedLayer);
      refined.bindPopup(`GWO refined source<br/>Error ${fmt(detection.refined.error_m)} m`);
      refinedMarkers.set(detection.id, refined);
    }
    showToast(`Leak #${detection.leak_id} confirmed (Purple Point) · Click to view drone recording`);
  });
}

function scanPopupHtml(scan, color) {
  const detectionText =
    scan.detected === true
      ? "<b style='color:#9b59b6'>LEAK RECORDED (Purple Point)</b>"
      : scan.detected === false
        ? "camera scan · clear"
        : "stage-1 clear frame";
  const imgHtml = scan.frame_url
    ? `<br/><img src="${scan.frame_url}" style="width:100%;max-width:240px;border-radius:4px;margin-top:6px;border:1px solid ${color};" />`
    : "";
  return (
    `<strong>${scan.letter} drone · scan ${scan.capture_id}</strong><br/>` +
    `${scan.label}<br/>` +
    `T+${formatClock(scan.t)} · waypoint ${scan.waypoint}<br/>` +
    `${detectionText}` +
    (scan.ratio !== null && scan.ratio !== undefined
      ? `<br/>Plume ${(scan.ratio * 100).toFixed(2)}% · conf ${((scan.confidence || 0) * 100).toFixed(1)}%`
      : "") +
    imgHtml
  );
}

function updateScans(drones) {
  if (!window.scanLayer) return;
  const anyScans = drones.some((drone) => (drone.scan_points || []).length);
  if (!anyScans && scanDots.size) {
    scanDots.forEach((layers) => {
      window.scanLayer.removeLayer(layers.dot);
      window.scanLayer.removeLayer(layers.ring);
    });
    scanDots.clear();
    return;
  }
  drones.forEach((drone) => {
    (drone.scan_points || []).forEach((scan) => {
      const isLeakScan = scan.detected === true;
      const color = isLeakScan ? "#9b59b6" : scan.color || (SECTOR_BY_KEY[scan.drone] || {}).color || "#00e0b8";
      const signature = [scan.detected, scan.ratio, scan.frame_url, scan.label].join("|");
      const existing = scanDots.get(scan.id);
      if (existing) {
        if (existing.signature !== signature) {
          existing.signature = signature;
          existing.dot.setStyle({
            radius: isLeakScan ? 7 : 5,
            color,
            weight: isLeakScan ? 2 : 1.6,
            fillColor: color,
            fillOpacity: 0.9,
          });
          existing.ring.setStyle({
            color,
            fillColor: color,
            fillOpacity: isLeakScan ? 0.15 : 0.07,
          });
          existing.dot.setPopupContent(scanPopupHtml(scan, color));
        }
        return;
      }
      const ring = L.circle([scan.lat, scan.lon], {
        radius: 38,
        color,
        weight: 0.8,
        opacity: 0.55,
        fillColor: color,
        fillOpacity: isLeakScan ? 0.15 : 0.07,
        interactive: false,
      }).addTo(window.scanLayer);
      const dot = L.circleMarker([scan.lat, scan.lon], {
        radius: isLeakScan ? 7 : 5,
        color,
        weight: isLeakScan ? 2 : 1.6,
        fillColor: color,
        fillOpacity: 0.9,
      }).addTo(window.scanLayer);
      dot.bindPopup(scanPopupHtml(scan, color));
      scanDots.set(scan.id, { dot, ring, signature });
    });
  });
}

function updateFleetMap(drones) {
  drones.forEach((drone) => {
    const marker = droneMarkers[drone.key];
    if (marker) {
      marker.setLatLng([drone.lat, drone.lon]);
      marker.setOpacity(drone.launched ? 1 : 0.7);
      const body = marker.getElement()?.querySelector(".drone-body");
      if (body) body.style.transform = `rotate(${drone.heading}deg)`;
      const pin = marker.getElement()?.querySelector(".drone-pin");
      if (pin) pin.title = `${drone.letter} · ${drone.name} — ${PHASE_NAMES[drone.phase] || drone.phase}`;
    }
    if (routeLines[drone.key]) routeLines[drone.key].setLatLngs(drone.route || []);
    if (trailLines[drone.key]) trailLines[drone.key].setLatLngs(drone.trail || []);
  });
}

const LEAK_STATUS = {
  open: { color: "#8b5cf6", label: "LEAK DETECTED", icon: "&#9888;" },
  queued: { color: "#8b5cf6", label: "QUEUED FOR REPAIR", icon: "&#8987;" },
  enroute: { color: "#f5b301", label: "REPAIR EN ROUTE", icon: "&#128666;" },
  repairing: { color: "#f5b301", label: "REPAIRING", icon: "&#128295;" },
  verify: { color: "#f5b301", label: "VERIFYING", icon: "&#128269;" },
  repaired: { color: "#15803d", label: "REPAIRED", icon: "&#10003;" },
  closed: { color: "#15803d", label: "VERIFIED & CLOSED", icon: "&#10003;" },
};

function leakPopupHtml(leak, detection) {
  const state = LEAK_STATUS[leak.status] || { label: leak.status, color: "#8b5cf6" };
  const img = detection && detection.frame_url
    ? `<br/><img src="${detection.frame_url}" style="width:100%;max-width:230px;border-radius:4px;margin-top:6px;border:1px solid ${state.color};" />`
    : "";
  const stamp = (value) => (value ? `${formatClock(value)}` : "--");
  return (
    `<strong>Leak #${leak.id} &middot; ${state.label}</strong><br/>` +
    `${fmt(leak.severity_kg_h, 0)} kg/h &middot; ${LEAK_STATUS_DRONE(leak)}<br/>` +
    `detected ${stamp(leak.detected_at)} &middot; repaired ${stamp(leak.repaired_at)} &middot; verified ${stamp(leak.verified_at)}` +
    img
  );
}

function LEAK_STATUS_DRONE(leak) {
  const sector = SECTOR_BY_KEY[leak.drone];
  return sector ? `<b style="color:${sector.color}">${sector.letter} drone</b>` : "";
}

function updateLeakStates(state) {
  if (!window.leakStateLayer) return;
  const visible = (state.leaks || []).filter((leak) => leak.detected);
  if (!visible.length && leakStateMarkers.size) {
    window.leakStateLayer.clearLayers();
    leakStateMarkers.clear();
    return;
  }
  visible.forEach((leak) => {
    const meta = LEAK_STATUS[leak.status] || LEAK_STATUS.open;
    const detection = (state.detections || []).find((item) => item.leak_id === leak.id);
    let entry = leakStateMarkers.get(leak.id);
    if (!entry) {
      const marker = L.marker([leak.lat, leak.lon], {
        icon: L.divIcon({
          className: "state-marker",
          html: `<div class="state-pin" style="--c:${meta.color}"><span>${meta.icon}</span></div>`,
          iconSize: [30, 30],
          iconAnchor: [15, 15],
        }),
        zIndexOffset: 800,
      }).addTo(window.leakStateLayer);
      marker.bindPopup(leakPopupHtml(leak, detection));
      entry = { marker, signature: "" };
      leakStateMarkers.set(leak.id, entry);
    }
    const signature = `${leak.status}|${leak.repaired_at}|${leak.verified_at}`;
    if (entry.signature !== signature) {
      entry.signature = signature;
      entry.marker.setIcon(
        L.divIcon({
          className: "state-marker",
          html: `<div class="state-pin ${leak.status}" style="--c:${meta.color}"><span>${meta.icon}</span></div>`,
          iconSize: [30, 30],
          iconAnchor: [15, 15],
        })
      );
      entry.marker.setPopupContent(leakPopupHtml(leak, detection));
    }
  });
}

function updateVehicles(vehicles) {
  if (!window.vehicleLayer) return;
  (vehicles || []).forEach((unit) => {
    let marker = vehicleMarkers.get(unit.id);
    if (!marker) {
      marker = L.marker([unit.lat, unit.lon], {
        icon: L.divIcon({
          className: "repair-marker",
          html: `<div class="repair-car"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M3 15V9h9l3 4h5v4h-2"/><path d="M3 17h2m6 0h6m3 0h1"/><circle cx="7.5" cy="17.5" r="1.6"/><circle cx="17.5" cy="17.5" r="1.6"/></svg><b>${unit.name.split(" ").pop()}</b></div>`,
          iconSize: [30, 26],
          iconAnchor: [15, 13],
        }),
        zIndexOffset: 900,
      }).addTo(window.vehicleLayer);
      marker.bindPopup("");
      vehicleMarkers.set(unit.id, marker);
    }
    marker.setLatLng([unit.lat, unit.lon]);
    const statusText = unit.status === "enroute" ? "en route" : unit.status === "repairing" ? "on site · repairing" : unit.status === "returning" ? "returning to base" : "standing by";
    marker.setPopupContent(`<strong>${unit.name}</strong><br/>${statusText}${unit.leak_id ? `<br/>Assigned to leak #${unit.leak_id}` : ""}`);
    const element = marker.getElement()?.querySelector(".repair-car");
    if (element) element.classList.toggle("busy", unit.status !== "idle");
  });
}

function updateChart(ga) {
  if (!window.Chart) return;
  const sectors = ga.sectors || [];
  if (!sectors.length) return;
  const signature = JSON.stringify(sectors.map((sector) => [sector.key, sector.convergence.length, sector.convergence[sector.convergence.length - 1]]));
  if (signature === gaSignature) return;
  gaSignature = signature;

  const maxLen = Math.max(...sectors.map((sector) => sector.convergence.length));
  const labels = Array.from({ length: maxLen }, (_, index) => index + 1);
  const datasets = sectors.map((sector) => ({
    label: `${sector.letter} · ${sector.name}`,
    data: sector.convergence,
    borderColor: sector.color,
    backgroundColor: "transparent",
    borderWidth: 1.4,
    pointRadius: 0,
    tension: 0.3,
    fill: false,
  }));

  if (!gaChart) {
    const context = $("ga-chart").getContext("2d");
    gaChart = new Chart(context, {
      type: "line",
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {
          legend: { display: true, labels: { color: "#617c82", boxWidth: 9, boxHeight: 2, font: { size: 8, family: "Consolas" } } },
          tooltip: { enabled: false },
        },
        scales: {
          x: { display: false },
          y: {
            grid: { color: "rgba(23, 50, 58, 0.55)", drawTicks: false },
            border: { display: false },
            ticks: { color: "#617c82", font: { size: 8, family: "Consolas" }, maxTicksLimit: 4 },
          },
        },
      },
    });
  } else {
    gaChart.data.labels = labels;
    gaChart.data.datasets = datasets;
    gaChart.update("none");
  }
}

function renderOrders(orders) {
  const body = $("orders-body");
  const auto = !!(state && state.auto_mode);
  const signature = JSON.stringify(orders) + "|" + auto;
  if (signature === lastOrdersSignature) return;
  lastOrdersSignature = signature;
  $("order-count").textContent = `${orders.length}`;
  if (!orders.length) {
    body.innerHTML = '<tr class="empty"><td colspan="8">No detections yet. Launch a drone to begin inspection.</td></tr>';
    return;
  }
  body.innerHTML = orders
    .map((order) => {
      const sector = SECTOR_BY_KEY[order.drone];
      const color = sector ? sector.color : "#7f9aa1";
      const statusClass =
        order.status === "Closed" || order.status === "Repaired"
          ? "done"
          : order.status === "En route" || order.status === "Repairing" || order.status === "Verifying"
            ? "moving"
            : "queued";
      const action =
        !auto && order.status === "Assigned"
          ? `<button class="mini-btn" data-dispatch="${order.leak_id}">Dispatch</button>`
          : order.status === "Closed"
            ? '<span class="ok-mark">&#10003;</span>'
            : "";
      return `<tr>
        <td>${order.id}</td>
        <td><span class="drone-chip" style="--c:${color}">${order.drone || "--"}</span></td>
        <td><span class="priority ${order.priority}">${order.priority}</span></td>
        <td>${order.crew}</td>
        <td>${fmt(order.severity_kg_h, 0)} kg/h</td>
        <td>${fmt(order.eta_min, 0)} min</td>
        <td><span class="order-status ${statusClass}">${order.status}</span></td>
        <td>${action}</td>
      </tr>`;
    })
    .join("");
}

function renderLogs(logs) {
  const stream = $("log-stream");
  const signature = JSON.stringify(logs);
  if (signature === lastLogsSignature) return;
  lastLogsSignature = signature;
  $("log-count").textContent = `${logs.length} events`;
  stream.innerHTML = logs
    .map((entry) => {
      const message = String(entry.message);
      const cls = message.includes("LEAK") ? "alert" : message.includes("GA") || message.includes("GWO") ? "hot" : "";
      return `<div class="log-line ${cls}"><span class="log-time">T+${formatClock(entry.t)}</span><span class="log-msg">${escapeHtml(message)}</span></div>`;
    })
    .join("");
  stream.scrollTop = stream.scrollHeight;
}

function renderFleetStrip(drones) {
  const html = drones
    .map((drone) => {
      const battery = Math.max(0, Math.min(100, drone.battery));
      const phase = drone.hold
        ? "HOLD"
        : drone.verifying
          ? "VERIFY"
          : PHASE_NAMES[drone.phase] || String(drone.phase).toUpperCase();
      return `<div class="fleet-cell" style="--c:${drone.color}">
        <div class="fc-head">
          <i class="fc-dot"></i><span class="fc-letter">${drone.letter}</span>
          <span class="fc-phase">${phase}</span>
        </div>
        <div class="fc-bar"><i style="width:${battery}%"></i></div>
        <div class="fc-meta"><span>${battery.toFixed(0)}%</span><span>${drone.scans} scans</span><span>${drone.detections} leak${drone.detections === 1 ? "" : "s"}</span></div>
      </div>`;
    })
    .join("");
  if (html !== lastFleetHtml) {
    $("fleet-strip").innerHTML = html;
    lastFleetHtml = html;
  }
}

function renderLaunchButtons(drones) {
  document.querySelectorAll(".drone-btn").forEach((button) => {
    const drone = drones.find((item) => item.key === button.dataset.drone);
    const flying = !!drone && drone.launched && drone.phase !== "done";
    button.classList.toggle("active", flying && !drone.hold);
    button.classList.toggle("hold", !!drone && !!drone.hold);
    button.classList.toggle("done", !!drone && drone.phase === "done");
    button.title = !drone
      ? "Launch"
      : drone.phase === "done"
        ? `${drone.name} drone complete`
        : drone.hold
          ? `${drone.name} drone on hold — click to resume`
          : flying
            ? `${drone.name} drone in flight — click to hold`
            : `Launch ${drone.name} drone`;
  });
}

function renderCamera(drones, selected) {
  const camera = selected.camera;
  const feed = $("feed");
  if (feed.src !== camera.data) feed.src = camera.data;

  document.querySelectorAll(".feed-tab, .feed-thumb").forEach((element) => {
    element.classList.toggle("active", element.dataset.drone === selected.key);
  });
  document.querySelectorAll(".feed-thumb").forEach((element) => {
    const drone = drones.find((item) => item.key === element.dataset.drone);
    const image = element.querySelector("img");
    if (drone && image && image.src !== drone.camera.data) image.src = drone.camera.data;
  });

  $("gas-ratio").textContent = `${(camera.ratio * 100).toFixed(2)}%`;
  $("gas-conf").textContent = `${(camera.confidence * 100).toFixed(1)}%`;
  $("infer-ms").textContent = camera.inference_ms ? `${camera.inference_ms.toFixed(0)} ms` : "-- ms";
  $("gas-bar").style.width = `${Math.min(100, camera.ratio * 100 * 6)}%`;
  const gasThreshold = camera.threshold || 0.008;
  $("feed-alert").classList.toggle("show", camera.ratio > gasThreshold);
  $("feed-model").textContent = camera.model ? camera.model.replace("_", " ").toUpperCase() : "U-NET / RESNET34";
  $("feed-mode").textContent = camera.ratio > gasThreshold ? "PLUME TRACK" : "ROUTINE SCAN";
  $("feed-drone").textContent = `${selected.letter} · ${selected.name.toUpperCase()}`;
  $("feed-drone").style.color = selected.color;
  $("scan-count").textContent = `${selected.scans}`;
}

function renderTelemetry(selected) {
  $("tm-drone").textContent = `${selected.letter} · ${selected.name.toUpperCase()}`;
  $("tm-drone").style.color = selected.color;
  $("gps").textContent = `${selected.lat.toFixed(6)} N  ${selected.lon.toFixed(6)} E`;
  $("map-coords").textContent = `${selected.lat.toFixed(6)} N / ${selected.lon.toFixed(6)} E`;
  $("alt").textContent = `${fmt(selected.alt, 0)} m`;
  $("heading").textContent = `${String(Math.round(selected.heading)).padStart(3, "0")}°`;
  $("ground-speed").textContent = `${fmt(selected.speed_mps, 0)} m/s`;
  $("leg").textContent = `${Math.min(selected.current_leg + 1, selected.total_legs)} / ${selected.total_legs}`;

  const battery = Math.max(0, selected.battery);
  $("battery-text").textContent = `${battery.toFixed(0)}%`;
  const batteryFill = $("battery-fill");
  batteryFill.style.width = `${battery}%`;
  batteryFill.style.background = battery < 20 ? "linear-gradient(90deg,#7d2519,#ff5b3d)" : `linear-gradient(90deg,${selected.color}88,${selected.color})`;
  $("compass-needle").style.transform = `rotate(${selected.heading}deg)`;
}

function render(state) {
  const drones = state.drones || [];
  if (!drones.length) return;
  const selected = drones.find((drone) => drone.key === selectedDrone) || drones[0];
  const launched = drones.filter((drone) => drone.launched);
  const airborne = launched.filter((drone) => drone.phase !== "done");
  const running = state.running;

  $("mission-clock").textContent = formatClock(state.sim_time);
  const repairOps = running && airborne.length === 0;
  let phaseText = "STANDBY";
  if (repairOps) phaseText = "REPAIR OPS";
  else if (state.phase === "done") phaseText = "COMPLETE";
  else if (state.phase !== "idle") phaseText = running ? `${airborne.length} AIRBORNE` : PHASE_NAMES[state.phase] || state.phase.toUpperCase();
  $("phase-label").textContent = phaseText;
  $("route-label").textContent = state.mode.toUpperCase();
  $("speed-select").value = String(state.speed);
  $("mode-select").value = state.mode;

  const pill = $("status-pill");
  const pillClass = running ? "running" : state.phase === "done" ? "done" : state.phase === "idle" ? "" : "paused";
  pill.className = "status-pill " + pillClass;
  pill.querySelector("span").textContent = running
    ? repairOps
      ? `REPAIR ${state.stats.repaired}/${state.stats.detections}`
      : `IN FLIGHT ${airborne.length}/${drones.length}`
    : state.phase === "done"
      ? "COMPLETE"
      : state.phase === "idle"
        ? "READY"
        : "PAUSED";

  const startButton = $("btn-start");
  if (running) {
    startButton.textContent = "Pause all";
    startButton.classList.add("pause");
  } else if (state.phase === "idle") {
    startButton.textContent = "Launch all";
    startButton.classList.remove("pause");
  } else if (state.phase === "done") {
    startButton.textContent = "New mission";
    startButton.classList.remove("pause");
  } else {
    startButton.textContent = "Resume all";
    startButton.classList.remove("pause");
  }

  renderLaunchButtons(drones);
  renderFleetStrip(drones);
  renderCamera(drones, selected);
  renderTelemetry(selected);

  const stats = state.stats;
  $("stat-coverage").textContent = `${fmt(stats.coverage_pct)}%`;
  $("stat-coverage-note").textContent = `${stats.points_covered || 0} / ${stats.points_total || state.waypoints.length} points · of ${fmt(state.pipeline.total_km)} km`;
  $("coverage-bar").style.width = `${Math.min(100, stats.coverage_pct)}%`;
  $("stat-leaks").textContent = `${stats.detections} / ${stats.leaks_total}`;
  $("stat-error").textContent = stats.avg_error_m ? `${fmt(stats.avg_error_m)} m` : "-- m";
  $("stat-refined").textContent = stats.avg_refined_error_m ? `GWO refined: ${fmt(stats.avg_refined_error_m)} m` : "GWO refined: -- m";
  $("stat-emissions").textContent = `${fmt(stats.emissions_kg_h, 0)} kg/h`;
  $("stat-distance").textContent = `${fmt(stats.distance_km)} km`;
  $("stat-time").textContent = `flight time ${fmt(stats.flight_time_min)} min`;
  $("stat-repairs").textContent = `${stats.repaired || 0} / ${stats.detections || 0}`;
  $("stat-repair-note").textContent = `response ${fmt(stats.avg_response_min)} min · avoided ${fmt(stats.avoided_kg, 0)} kg`;

  document.querySelectorAll(".auto-btn").forEach((button) => {
    button.classList.toggle("active", (button.dataset.auto === "auto") === !!state.auto_mode);
  });

  const ga = state.ga;
  $("ga-improvement").textContent = `${ga.improvement_pct >= 0 ? "-" : "+"}${fmt(Math.abs(ga.improvement_pct))}%`;
  $("ga-baseline").textContent = `${fmt(ga.baseline_distance_km, 2)} km`;
  $("ga-optimized").textContent = `${fmt(ga.distance_km, 2)} km`;
  $("ga-time").textContent = `${fmt(ga.flight_time_min)} min`;
  updateChart(ga);

  renderOrders(state.work_orders);
  renderLogs(state.logs);

  const model = state.model_info;
  if (model) {
    const dice = model.test_dice !== null && model.test_dice !== undefined ? `${(model.test_dice * 100).toFixed(1)}%` : "n/a";
    const iou = model.test_iou !== null && model.test_iou !== undefined ? `${(model.test_iou * 100).toFixed(1)}%` : "n/a";
    $("model-line").textContent = `Model ${model.checkpoint} · test Dice ${dice} · IoU ${iou} · ${model.train_images || "?"} training frames · ${model.image_size.join("x")}`;
  }

  if (!state.detections.length && detectionMarkers.size) {
    window.detectionLayer.clearLayers();
    window.refinedLayer.clearLayers();
    detectionMarkers.clear();
    refinedMarkers.clear();
  }

  drawPipeline(state.pipeline);
  updateWaypoints(state.waypoints);
  updateLeaks(state.leaks);
  updateScans(drones);
  updateDetections(state.detections);
  updateLeakStates(state);
  updateVehicles(state.vehicles);
  updateFleetMap(drones);

  if (state.phase === "done" && lastPhase !== "done") {
    showToast(`Mission complete · ${stats.detections}/${stats.leaks_total} leaks found · ${stats.scans} camera readings · press New mission to fly again`);
  }
  lastPhase = state.phase;
}

function formatClock(seconds) {
  const total = Math.max(0, Math.floor(seconds || 0));
  const minutes = Math.floor(total / 60);
  const secs = total % 60;
  return `${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[character]));
}

async function post(path) {
  try {
    await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" } });
  } catch (error) {
    showToast("Command failed: " + error.message);
  }
}

function showToast(message) {
  let toast = document.querySelector(".toast");
  if (!toast) {
    toast = document.createElement("div");
    toast.className = "toast";
    document.body.appendChild(toast);
  }
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => toast.classList.remove("show"), 3800);
}

function connect() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${protocol}://${location.host}/ws`);
  socket.onmessage = (event) => {
    try {
      state = JSON.parse(event.data);
      render(state);
    } catch (error) {
      console.error(error);
    }
  };
  socket.onclose = () => {
    setTimeout(connect, 1500);
  };
  socket.onerror = () => socket.close();
}

window.addEventListener("DOMContentLoaded", () => {
  initMap();
  connect();

  $("btn-start").addEventListener("click", () => {
    if (state && state.running) post("/api/pause");
    else post("/api/start");
  });
  $("btn-reset").addEventListener("click", () => post("/api/reset"));
  $("btn-optimize").addEventListener("click", async () => {
    await post("/api/optimize");
    gaSignature = "";
    showToast("GA routes re-optimized for all 4 sectors.");
  });
  $("speed-select").addEventListener("change", (event) => post(`/api/speed/${event.target.value}`));
  $("mode-select").addEventListener("change", (event) => post(`/api/mode/${event.target.value}`));
  $("reveal-truth").addEventListener("change", (event) => {
    revealTruth = event.target.checked;
    if (state) updateLeaks(state.leaks);
  });

  document.querySelectorAll(".auto-btn").forEach((button) => {
    button.addEventListener("click", () => post(`/api/automation/${button.dataset.auto}`));
  });
  $("btn-verify").addEventListener("click", () => post("/api/verify"));
  document.addEventListener("click", async (event) => {
    const trigger = event.target.closest("[data-dispatch]");
    if (!trigger) return;
    const leakId = trigger.dataset.dispatch;
    try {
      const response = await fetch(`/api/dispatch/${leakId}`, { method: "POST" });
      const data = await response.json();
      showToast(data.message ? `Leak #${leakId}: ${data.message}` : `Leak #${leakId}: command sent`);
    } catch (error) {
      showToast("Dispatch failed: " + error.message);
    }
  });

  document.querySelectorAll(".drone-btn").forEach((button) => {
    button.addEventListener("click", async () => {
      const key = button.dataset.drone;
      selectedDrone = key;
      const drone = state && (state.drones || []).find((item) => item.key === key);
      if (drone && drone.launched && drone.phase !== "done") {
        try {
          const response = await fetch(`/api/hold/${key}`, { method: "POST" });
          const data = await response.json();
          showToast(data.hold ? `${key} drone on hold` : `${key} drone resumed`);
        } catch (error) {
          showToast("Hold failed: " + error.message);
        }
      } else {
        post(`/api/launch/${key}`);
        showToast(`${key} drone launched`);
      }
    });
  });

  document.querySelectorAll(".feed-tab, .feed-thumb").forEach((element) => {
    element.addEventListener("click", () => {
      selectedDrone = element.dataset.drone;
      if (state) render(state);
    });
  });
});
