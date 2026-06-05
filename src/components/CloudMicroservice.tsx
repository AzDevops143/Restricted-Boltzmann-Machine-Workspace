/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from "react";
import { MicroservicePod, CloudSyncState } from "../types";
import { Server, Database, Cloud, Terminal, CheckCircle2, RotateCcw, FileCode, Play } from "lucide-react";

export default function CloudMicroservice() {
  const [pods, setPods] = useState<MicroservicePod[]>([
    { name: "rbm-inference-replica-1", role: "inference", status: "Running", cpuUsage: 14, memoryUsage: 112, restarts: 0 },
    { name: "rbm-inference-replica-2", role: "inference", status: "Running", cpuUsage: 18, memoryUsage: 118, restarts: 0 },
    { name: "rbm-cd-trainer-worker-1", role: "trainer", status: "Running", cpuUsage: 84, memoryUsage: 340, restarts: 1 },
    { name: "rbm-storage-sync-daemon-1", role: "storage-sync", status: "Running", cpuUsage: 2, memoryUsage: 89, restarts: 0 },
  ]);

  const [syncState, setSyncState] = useState<CloudSyncState>({
    lastSyncTime: null,
    syncedEpochs: [],
    uploadedBytes: 0,
    storageProvider: "Google Cloud Storage",
    bucketName: "rbm-academic-checkpoints",
  });

  const [syncLogs, setSyncSyncLogs] = useState<string[]>([]);
  const [isSyncing, setIsSyncing] = useState(false);
  const [activeManifestTab, setActiveManifestTab] = useState<"dockerfile" | "compose" | "k8s">("dockerfile");
  const [registryFiles, setRegistryFiles] = useState<{ name: string; size: string; time: string }[]>([]);

  // Simulation controls
  const handleScaleUp = () => {
    const nextIdx = pods.filter((p) => p.role === "inference").length + 1;
    const newPod: MicroservicePod = {
      name: `rbm-inference-replica-${nextIdx}`,
      role: "inference",
      status: "Running",
      cpuUsage: 12 + Math.floor(Math.random() * 10),
      memoryUsage: 110 + Math.floor(Math.random() * 20),
      restarts: 0,
    };
    setPods([...pods, newPod]);
  };

  const handleCrashAndRecover = (idx: number) => {
    const updated = [...pods];
    updated[idx].status = "Failed";
    updated[idx].cpuUsage = 0;
    updated[idx].memoryUsage = 0;
    setPods(updated);

    // Auto heal simulation
    setTimeout(() => {
      setPods((prev) => {
        const h = [...prev];
        if (h[idx]) {
          h[idx].status = "Running";
          h[idx].cpuUsage = 10;
          h[idx].memoryUsage = 110;
          h[idx].restarts += 1;
        }
        return h;
      });
    }, 1200);
  };

  const executeCloudSynchronizer = async () => {
    if (isSyncing) return;
    setIsSyncing(true);
    setSyncSyncLogs([]);

    const logList: string[] = [];
    const addLog = (msg: string) => {
      logList.push(`[Sync-Client] ${msg}`);
      setSyncSyncLogs([...logList]);
    };

    addLog("Analyzing model checkpoint telemetry in-memory.");
    addLog(`Reading credentials payload on target domain: ${syncState.storageProvider}`);
    await new Promise((r) => setTimeout(r, 650));

    addLog(`Establishing secure TCP tunnel with bucket: ${syncState.bucketName}`);
    addLog("Formulating JSON weight representation: W (64x16), visible_bias (64), hidden_bias (16)");
    await new Promise((r) => setTimeout(r, 800));

    addLog("Calculating payload SHA-256 checksum: 8a4c84de..e326");
    addLog("Executing multipart stream upload. Buffering chunk parts.");
    await new Promise((r) => setTimeout(r, 700));

    const currTime = new Date().toLocaleTimeString();
    addLog(`HTTP/2 response: 200 OK. Weights synchronized to storage object gs://${syncState.bucketName}/weights-epoch-100-${Date.now().toString().slice(-4)}.json`);

    setSyncState((prev) => ({
      ...prev,
      lastSyncTime: currTime,
      uploadedBytes: prev.uploadedBytes + 12048,
    }));

    setRegistryFiles((prev) => [
      {
        name: `weights-epoch-100-${Date.now().toString().slice(-4)}.json`,
        size: "11.7 KB",
        time: currTime,
      },
      ...prev,
    ]);

    setIsSyncing(false);
  };

  // Manifest files string literals
  const manifests = {
    dockerfile: `# Optimized multi-stage build configuration
FROM python:3.10-slim AS builder

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final minimal runner stage
FROM python:3.10-slim AS runner
WORKDIR /app

COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PORT=3000

EXPOSE 3000
CMD ["python", "rbm_service_server.py"]`,

    compose: `# Multi-service orchestrator mapping components
version: '3.8'

services:
  rbm-inference-service:
    build: .
    image: rbm-inference:latest
    ports:
      - "3000:3000"
    environment:
      - REDIS_HOST=cache-store
      - GCS_BUCKET_NAME=rbm-academic-checkpoints
    deploy:
      replicas: 2
    depends_on:
      - cache-store

  rbm-cd-trainer:
    build: .
    command: ["python", "train_worker.py"]
    environment:
      - BATCH_SIZE=16
    depends_on:
      - database-store

  cache-store:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  database-store:
    image: postgres:15-alpine
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_DB=rbm_performance_db
      - POSTGRES_USER=admin
      - POSTGRES_PASSWORD=securesql`,

    k8s: `# Kubernetes configuration with scale parameters
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rbm-inference-deployment
  labels:
    app: rbm-inference
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rbm-inference
  template:
    metadata:
      labels:
        app: rbm-inference
    spec:
      containers:
      - name: inference-node
        image: gcr.io/academic-rbm-service/rbm-inference:v1.2.0
        ports:
        - containerPort: 3000
        resources:
          limits:
            cpu: "1.0"
            memory: "512Mi"
          requests:
            cpu: "250m"
            memory: "256Mi"
        livenessProbe:
          httpGet:
            path: /api/health
            port: 3000
          initialDelaySeconds: 15
          periodSeconds: 20
        readinessProbe:
          httpGet:
            path: /api/health
            port: 3000
          periodSeconds: 10`,
  };

  return (
    <div className="space-y-6 select-text">
      {/* Topology Header bar */}
      <div className="bg-white border-2 border-[#141414] rounded-none p-5 flex flex-col md:flex-row md:items-center justify-between gap-4 shadow-[4px_4px_0px_0px_rgba(20,20,20,1)]">
        <div className="space-y-1">
          <h2 className="font-display text-lg sm:text-xl font-black text-[#141414] uppercase tracking-tight">
            Microservice Topology & Cloud Storage Hub
          </h2>
          <p className="font-sans text-xs text-zinc-650 font-semibold uppercase tracking-wide">
            Simulate and study container deployment paradigms, horizontal replica auto-healing, and secure checkpoints synchronization with remote databases.
          </p>
        </div>
        <button
          onClick={handleScaleUp}
          className="px-4 py-2 border-2 border-[#141414] bg-[#141414] text-white hover:bg-zinc-800 rounded-none font-mono text-xs font-bold uppercase shadow-[2px_2px_0px_0px_rgba(20,20,20,1)] active:translate-x-0.5 active:translate-y-0.5 active:shadow-none transition-all duration-100 cursor-pointer flex items-center gap-1.5 shrink-0"
        >
          <Server className="w-4 h-4 text-white" />
          Scale Up Inference Pods (+1)
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Kubernetes replicas panel (Left - Spans 6) */}
        <div className="lg:col-span-6 space-y-6">
          <div className="bg-white border-2 border-[#141414] rounded-none p-5 shadow-[4px_4px_0px_0px_rgba(20,20,20,1)] space-y-4">
            <h3 className="font-display text-xs font-black text-black uppercase tracking-wider border-b-2 border-[#141414] pb-2">
              Virtual Host Cluster Orchestrator Replicas
            </h3>

            <div className="space-y-3.5">
              {pods.map((pod, pIdx) => (
                <div
                  key={pod.name}
                  className="p-3.5 border-2 border-[#141414] bg-[#E4E3E0]/10 rounded-none flex items-center justify-between transition-all font-mono shadow-[2px_2px_0px_0px_rgba(20,20,20,1)]"
                >
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <span className={`w-2.5 h-2.5 rounded-full border border-black ${
                        pod.status === "Running" ? "bg-emerald-400" : "bg-amber-400 animate-pulse"
                      }`} />
                      <span className="font-mono text-xs font-black text-black uppercase">{pod.name}</span>
                    </div>
                    <div className="font-mono text-4xs text-zinc-650 font-semibold uppercase">
                      CPU: {pod.cpuUsage}% | MEM: {pod.memoryUsage}MB | RESTARTS: {pod.restarts}
                    </div>
                  </div>

                  <div className="flex items-center gap-2 shrink-0">
                    <span
                      className={`px-2 py-0.5 border border-black rounded-none text-4xs font-mono font-black uppercase ${
                        pod.status === "Running"
                          ? "bg-emerald-100 text-emerald-950"
                          : "bg-amber-100 text-amber-950 animate-pulse"
                      }`}
                    >
                      {pod.status}
                    </span>
                    <button
                      onClick={() => handleCrashAndRecover(pIdx)}
                      title="Crash pod to trigger replica auto-healing"
                      className="px-2 py-1 border-2 border-[#141414] bg-white hover:bg-[#E4E3E0] text-[#141414] rounded-none cursor-pointer text-4xs font-sans font-bold uppercase transition-all shadow-[1px_1px_0px_0px_rgba(20,20,20,1)] active:translate-x-[0.5px] active:translate-y-[0.5px] active:shadow-none"
                    >
                      Simulate Crash
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Secure Cloud Sync Module */}
          <div className="bg-white border-2 border-[#141414] rounded-none p-5 shadow-[4px_4px_0px_0px_rgba(20,20,20,1)] space-y-4">
            <h3 className="font-display text-xs font-black text-black uppercase tracking-wider border-b-2 border-[#141414] pb-2 flex items-center gap-1.5">
              <Cloud className="w-4 h-4 text-black" />
              <span>GCS & S3 Storage Synchronizer</span>
            </h3>

            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="font-mono text-4xs font-black text-zinc-500 block mb-1 uppercase tracking-wider">
                    STORAGE PROVIDER
                  </label>
                  <select
                    value={syncState.storageProvider}
                    onChange={(e) =>
                      setSyncState({ ...syncState, storageProvider: e.target.value as CloudSyncState["storageProvider"] })
                    }
                    className="w-full text-2xs font-sans px-3 py-2 border-2 border-[#141414] rounded-none bg-white font-bold uppercase focus:outline-none focus:bg-[#E4E3E0]/15"
                  >
                    <option value="Google Cloud Storage">Google Cloud Storage (GCS)</option>
                    <option value="Amazon S3">Amazon Simple Storage S3</option>
                  </select>
                </div>
                <div>
                  <label className="font-mono text-4xs font-black text-zinc-500 block mb-1 uppercase tracking-wider">
                    BUCKET DIRECTORY
                  </label>
                  <input
                    type="text"
                    value={syncState.bucketName}
                    onChange={(e) => setSyncState({ ...syncState, bucketName: e.target.value })}
                    className="w-full text-xs font-mono px-3 py-2 border-2 border-[#141414] rounded-none bg-white font-bold uppercase focus:outline-none focus:bg-[#E4E3E0]/15"
                  />
                </div>
              </div>

              <div className="flex">
                <button
                  onClick={executeCloudSynchronizer}
                  disabled={isSyncing}
                  className="px-4 py-2 border-2 border-[#141414] bg-[#141414] text-white hover:bg-zinc-800 font-mono text-xs font-bold uppercase rounded-none shadow-[2px_2px_0px_0px_rgba(20,20,20,1)] active:translate-x-0.5 active:translate-y-0.5 active:shadow-none transition-all duration-100 cursor-pointer disabled:bg-[#E4E3E0]/40 disabled:text-zinc-500 w-full"
                >
                  {isSyncing ? "INITIALIZING SECURE HANDSHAKES..." : "SYNC MODEL CHECKPOINT PARAMETERS"}
                </button>
              </div>

              {/* Sync output Terminal */}
              <div className="bg-black border-2 border-black rounded-none p-4 text-zinc-100 shadow-[2px_2px_0px_0px_rgba(20,20,20,1)]">
                <div className="flex justify-between items-center border-b border-zinc-800 pb-1.5 mb-2">
                  <span className="font-mono text-3xs text-zinc-400 font-bold uppercase tracking-wider">Database Sync Logs</span>
                  <Terminal className="w-3.5 h-3.5 text-zinc-400" />
                </div>
                <div className="h-28 overflow-y-auto font-mono text-4xs leading-normal space-y-1 text-zinc-350 font-semibold select-text">
                  {syncLogs.length === 0 ? (
                    <span className="text-zinc-500 italic block py-4 text-center">No upload processes flagged. Trigger bucket synchronization.</span>
                  ) : (
                    syncLogs.map((lg, i) => <div key={i}>{lg}</div>)
                  )}
                </div>
              </div>

              {/* Upload checklist */}
              {registryFiles.length > 0 && (
                <div className="space-y-1.5 pt-2 border-t border-zinc-100">
                  <span className="font-mono text-4xs text-zinc-500 font-black block uppercase tracking-wider">
                    Bucket Synchronized Objects:
                  </span>
                  <div className="space-y-1.5">
                    {registryFiles.map((f, idx) => (
                      <div key={idx} className="flex justify-between items-center text-4xs font-mono text-zinc-800 bg-[#E4E3E0]/20 p-2 border border-black rounded-none font-bold uppercase">
                        <span className="font-extrabold text-black">{f.name}</span>
                        <span>{f.size} | {f.time}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Configurations Manifest File Viewer (Right - Spans 6) */}
        <div className="lg:col-span-6">
          <div className="bg-white border-2 border-[#141414] rounded-none p-5 shadow-[4px_4px_0px_0px_rgba(20,20,20,1)] space-y-4 h-full flex flex-col justify-between">
            <div className="space-y-4 flex-1">
              <div className="border-b-2 border-[#141414] pb-2 flex justify-between items-center">
                <h3 className="font-display text-xs font-black text-black uppercase tracking-wider">
                  Microservice Container Blueprints
                </h3>
                <FileCode className="w-4 h-4 text-black" />
              </div>

              {/* Selector tabs */}
              <div className="grid grid-cols-3 gap-1.5">
                {(["dockerfile", "compose", "k8s"] as const).map((tab) => (
                  <button
                    key={tab}
                    onClick={() => setActiveManifestTab(tab)}
                    className={`py-2 rounded-none font-mono text-3xs border-2 border-[#141414] uppercase font-bold tracking-wider transition-colors cursor-pointer ${
                      activeManifestTab === tab
                        ? "bg-[#141414] text-white"
                        : "bg-white hover:bg-[#E4E3E0] text-zinc-700"
                    }`}
                  >
                    {tab === "dockerfile" && "Dockerfile"}
                    {tab === "compose" && "compose.yml"}
                    {tab === "k8s" && "kubernetes.yaml"}
                  </button>
                ))}
              </div>

              {/* Configurations Code Frame */}
              <div className="border-2 border-black rounded-none overflow-hidden bg-black font-mono text-3xs text-zinc-200 shadow-[2px_2px_0px_0px_rgba(20,20,20,1)] mt-4">
                <pre className="p-4 max-h-[432px] overflow-y-auto leading-relaxed select-text">
                  <code>{manifests[activeManifestTab]}</code>
                </pre>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
