/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useRef } from "react";
import { RbmWeights, RbmConfig, TrainingLog, PatternData } from "../types";
import {
  initializeRbmWeights,
  runGibbsStep,
  computeHiddenProbabilities,
  computeVisibleProbabilities,
  calculateFreeEnergy,
  trainBatchCdK,
  RBM_PATTERNS,
  addNoise
} from "../rbm";
import { Play, RotateCcw, Sparkles, Flame, RefreshCw, BarChart2, CheckCircle2 } from "lucide-react";

export default function RbmPlayground() {
  // RBM Configuration states
  const [config, setConfig] = useState<RbmConfig>({
    visibleCount: 64, // 8x8 grid
    hiddenCount: 16,
    learningRate: 0.1,
    weightDecay: 0.0001,
    momentum: 0.5,
    cdSteps: 1,
    batchSize: 8,
    epochs: 100,
  });

  // Math States
  const [weights, setWeights] = useState<RbmWeights>(() =>
    initializeRbmWeights(config.visibleCount, config.hiddenCount)
  );

  // Momentum history for tracking optimization
  const prevWGrad = useRef<number[][]>([]);
  const prevvGrad = useRef<number[]>([]);
  const prevhGrad = useRef<number[]>([]);

  // Initialize gradient states
  useEffect(() => {
    prevWGrad.current = Array.from({ length: config.visibleCount }, () =>
      new Array(config.hiddenCount).fill(0)
    );
    prevvGrad.current = new Array(config.visibleCount).fill(0);
    prevhGrad.current = new Array(config.hiddenCount).fill(0);
  }, [config.visibleCount, config.hiddenCount]);

  // UI Active patterns
  const [visibleState, setVisibleState] = useState<number[]>(() => [...RBM_PATTERNS[0].data]);
  const [hiddenProbs, setHiddenProbs] = useState<number[]>(() => new Array(config.hiddenCount).fill(0.5));
  const [hiddenStates, setHiddenStates] = useState<number[]>(() => new Array(config.hiddenCount).fill(0));
  const [reconstructedStates, setReconstructedStates] = useState<number[]>(() => new Array(config.visibleCount).fill(0));
  const [reconstructedProbs, setReconstructedProbs] = useState<number[]>(() => new Array(config.visibleCount).fill(0.1));

  // Visual Interactive Hovers
  const [hoveredVisibleNode, setHoveredVisibleNode] = useState<number | null>(null);
  const [hoveredHiddenNode, setHoveredHiddenNode] = useState<number | null>(null);

  // Parameter controls & training logs
  const [logs, setLogs] = useState<TrainingLog[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [noiseLevel, setNoiseLevel] = useState(0.2);
  const [selectedPatternIndex, setSelectedPatternIndex] = useState(0);

  // Re-initialize weights upon request
  const resetWeights = () => {
    const initW = initializeRbmWeights(config.visibleCount, config.hiddenCount);
    setWeights(initW);
    setLogs([]);
    prevWGrad.current = Array.from({ length: config.visibleCount }, () =>
      new Array(config.hiddenCount).fill(0)
    );
    prevvGrad.current = new Array(config.visibleCount).fill(0);
    prevhGrad.current = new Array(config.hiddenCount).fill(0);

    // Reset hidden and reconstructions
    const hPr = computeHiddenProbabilities(visibleState, initW);
    setHiddenProbs(hPr);
    setHiddenStates(hPr.map((p) => (p > 0.5 ? 1 : 0)));
  };

  // Synchronize weights size if parameter configurations are modified
  useEffect(() => {
    resetWeights();
  }, [config.hiddenCount]);

  // Sync hidden probabilities when visible layer changes
  useEffect(() => {
    const hPr = computeHiddenProbabilities(visibleState, weights);
    setHiddenProbs(hPr);
    setHiddenStates(hPr.map((p) => (p > 0.5 ? 1 : 0)));
  }, [visibleState, weights]);

  // Execute single Gibbs reconstruction step in UI
  const handleGibbsStep = () => {
    const res = runGibbsStep(visibleState, weights);
    setHiddenProbs(res.hiddenProbs);
    setHiddenStates(res.hiddenStates);
    setReconstructedProbs(res.reconstructedProbs);
    setReconstructedStates(res.reconstructedStates);

    // Apply the reconstruction output to visible layers for feedback
    setVisibleState([...res.reconstructedStates]);
  };

  // Perform CD-1 batch training epoch
  const runTrainingEpoch = (batchData: number[][]) => {
    const res = trainBatchCdK(
      batchData,
      weights,
      config,
      config.momentum,
      prevWGrad.current,
      prevvGrad.current,
      prevhGrad.current
    );

    // Record gradients for momentum tracking
    prevWGrad.current = res.weightGradients;
    prevvGrad.current = res.vBiasGradients;
    prevhGrad.current = res.hBiasGradients;

    // Calculate sparsities
    let nearZeros = 0;
    let totalW = 0;
    for (let v = 0; v < config.visibleCount; v++) {
      for (let h = 0; h < config.hiddenCount; h++) {
        if (Math.abs(res.updatedWeights.weights[v][h]) < 0.05) nearZeros++;
        totalW++;
      }
    }

    let actSparsity = 0;
    let avgEnergy = 0;
    batchData.forEach((sample) => {
      const hPr = computeHiddenProbabilities(sample, res.updatedWeights);
      hPr.forEach((p) => {
        if (p > 0.5) actSparsity++;
      });
      avgEnergy += calculateFreeEnergy(sample, res.updatedWeights);
    });

    const activeRatio = actSparsity / (batchData.length * config.hiddenCount);
    avgEnergy = avgEnergy / batchData.length;

    // Apply updated state
    setWeights(res.updatedWeights);

    return {
      reconstructionLoss: res.mse,
      averageFreeEnergy: avgEnergy,
      weightSparsity: nearZeros / totalW,
      hiddenActivationSparsity: activeRatio,
    };
  };

  // Interactive bulk training loop
  const handleTrainRbm = async () => {
    if (isTraining) return;
    setIsTraining(true);

    // Prepare training batch: Expand pattern collection with slight random noise variants to facilitate generalized learning
    const batchData: number[][] = [];
    for (let epochIter = 0; epochIter < 120; epochIter++) {
      RBM_PATTERNS.forEach((p) => {
        batchData.push(addNoise(p.data, 0.05)); // Slight noise adds statistical robustness
      });
    }

    // Split batches
    const batches: number[][][] = [];
    const size = config.batchSize;
    for (let index = 0; index < batchData.length; index += size) {
      batches.push(batchData.slice(index, index + size));
    }

    const epochTarget = config.epochs;
    let currentWeights = { ...weights };
    const tempLogs: TrainingLog[] = [];

    for (let epoch = 1; epoch <= epochTarget; epoch++) {
      let epochMse = 0;
      let epochEnergy = 0;
      let epochWeightSpar = 0;
      let epochActSpar = 0;

      // Run sequential mini-batches
      batches.forEach((batch) => {
        const res = trainBatchCdK(
          batch,
          currentWeights,
          config,
          config.momentum,
          prevWGrad.current,
          prevvGrad.current,
          prevhGrad.current
        );
        currentWeights = res.updatedWeights;
        prevWGrad.current = res.weightGradients;
        prevvGrad.current = res.vBiasGradients;
        prevhGrad.current = res.hBiasGradients;

        epochMse += res.mse;
        epochEnergy += calculateFreeEnergy(batch[0], currentWeights);
      });

      epochMse /= batches.length;
      epochEnergy /= batches.length;

      // Calculate Sparsity
      let nearZeros = 0;
      let totalW = 0;
      for (let v = 0; v < config.visibleCount; v++) {
        for (let h = 0; h < config.hiddenCount; h++) {
          if (Math.abs(currentWeights.weights[v][h]) < 0.05) nearZeros++;
          totalW++;
        }
      }
      epochWeightSpar = nearZeros / totalW;

      // Hidden layer response for primary pattern
      const hPr = computeHiddenProbabilities(visibleState, currentWeights);
      const activeCount = hPr.filter((p) => p > 0.5).length;
      epochActSpar = activeCount / config.hiddenCount;

      tempLogs.push({
        epoch,
        reconstructionLoss: epochMse,
        averageFreeEnergy: epochEnergy,
        weightSparsity: epochWeightSpar,
        hiddenActivationSparsity: epochActSpar,
      });

      // Update state incrementally to show animation progress
      if (epoch % 5 === 0 || epoch === epochTarget) {
        setWeights(currentWeights);
        setLogs([...tempLogs]);
        // Slight yield for rendering loop smoothness
        await new Promise((r) => setTimeout(r, 15));
      }
    }

    setIsTraining(false);
  };

  // UI Interactive operations: Node changes
  const toggleVisibleCell = (idx: number) => {
    const nextState = [...visibleState];
    nextState[idx] = nextState[idx] === 1.0 ? 0.0 : 1.0;
    setVisibleState(nextState);
  };

  // Load a preset pattern with custom noise addition
  const applyPresetPattern = (idx: number, noiseLvl: number) => {
    setSelectedPatternIndex(idx);
    const orig = RBM_PATTERNS[idx].data;
    const noisyVec = addNoise(orig, noiseLvl);
    setVisibleState(noisyVec);
  };

  // Grid coordinates mapping index to row/col
  const getCellLabel = (index: number) => {
    const row = Math.floor(index / 8);
    const col = index % 8;
    return `V[${row},${col}]`;
  };

  // Render responsive SVG line graphs without heavy recharts dependencies
  const renderSvgGraph = () => {
    if (logs.length === 0) {
      return (
        <div className="h-44 border-2 border-dashed border-[#141414]/30 rounded-none flex items-center justify-center bg-white">
          <p className="font-mono text-xs font-bold text-zinc-500 uppercase">Await training telemetry execution logs...</p>
        </div>
      );
    }

    const chartWidth = 520;
    const chartHeight = 160;
    const padding = 25;

    const maxLoss = Math.max(...logs.map((l) => l.reconstructionLoss), 0.01);
    const minLoss = Math.min(...logs.map((l) => l.reconstructionLoss), 0);

    // Compute coordinate points
    const lossPoints = logs
      .map((l, i) => {
        const x = padding + (i / (logs.length - 1)) * (chartWidth - padding * 2);
        // Map loss (min to max) inverse height
        const range = maxLoss - minLoss;
        const normY = (l.reconstructionLoss - minLoss) / (range || 1);
        const y = chartHeight - padding - normY * (chartHeight - padding * 2);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");

    return (
      <div className="w-full bg-white border-2 border-[#141414] rounded-none p-4 shadow-[4px_4px_0px_0px_rgba(20,20,20,1)]">
        <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-2 mb-3 border-b-2 border-dashed border-[#141414]/20 pb-2">
          <div className="flex items-center gap-2">
            <BarChart2 className="w-4 h-4 text-[#141414]" />
            <span className="font-mono text-xs font-bold text-[#141414] uppercase tracking-tight">
              Convergence Profile: CD-{config.cdSteps} Loss
            </span>
          </div>
          <span className="font-mono text-3xs font-extrabold text-[#141414] bg-[#E4E3E0] px-2 py-0.5 border border-[#141414] uppercase">
            MSE: {logs[logs.length - 1].reconstructionLoss.toFixed(4)} at Ep {logs[logs.length - 1].epoch}
          </span>
        </div>
        <svg viewBox={`0 0 ${chartWidth} ${chartHeight}`} className="w-full h-40 overflow-visible text-[#141414]">
          {/* Guide horizontal lines */}
          <line x1={padding} y1={padding} x2={chartWidth - padding} y2={padding} stroke="#141414" strokeWidth="1" strokeDasharray="3 3" opacity="0.15" />
          <line
            x1={padding}
            y1={chartHeight / 2}
            x2={chartWidth - padding}
            y2={chartHeight / 2}
            stroke="#141414"
            strokeWidth="1"
            strokeDasharray="3 3"
            opacity="0.15"
          />
          <line
            x1={padding}
            y1={chartHeight - padding}
            x2={chartWidth - padding}
            y2={chartHeight - padding}
            stroke="#141414"
            strokeWidth="2"
          />

          {/* Boundaries labels */}
          <text x={padding - 5} y={padding + 4} textAnchor="end" className="font-mono text-3xs font-bold fill-[#141414]">
            {maxLoss.toFixed(2)}
          </text>
          <text x={padding - 5} y={chartHeight - padding + 4} textAnchor="end" className="font-mono text-3xs font-bold fill-[#141414]">
            {minLoss.toFixed(2)}
          </text>
          <text x={padding} y={chartHeight - 6} className="font-mono text-3xs font-bold fill-[#141414]/60">
            EP 1
          </text>
          <text x={chartWidth - padding} y={chartHeight - 6} textAnchor="end" className="font-mono text-3xs font-bold fill-[#141414]/60">
            EP {logs.length}
          </text>

          {/* Plotting Line curve */}
          <polyline fill="none" stroke="currentColor" strokeWidth="3" points={lossPoints} className="transition-all" />

          {/* Data Points hover overlays */}
          {logs.length > 0 && (
            <circle
              cx={padding + (chartWidth - padding * 2)}
              cy={chartHeight - padding - ((logs[logs.length - 1].reconstructionLoss - minLoss) / (maxLoss - minLoss || 1)) * (chartHeight - padding * 2)}
              r="6"
              fill="#ff4a4a"
              stroke="#141414"
              strokeWidth="2"
            />
          )}
        </svg>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      {/* Title block detailing simulation properties */}
      <div className="bg-white border-2 border-[#141414] rounded-none p-5 flex flex-col md:flex-row md:items-center justify-between gap-4 shadow-[4px_4px_0px_0px_rgba(20,20,20,1)]">
        <div className="space-y-1">
          <h2 className="font-display text-lg sm:text-xl font-black text-[#141414] uppercase tracking-tight">
            Restricted Boltzmann Machine Stochastic Playground
          </h2>
          <p className="font-sans text-xs text-zinc-600 font-semibold uppercase tracking-wide">
            Interactive playground demonstrating contrastive divergence (CD-k), visible unit repairing, and latent feature detectors.
          </p>
        </div>
        <div className="flex flex-wrap gap-2 shrink-0">
          <button
            onClick={resetWeights}
            className="px-3.5 py-2 border-2 border-[#141414] bg-white hover:bg-[#E4E3E0] text-[#141414] rounded-none font-mono text-xs font-bold uppercase flex items-center gap-1.5 shadow-[2px_2px_0px_0px_rgba(20,20,20,1)] cursor-pointer active:translate-x-[1px] active:translate-y-[1px] active:shadow-none transition-all duration-100"
          >
            <RotateCcw className="w-3.5 h-3.5 text-[#141414]" />
            Initialize Weights
          </button>
          <button
            onClick={handleTrainRbm}
            disabled={isTraining}
            className={`px-4 py-2 border-2 border-[#141414] font-mono text-xs font-bold uppercase transition-all duration-100 flex items-center gap-1.5 cursor-pointer rounded-none shadow-[2px_2px_0px_0px_rgba(20,20,20,1)] active:translate-x-[1px] active:translate-y-[1px] active:shadow-none ${
              isTraining
                ? "bg-[#E4E3E0] text-zinc-500 cursor-not-allowed"
                : "bg-[#141414] text-white hover:bg-zinc-850"
            }`}
          >
            <Play className={`w-3.5 h-3.5 ${isTraining ? "animate-spin" : ""}`} />
            {isTraining ? "Training Models..." : "Train RBM Model (100 Epochs)"}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Play control room configs (Left Col - spans 4) */}
        <div className="lg:col-span-4 space-y-5">
          <div className="bg-white border-2 border-[#141414] rounded-none p-5 shadow-[4px_4px_0px_0px_rgba(20,20,20,1)] space-y-4">
            <h3 className="font-display text-xs font-black text-[#141414] uppercase tracking-wider border-b-2 border-[#141414] pb-2 mb-4">
              Hyperparameters Console
            </h3>

            {/* Config Fields */}
            <div className="space-y-4">
              <div>
                <label className="flex justify-between font-mono text-3xs text-zinc-700 font-bold uppercase tracking-wider mb-1.5">
                  <span>HIDDEN NEURONS (H)</span>
                  <span className="text-[#141414] font-extrabold">{config.hiddenCount} Nodes</span>
                </label>
                <select
                  value={config.hiddenCount}
                  onChange={(e) => setConfig({ ...config, hiddenCount: Number(e.target.value) })}
                  className="w-full text-xs font-mono font-bold uppercase px-2.5 py-2 border-2 border-[#141414] rounded-none bg-white focus:outline-none focus:bg-[#E4E3E0]/30 transition-all cursor-pointer"
                >
                  <option value="8">8 Nodes (Sub-feature resolution)</option>
                  <option value="16">16 Nodes (Standard balance)</option>
                  <option value="24">24 Nodes (High resolution)</option>
                  <option value="32">32 Nodes (Extensive latent space)</option>
                </select>
              </div>

              <div>
                <label className="flex justify-between font-mono text-3xs text-zinc-700 font-bold uppercase tracking-wider mb-1.5">
                  <span>LEARNING RATE (ETA)</span>
                  <span className="text-[#141414] font-extrabold">{config.learningRate}</span>
                </label>
                <div className="flex gap-2 items-center">
                  <input
                    type="range"
                    min="0.01"
                    max="0.5"
                    step="0.01"
                    value={config.learningRate}
                    onChange={(e) => setConfig({ ...config, learningRate: parseFloat(e.target.value) })}
                    className="w-full accent-[#141414] h-2 border-2 border-[#141414] bg-[#E4E3E0] rounded-none appearance-none cursor-pointer"
                  />
                </div>
              </div>

              <div>
                <label className="flex justify-between font-mono text-3xs text-zinc-700 font-bold uppercase tracking-wider mb-1.5">
                  <span>CD SAMPLING STEPS (CD-K)</span>
                  <span className="text-zinc-950 font-extrabold">k = {config.cdSteps}</span>
                </label>
                <div className="grid grid-cols-3 gap-2">
                  {[1, 3, 10].map((kVal) => (
                    <button
                      key={kVal}
                      type="button"
                      onClick={() => setConfig({ ...config, cdSteps: kVal })}
                      className={`py-1.5 rounded-none font-mono text-xs font-bold border-2 border-[#141414] transition-all cursor-pointer ${
                        config.cdSteps === kVal
                          ? "bg-[#141414] text-white"
                          : "bg-white hover:bg-[#E4E3E0] text-[#141414]"
                      }`}
                    >
                      CD-{kVal}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <label className="flex justify-between font-mono text-3xs text-zinc-700 font-bold uppercase tracking-wider mb-1.5">
                  <span>BATCH SIZE</span>
                  <span className="text-[#141414] font-extrabold">{config.batchSize} Samples</span>
                </label>
                <select
                  value={config.batchSize}
                  onChange={(e) => setConfig({ ...config, batchSize: Number(e.target.value) })}
                  className="w-full text-xs font-mono font-bold uppercase px-2.5 py-2 border-2 border-[#141414] rounded-none bg-white focus:outline-none transition-all cursor-pointer"
                >
                  <option value="4">4 (Slow robust steps)</option>
                  <option value="8">8 (Recommended balance)</option>
                  <option value="16">16 (Fast average steps)</option>
                </select>
              </div>
            </div>
          </div>

          <div className="bg-white border-2 border-[#141414] rounded-none p-5 shadow-[4px_4px_0px_0px_rgba(20,20,20,1)] space-y-4">
            <h3 className="font-display text-xs font-black text-[#141414] uppercase tracking-wider border-b-2 border-[#141414] pb-2 mb-4">
              Academic Pattern Presets
            </h3>
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-2">
                {RBM_PATTERNS.map((p, pIdx) => (
                  <button
                    key={p.name}
                    onClick={() => applyPresetPattern(pIdx, noiseLevel)}
                    className={`px-2 py-2 text-left border-2 border-[#141414] rounded-none text-2xs font-mono font-bold transition-all truncate cursor-pointer uppercase ${
                      selectedPatternIndex === pIdx
                        ? "bg-[#141414] text-white"
                        : "bg-white hover:bg-[#E4E3E0] text-[#141414] shadow-[1.5px_1.5px_0px_0px_rgba(20,20,20,1)] active:translate-x-[0.5px] active:translate-y-[0.5px] active:shadow-none"
                    }`}
                  >
                    {p.name}
                  </button>
                ))}
              </div>

              <div className="space-y-2 border-t-2 border-dashed border-zinc-200 pt-3.5">
                <div className="flex justify-between items-center text-3xs font-mono text-zinc-700 font-bold uppercase tracking-wider">
                  <span>ADD RANDOM NOISE</span>
                  <span className="text-[#141414] font-extrabold">{(noiseLevel * 100).toFixed(0)}%</span>
                </div>
                <div className="flex gap-2">
                  <input
                    type="range"
                    min="0"
                    max="0.8"
                    step="0.05"
                    value={noiseLevel}
                    onChange={(e) => {
                      const lvl = parseFloat(e.target.value);
                      setNoiseLevel(lvl);
                      applyPresetPattern(selectedPatternIndex, lvl);
                    }}
                    className="w-full accent-[#141414] h-2 border-2 border-[#141414] bg-[#E4E3E0] rounded-none appearance-none cursor-pointer"
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Quick Info details */}
          <div className="bg-white border-2 border-[#141414] rounded-none p-4 text-3xs text-[#141414] font-mono space-y-1.5 shadow-[2px_2px_0px_0px_rgba(20,20,20,1)] uppercase">
            <span className="font-extrabold block text-2xs text-[#141414] border-b border-zinc-200 pb-1">Energy State Theory:</span>
            <span className="block font-semibold text-zinc-650 leading-relaxed text-justify">
              Free Energy (F) calculates model confidence. High noise yields higher F, while structured, recognized patterns drive energy down.
            </span>
          </div>
        </div>

        {/* Vis Sandbox Grid & Nodes (Center/Right Col - spans 8) */}
        <div className="lg:col-span-8 space-y-5">
          <div className="bg-white border-2 border-[#141414] rounded-none p-5 shadow-[4px_4px_0px_0px_rgba(20,20,20,1)] space-y-6">
            <h3 className="font-display text-xs font-black text-[#141414] uppercase tracking-wider border-b-2 border-[#141414] pb-2">
              RBM Computational Pipeline Simulation
            </h3>

            {/* Flow Visualizer mapping V - W - H */}
            <div className="grid grid-cols-1 md:grid-cols-7 gap-5 items-center">
              {/* 1. Visible Layer Grid Input (Colspan 2) */}
              <div className="md:col-span-2 space-y-3 text-center">
                <span className="font-mono text-3xs text-zinc-700 font-bold tracking-wider block uppercase">
                  VISIBLE LAYER (V) - 8x8 GRID
                </span>
                <div className="inline-block p-1.5 border-2 border-[#141414] bg-[#E4E3E0]/20 rounded-none shadow-[2px_2px_0px_0px_rgba(20,20,20,1)]">
                  <div className="grid grid-cols-8 gap-1 w-40 h-40">
                    {visibleState.map((cell, idx) => (
                      <div
                        key={idx}
                        onClick={() => toggleVisibleCell(idx)}
                        onMouseEnter={() => setHoveredVisibleNode(idx)}
                        onMouseLeave={() => setHoveredVisibleNode(null)}
                        className={`aspect-square border cursor-pointer transition-all duration-75 rounded-none flex items-center justify-center ${
                          cell === 1.0
                            ? "bg-[#141414] border-[#141414] scale-102 hover:bg-zinc-800"
                            : "bg-white border-zinc-250 hover:bg-zinc-100"
                        } ${hoveredVisibleNode === idx ? "ring-2 ring-amber-500" : ""}`}
                        title={getCellLabel(idx)}
                      />
                    ))}
                  </div>
                </div>
                <div className="flex gap-2 justify-center">
                  <button
                    onClick={() => setVisibleState(new Array(64).fill(0))}
                    className="px-2 py-1 text-3xs font-mono font-bold uppercase border-2 border-[#141414] bg-white hover:bg-[#E4E3E0] text-[#141414] rounded-none cursor-pointer shadow-[1px_1px_0px_0px_rgba(20,20,20,1)] active:translate-x-[0.5px] active:translate-y-[0.5px] active:shadow-none"
                  >
                    Clear Vector
                  </button>
                  <button
                    onClick={() => applyPresetPattern(selectedPatternIndex, noiseLevel)}
                    className="px-2 py-1 text-3xs font-mono font-bold uppercase border-2 border-[#141414] bg-white hover:bg-[#E4E3E0] text-[#141414] rounded-none cursor-pointer shadow-[1px_1px_0px_0px_rgba(20,20,20,1)] active:translate-x-[0.5px] active:translate-y-[0.5px] active:shadow-none"
                  >
                    Reset Pattern
                  </button>
                </div>
              </div>

              {/* 2. Gibbs Sampling Operator Arrows (Colspan 1) */}
              <div className="md:col-span-1 flex flex-col items-center justify-center gap-3">
                <button
                  onClick={handleGibbsStep}
                  title="Run reconstruction step"
                  className="p-3 border-2 border-black hover:bg-[#E4E3E0] rounded-full cursor-pointer bg-white shadow-[2px_2px_0px_0px_rgba(20,20,20,1)] active:translate-x-0.5 active:translate-y-0.5 active:shadow-none transition-all duration-100"
                >
                  <RefreshCw className="w-5 h-5 text-black hover:rotate-45 transition-transform" />
                </button>
                <span className="font-mono text-3xs text-zinc-900 font-extrabold tracking-wider text-center uppercase leading-tight">
                  GIBBS<br/>SAMPLING
                </span>
              </div>

              {/* 3. Reconstructed Probabilities Grid Output (Colspan 2) */}
              <div className="md:col-span-2 space-y-3 text-center">
                <span className="font-mono text-3xs text-zinc-700 font-bold tracking-wider block uppercase">
                  RECONSTRUCTION P(V'|h)
                </span>
                <div className="inline-block p-1.5 border-2 border-[#141414] bg-[#E4E3E0]/20 rounded-none shadow-[2px_2px_0px_0px_rgba(20,20,20,1)]">
                  <div className="grid grid-cols-8 gap-1 w-40 h-40">
                    {reconstructedProbs.map((prob, idx) => {
                      const colorScale = Math.round(prob * 255);
                      // Custom inline color variable to avoid Tailwind dynamic class stripping
                      const cellStyle = {
                        backgroundColor: `rgb(${255 - colorScale}, ${255 - Math.round(colorScale * 0.4)}, ${255 - colorScale})`,
                      };
                      return (
                        <div
                          key={idx}
                          style={cellStyle}
                          className="aspect-square border border-[#141414]/30 rounded-none"
                          title={`Reconstructed coordinate V'[${idx}]: ${(prob * 100).toFixed(1)}%`}
                        />
                      );
                    })}
                  </div>
                </div>
                <div className="text-3xs font-mono font-bold uppercase text-zinc-700">
                  Free Energy: <span className="font-extrabold text-[#141414] bg-zinc-100 px-1 border border-zinc-300">{calculateFreeEnergy(visibleState, weights).toFixed(2)}</span>
                </div>
              </div>

              {/* 4. Connection Network weight mapping (Colspan 2) */}
              <div className="md:col-span-2 space-y-3 bg-[#E4E3E0] border-2 border-[#141414] rounded-none p-3.5 shadow-inner">
                <span className="font-mono text-3xs text-zinc-900 font-extrabold tracking-wider block uppercase border-b border-[#141414]/20 pb-1">
                  WEIGHT EXAMINER (W)
                </span>
                {hoveredVisibleNode !== null ? (
                  <div className="space-y-2">
                    <div className="font-bold text-3xs text-zinc-900 font-mono uppercase">
                      Connections for {getCellLabel(hoveredVisibleNode)}
                    </div>
                    <div className="space-y-1 max-h-32 overflow-y-auto pr-1">
                      {weights.weights[hoveredVisibleNode].slice(0, 8).map((w, hIdx) => (
                        <div key={hIdx} className="flex justify-between items-center font-mono text-3xs font-semibold">
                          <span className="text-zinc-650">To Hidden H[{hIdx}]</span>
                          <span className={w >= 0 ? "text-emerald-800 font-extrabold" : "text-red-800 font-extrabold"}>
                            {w >= 0 ? "+" : ""}
                            {w.toFixed(3)}
                          </span>
                        </div>
                      ))}
                      <div className="text-4xs text-zinc-650 font-bold uppercase text-center pt-1 border-t border-[#141414]/10">
                        Showing first 8 nodes
                      </div>
                    </div>
                  </div>
                ) : hoveredHiddenNode !== null ? (
                  <div className="space-y-2">
                    <div className="font-bold text-3xs text-[#141414] font-mono uppercase">
                      Hidden Node H[{hoveredHiddenNode}]
                    </div>
                    <div className="font-mono text-4xs text-zinc-700 leading-normal font-semibold">
                      Represented as receptive field on the left grid.
                    </div>
                  </div>
                ) : (
                  <p className="font-mono text-4xs text-zinc-700 leading-normal font-semibold uppercase">
                    Hover on any cell of input grid to inspect weight parameters!
                  </p>
                )}
              </div>
            </div>

            {/* Hidden neurons rendering blocks */}
            <div className="space-y-3 border-t-2 border-[#141414] pt-5">
              <span className="font-mono text-3xs text-zinc-900 font-extrabold tracking-wider block text-center md:text-left uppercase">
                LATENT FEATURE DETECTORS: HIDDEN LAYER (H) ACTIVATION PROBABILITIES P(H|v)
              </span>
              <div className="flex flex-wrap gap-2 justify-center md:justify-start">
                {hiddenProbs.map((prob, hIdx) => {
                  const isActive = prob > 0.5;
                  const intensity = Math.round(prob * 100);
                  return (
                    <div
                      key={hIdx}
                      onMouseEnter={() => setHoveredHiddenNode(hIdx)}
                      onMouseLeave={() => setHoveredHiddenNode(null)}
                      className={`px-3 py-2 border-2 border-[#141414] rounded-none text-center font-mono text-2xs transition-all flex flex-col items-center min-w-16 cursor-pointer relative ${
                        isActive
                          ? "bg-[#141414] text-white"
                          : "bg-white text-[#141414] hover:bg-[#E4E3E0]"
                      } ${hoveredHiddenNode === hIdx ? "ring-2 ring-amber-500" : ""}`}
                    >
                      <span className={`font-extrabold block text-4xs ${isActive ? "text-zinc-450" : "text-zinc-600"}`}>H[{hIdx}]</span>
                      <span className="font-bold">
                        {intensity}%
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Hidden Neuron Receptive Fields visualizer */}
            <div className="space-y-3.5 border-t-2 border-[#141414] pt-5">
              <h4 className="font-display text-xs font-black text-[#141414] uppercase tracking-wide">
                Feature Detectors (Hidden Unit Receptive Fields: W_j projected to 8x8)
              </h4>
              <p className="font-mono text-4xs text-zinc-650 leading-relaxed max-w-2xl font-semibold uppercase">
                Every hidden unit learns to respond to specific spatial motifs. Black represents positive weight couplings, white represents negative. Edge filters self-organize automatically here!
              </p>
              <div className="grid grid-cols-4 sm:grid-cols-8 gap-3">
                {Array.from({ length: config.hiddenCount }).map((_, hIdx) => {
                  // Compute receptive field for node hIdx
                  const rfWeights = Array.from({ length: 64 }).map(
                    (_, vIdx) => weights.weights[vIdx][hIdx]
                  );
                  const minW = Math.min(...rfWeights, -0.01);
                  const maxW = Math.max(...rfWeights, 0.01);

                  return (
                    <div
                      key={hIdx}
                      className={`p-1.5 border-2 border-[#141414] rounded-none bg-white flex flex-col items-center transition-all ${
                        hoveredHiddenNode === hIdx ? "bg-amber-100 shadow-[2px_2px_0px_0px_rgba(20,20,20,1)] -translate-x-0.5 -translate-y-0.5" : ""
                      }`}
                    >
                      <div className="grid grid-cols-8 gap-0.5 w-10 h-10 mb-1.5 border border-zinc-200 pointer-events-none">
                        {rfWeights.map((wVal, i) => {
                          const normW = (wVal - minW) / (maxW - minW || 1);
                          const rgbVal = Math.round(normW * 255);
                          const cellColor = {
                            backgroundColor: `rgb(${rgbVal}, ${Math.round(rgbVal * 0.95)}, ${255 - rgbVal})`,
                          };
                          return <div key={i} style={cellColor} className="aspect-square" />;
                        })}
                      </div>
                      <span className="font-mono text-4xs text-[#141414] font-bold">RF[{hIdx}]</span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Model loss graph convergence */}
          {renderSvgGraph()}
        </div>
      </div>
    </div>
  );
}
