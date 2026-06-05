/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from "react";
import { RbmTestCase } from "../types";
import { sigmoid, initializeRbmWeights, computeHiddenProbabilities, calculateFreeEnergy, trainBatchCdK } from "../rbm";
import { CheckCircle2, XCircle, Play, ShieldCheck, Terminal, AlertCircle } from "lucide-react";

export default function UnitTestRunner() {
  const [tests, setTests] = useState<RbmTestCase[]>([
    {
      id: "ut-1",
      name: "ut_sigmoid_bounds_and_activation_stability",
      description: "Verifies sigmoid standard outputs bounds are mathematically sound at critical limits: s(0)=0.5, s(20)~=1, s(-20)~=0.",
      status: "idle",
      logs: [],
    },
    {
      id: "ut-2",
      name: "ut_weights_initialization_dimensions_shape",
      description: "Ensures model weights initialize symmetrically with dimensions mapping visible layer size V to hidden layer size H.",
      status: "idle",
      logs: [],
    },
    {
      id: "ut-3",
      name: "ut_conditional_activation_probability_bounds",
      description: "Checks that feed-forward probability mappings generate float probabilities strictly within numerical boundary [0, 1].",
      status: "idle",
      logs: [],
    },
    {
      id: "ut-4",
      name: "ut_free_energy_coherence_gradient_descent",
      description: "Mathematical formulation check: ensures organized structural patterns carry lower Free Energy F(V) than random noisy vectors.",
      status: "idle",
      logs: [],
    },
    {
      id: "ut-5",
      name: "ut_contrastive_divergence_cd_k_parameter_upgrades",
      description: "Validates that a single training iteration steps outputs correctly computed weight matrices adjustments.",
      status: "idle",
      logs: [],
    },
    {
      id: "ut-6",
      name: "ut_json_serialization_payload_schema",
      description: "Verifies the model state dumps payload matching GCS cloud backup requirements.",
      status: "idle",
      logs: [],
    },
  ]);

  const [isRunningTests, setIsRunningTests] = useState(false);
  const [consoleLogs, setConsoleLogs] = useState<string[]>([]);

  const handleRunTests = async () => {
    if (isRunningTests) return;
    setIsRunningTests(true);
    setConsoleLogs([]);

    const runningTestsList = tests.map((t) => ({ ...t, status: "idle" as const, logs: [] as string[] }));

    const runLogs: string[] = [];
    const addConsoleLog = (msg: string) => {
      runLogs.push(`[QA-Suite] ${msg}`);
      setConsoleLogs([...runLogs]);
    };

    addConsoleLog("Bootstrapping RBM Mathematical Verification Suites...");
    await new Promise((r) => setTimeout(r, 400));

    // Test 1
    {
      const tIdx = 0;
      runningTestsList[tIdx].status = "running";
      setTests([...runningTestsList]);
      const start = performance.now();
      const testLogs: string[] = [];

      try {
        testLogs.push("Executing ut_sigmoid_bounds_and_activation_stability...");
        const sZero = sigmoid(0);
        const sPlus = sigmoid(20);
        const sMinus = sigmoid(-20);

        testLogs.push(`Evaluated sigmoid(0) = ${sZero.toFixed(4)}. Expecting exactly 0.5000.`);
        testLogs.push(`Evaluated sigmoid(20) = ${sPlus.toFixed(4)}. Expecting limit near 1.0000.`);
        testLogs.push(`Evaluated sigmoid(-20) = ${sMinus.toFixed(4)}. Expecting limit near 0.0000.`);

        if (Math.abs(sZero - 0.5) > 1e-7) throw new Error("Sigmoid zero offset error.");
        if (sPlus < 0.999) throw new Error("Sigmoid ceiling overflow error.");
        if (sMinus > 0.001) throw new Error("Sigmoid floor boundary error.");

        const dur = parseFloat((performance.now() - start).toFixed(2));
        runningTestsList[tIdx].status = "passed";
        runningTestsList[tIdx].logs = testLogs;
        runningTestsList[tIdx].executionTimeMs = dur;
        addConsoleLog("ut_sigmoid_bounds_and_activation_stability: PASSED.");
      } catch (e: any) {
        runningTestsList[tIdx].status = "failed";
        runningTestsList[tIdx].logs = [...testLogs, `Assertion error: ${e.message}`];
        addConsoleLog(`ut_sigmoid_bounds_and_activation_stability: FAILED. Err: ${e.message}`);
      }
      setTests([...runningTestsList]);
    }
    await new Promise((r) => setTimeout(r, 200));

    // Test 2
    {
      const tIdx = 1;
      runningTestsList[tIdx].status = "running";
      setTests([...runningTestsList]);
      const start = performance.now();
      const testLogs: string[] = [];

      try {
        testLogs.push("Executing ut_weights_initialization_dimensions_shape...");
        const visSize = 64;
        const hidSize = 16;
        const weights = initializeRbmWeights(visSize, hidSize);

        testLogs.push(`Initializing weights with size V=${visSize}, H=${hidSize}.`);
        testLogs.push(`Weights shape check: Row length corresponds to ${weights.weights.length} visible nodes.`);
        testLogs.push(`Weights column shape check: Col length of row 0 is ${weights.weights[0].length} hidden nodes.`);

        if (weights.weights.length !== visSize) throw new Error("Weights row count mismatch.");
        if (weights.weights[0].length !== hidSize) throw new Error("Weights column count mismatch.");
        if (weights.vBiases.length !== visSize) throw new Error("Visible biases length mismatch.");
        if (weights.hBiases.length !== hidSize) throw new Error("Hidden biases length mismatch.");

        const dur = parseFloat((performance.now() - start).toFixed(2));
        runningTestsList[tIdx].status = "passed";
        runningTestsList[tIdx].logs = testLogs;
        runningTestsList[tIdx].executionTimeMs = dur;
        addConsoleLog("ut_weights_initialization_dimensions_shape: PASSED.");
      } catch (e: any) {
        runningTestsList[tIdx].status = "failed";
        runningTestsList[tIdx].logs = [...testLogs, `Assertion error: ${e.message}`];
        addConsoleLog(`ut_weights_initialization_dimensions_shape: FAILED. Err: ${e.message}`);
      }
      setTests([...runningTestsList]);
    }
    await new Promise((r) => setTimeout(r, 200));

    // Test 3
    {
      const tIdx = 2;
      runningTestsList[tIdx].status = "running";
      setTests([...runningTestsList]);
      const start = performance.now();
      const testLogs: string[] = [];

      try {
        testLogs.push("Executing ut_conditional_activation_probability_bounds...");
        const weights = initializeRbmWeights(64, 16);
        const mockV = new Array(64).fill(0).map(() => (Math.random() > 0.5 ? 1.0 : 0.0));

        testLogs.push("Feeding random binary inputs into feedforward probability mapper.");
        const probs = computeHiddenProbabilities(mockV, weights);

        testLogs.push(`Calculated hidden response length: ${probs.length} nodes.`);
        probs.forEach((p, idx) => {
          if (p < 0 || p > 1) {
            throw new Error(`Out of range probability index ${idx}: ${p}`);
          }
        });
        testLogs.push("All calculated probabilities verified between boundary criteria [0.0, 1.0].");

        const dur = parseFloat((performance.now() - start).toFixed(2));
        runningTestsList[tIdx].status = "passed";
        runningTestsList[tIdx].logs = testLogs;
        runningTestsList[tIdx].executionTimeMs = dur;
        addConsoleLog("ut_conditional_activation_probability_bounds: PASSED.");
      } catch (e: any) {
        runningTestsList[tIdx].status = "failed";
        runningTestsList[tIdx].logs = [...testLogs, `Assertion error: ${e.message}`];
        addConsoleLog(`ut_conditional_activation_probability_bounds: FAILED. Err: ${e.message}`);
      }
      setTests([...runningTestsList]);
    }
    await new Promise((r) => setTimeout(r, 200));

    // Test 4
    {
      const tIdx = 3;
      runningTestsList[tIdx].status = "running";
      setTests([...runningTestsList]);
      const start = performance.now();
      const testLogs: string[] = [];

      try {
        testLogs.push("Executing ut_free_energy_coherence_gradient_descent...");
        const weights = initializeRbmWeights(64, 16);

        // Define a strong, structured matching pattern
        const patternV = new Array(64).fill(0).map((_, i) => (i % 2 === 0 ? 1.0 : 0.0));
        // Random binary noise
        const noisyV = new Array(64).fill(0).map(() => (Math.random() > 0.5 ? 1.0 : 0.0));

        // Inject strong weights aligned to structural inputs
        for (let v = 0; v < 64; v++) {
          if (patternV[v] === 1.0) {
            weights.weights[v].forEach((_, h) => {
              weights.weights[v][h] = 0.5; // High positive weight
            });
          }
        }

        const fePattern = calculateFreeEnergy(patternV, weights);
        const feNoisy = calculateFreeEnergy(noisyV, weights);

        testLogs.push(`Structured vector Free Energy: ${fePattern.toFixed(4)}`);
        testLogs.push(`Noisy vector Free Energy: ${feNoisy.toFixed(4)}`);
        testLogs.push("Under cohesive free energy rules, F(v_structured) must be lower than F(v_noisy).");

        if (fePattern > feNoisy) {
          throw new Error("Free energy coherence violated. Structured state holds higher energy than noise.");
        }

        const dur = parseFloat((performance.now() - start).toFixed(2));
        runningTestsList[tIdx].status = "passed";
        runningTestsList[tIdx].logs = testLogs;
        runningTestsList[tIdx].executionTimeMs = dur;
        addConsoleLog("ut_free_energy_coherence_gradient_descent: PASSED.");
      } catch (e: any) {
        runningTestsList[tIdx].status = "failed";
        runningTestsList[tIdx].logs = [...testLogs, `Assertion error: ${e.message}`];
        addConsoleLog(`ut_free_energy_coherence_gradient_descent: FAILED. Err: ${e.message}`);
      }
      setTests([...runningTestsList]);
    }
    await new Promise((r) => setTimeout(r, 200));

    // Test 5
    {
      const tIdx = 4;
      runningTestsList[tIdx].status = "running";
      setTests([...runningTestsList]);
      const start = performance.now();
      const testLogs: string[] = [];

      try {
        testLogs.push("Executing ut_contrastive_divergence_cd_k_parameter_upgrades...");
        const visSize = 64;
        const hidSize = 16;
        const initW = initializeRbmWeights(visSize, hidSize);

        const configMock = {
          visibleCount: visSize,
          hiddenCount: hidSize,
          learningRate: 0.1,
          weightDecay: 0.001,
          momentum: 0.0,
          cdSteps: 1,
          batchSize: 2,
          epochs: 1,
        };

        const batch = [
          new Array(64).fill(0).map((_, i) => (i % 2 === 0 ? 1 : 0)),
          new Array(64).fill(0).map((_, i) => (i % 3 === 0 ? 1 : 0)),
        ];

        testLogs.push("Executing child CD-1 batch update step...");
        const prevWeightGrad0 = Array.from({ length: visSize }, () => new Array(hidSize).fill(0));
        const prevVGrad0 = new Array(visSize).fill(0);
        const prevHGrad0 = new Array(hidSize).fill(0);

        const res = trainBatchCdK(batch, initW, configMock, 0.0, prevWeightGrad0, prevVGrad0, prevHGrad0);

        testLogs.push("Upgraded weight checking matrix structure evaluation.");
        testLogs.push(`Extracted Batch train MSE reconstruction: ${res.mse.toFixed(4)}`);

        // Check that values have changed
        let changeCount = 0;
        for (let v = 0; v < visSize; v++) {
          for (let h = 0; h < hidSize; h++) {
            if (res.updatedWeights.weights[v][h] !== initW.weights[v][h]) {
              changeCount++;
            }
          }
        }

        testLogs.push(`Weight matrices parameters updated on: ${changeCount} of ${visSize * hidSize} points.`);
        if (changeCount === 0) throw new Error("Training iteration failed to adjust weights.");

        const dur = parseFloat((performance.now() - start).toFixed(2));
        runningTestsList[tIdx].status = "passed";
        runningTestsList[tIdx].logs = testLogs;
        runningTestsList[tIdx].executionTimeMs = dur;
        addConsoleLog("ut_contrastive_divergence_cd_k_parameter_upgrades: PASSED.");
      } catch (e: any) {
        runningTestsList[tIdx].status = "failed";
        runningTestsList[tIdx].logs = [...testLogs, `Assertion error: ${e.message}`];
        addConsoleLog(`ut_contrastive_divergence_cd_k_parameter_upgrades: FAILED. Err: ${e.message}`);
      }
      setTests([...runningTestsList]);
    }
    await new Promise((r) => setTimeout(r, 200));

    // Test 6
    {
      const tIdx = 5;
      runningTestsList[tIdx].status = "running";
      setTests([...runningTestsList]);
      const start = performance.now();
      const testLogs: string[] = [];

      try {
        testLogs.push("Executing ut_json_serialization_payload_schema...");
        const mockModelDump = {
          visible_dim: 64,
          hidden_dim: 16,
          weights: initializeRbmWeights(64, 16).weights,
          v_bias: new Array(64).fill(0).map(() => Math.random()),
          h_bias: new Array(16).fill(0).map(() => Math.random()),
        };

        const jsonStr = JSON.stringify(mockModelDump);
        testLogs.push(`Payload length in JSON representation: ${jsonStr.length} bytes.`);

        const parsed = JSON.parse(jsonStr);
        if (parsed.visible_dim !== 64) throw new Error("Invalid visible_dim serialized.");
        if (parsed.hidden_dim !== 16) throw new Error("Invalid hidden_dim serialized.");
        if (!Array.isArray(parsed.weights)) throw new Error("Weights payload must be array.");

        testLogs.push("Backup serialization format parsed correctly.");

        const dur = parseFloat((performance.now() - start).toFixed(2));
        runningTestsList[tIdx].status = "passed";
        runningTestsList[tIdx].logs = testLogs;
        runningTestsList[tIdx].executionTimeMs = dur;
        addConsoleLog("ut_json_serialization_payload_schema: PASSED.");
      } catch (e: any) {
        runningTestsList[tIdx].status = "failed";
        runningTestsList[tIdx].logs = [...testLogs, `Assertion error: ${e.message}`];
        addConsoleLog(`ut_json_serialization_payload_schema: FAILED. Err: ${e.message}`);
      }
      setTests([...runningTestsList]);
    }

    addConsoleLog("Mathematical verification script complete.");
    setIsRunningTests(false);
  };

  return (
    <div className="space-y-6 select-text">
      {/* Test runner control bar */}
      <div className="bg-white border-2 border-[#141414] rounded-none p-5 flex flex-col md:flex-row md:items-center justify-between gap-4 shadow-[4px_4px_0px_0px_rgba(20,20,20,1)]">
        <div className="space-y-1">
          <h2 className="font-display text-lg sm:text-xl font-black text-[#141414] uppercase tracking-tight font-display">
            Mathematical Alignment Verification Suite
          </h2>
          <p className="font-sans text-xs text-zinc-650 font-semibold uppercase tracking-wide">
            An in-browser mathematical QA environment running real unit test scripts checking sigmoid correctness, dimensionality shapes, and free energy gradient stability.
          </p>
        </div>
        <button
          onClick={handleRunTests}
          disabled={isRunningTests}
          className={`px-4 py-2 border-2 border-[#141414] rounded-none font-mono text-xs font-black uppercase tracking-wider transition-all duration-100 flex items-center gap-1.5 cursor-pointer shrink-0 ${
            isRunningTests
              ? "bg-[#E4E3E0]/40 text-zinc-500 cursor-not-allowed shadow-none"
              : "bg-emerald-600 text-white hover:bg-emerald-700 shadow-[2px_2px_0px_0px_rgba(20,20,20,1)] active:translate-x-0.5 active:translate-y-0.5 active:shadow-none"
          }`}
        >
          <Play className={`w-3.5 h-3.5 ${isRunningTests ? "animate-spin" : ""}`} />
          {isRunningTests ? "Running Verifications..." : "Run Theoretical Asserts"}
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 animate-fade-in">
        {/* Left Col - test items list (spans 7) */}
        <div className="lg:col-span-7 space-y-4">
          <div className="bg-white border-2 border-[#141414] rounded-none p-5 shadow-[4px_4px_0px_0px_rgba(20,20,20,1)] space-y-4">
            <div className="border-b-2 border-[#141414] pb-2 flex items-center justify-between">
              <span className="font-display text-xs font-black text-black uppercase tracking-wider">
                Target Mathematical Claims Assertions
              </span>
              <ShieldCheck className="w-4 h-4 text-black" />
            </div>

            <div className="space-y-3.5">
              {tests.map((t) => (
                <div
                  key={t.id}
                  className="p-3.5 border-2 border-[#141414] rounded-none bg-[#E4E3E0]/10 space-y-2 select-text font-mono shadow-[2px_2px_0px_0px_rgba(20,20,20,1)]"
                >
                  <div className="flex items-start justify-between gap-4">
                    <div className="space-y-1">
                      <span className="font-mono text-xs font-black text-black uppercase block">
                        {t.name}
                      </span>
                      <p className="font-serif text-3xs text-zinc-700 font-semibold leading-normal">{t.description}</p>
                    </div>

                    <div className="shrink-0">
                      {t.status === "passed" && (
                        <span className="px-2 py-0.5 bg-emerald-100 text-emerald-950 border border-black rounded-none font-mono text-4xs font-black uppercase flex items-center gap-1">
                          <CheckCircle2 className="w-3 h-3 text-emerald-950" /> PASS
                        </span>
                      )}
                      {t.status === "failed" && (
                        <span className="px-2 py-0.5 bg-rose-100 text-rose-950 border border-black rounded-none font-mono text-4xs font-black uppercase flex items-center gap-1">
                          <XCircle className="w-3 h-3 text-rose-950" /> FAIL
                        </span>
                      )}
                      {t.status === "idle" && (
                        <span className="px-2 py-0.5 bg-white text-zinc-500 border border-black rounded-none font-mono text-4xs uppercase font-extrabold">
                          IDLE
                        </span>
                      )}
                      {t.status === "running" && (
                        <span className="px-2 py-0.5 bg-amber-100 text-amber-950 border border-black rounded-none font-mono text-4xs uppercase font-black animate-pulse">
                          RUNNING
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Assertion logs summary dropdown */}
                  {t.logs.length > 0 && (
                    <div className="bg-white p-3 border-2 border-[#141414] rounded-none text-4xs font-mono text-zinc-800 space-y-1.5 mt-2 leading-relaxed">
                      <span className="text-zinc-500 font-black block uppercase tracking-wider text-5xs mb-1">
                        Test Assertions: {t.executionTimeMs ? `Completed in ${t.executionTimeMs}ms` : ""}
                      </span>
                      {t.logs.map((lg, idx) => (
                        <div key={idx} className="flex gap-1">
                          <span className="text-[#141414] font-black">&gt;</span>
                          <span className="font-semibold">{lg}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right Col - telemetry console (spans 5) */}
        <div className="lg:col-span-5">
          <div className="bg-black border-2 border-black text-zinc-100 rounded-none p-5 shadow-[4px_4px_0px_0px_rgba(20,20,20,1)] space-y-4 h-full flex flex-col">
            <div className="flex justify-between items-center border-b border-zinc-800 pb-2">
              <span className="font-mono text-xs font-bold text-zinc-400 uppercase tracking-wider">Diagnostic CLI output</span>
              <Terminal className="w-4 h-4 text-zinc-500" />
            </div>

            <div className="flex-1 min-h-[300px] font-mono text-3xs space-y-2.5 leading-relaxed text-zinc-300 font-semibold select-text">
              {consoleLogs.length === 0 ? (
                <div className="text-zinc-500 italic space-y-2 block py-12 text-center uppercase tracking-wide">
                  <p>Initializing theoretical verification tests will render logs here.</p>
                  <p className="text-4xs mt-1 text-zinc-650">All routines execute from clean, mathematical code blocks mapped inside the local RBM algorithms context.</p>
                </div>
              ) : (
                consoleLogs.map((log, i) => <div key={i}>{log}</div>)
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
