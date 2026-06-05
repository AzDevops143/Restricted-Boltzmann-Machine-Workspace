/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { RbmWeights, RbmConfig, PatternData } from "./types";

/**
 * Standard Sigmoid activation function
 */
export function sigmoid(x: number): number {
  return 1.0 / (1.0 + Math.exp(-Math.max(-20, Math.min(20, x))));
}

/**
 * Initialize weights and biases based on standard normal guidelines
 * or uniform small values for controlled training convergence.
 */
export function initializeRbmWeights(visibleCount: number, hiddenCount: number): RbmWeights {
  const weights: number[][] = [];
  for (let v = 0; v < visibleCount; v++) {
    const row: number[] = [];
    for (let h = 0; h < hiddenCount; h++) {
      // Small random weights symmetric around zero
      row.push((Math.random() - 0.5) * 0.1);
    }
    weights.push(row);
  }

  const vBiases = new Array(visibleCount).fill(0).map(() => (Math.random() - 0.5) * 0.05);
  const hBiases = new Array(hiddenCount).fill(0).map(() => (Math.random() - 0.5) * 0.05);

  return { weights, vBiases, hBiases };
}

/**
 * Compute the activation probability vector for hidden units given visible units
 */
export function computeHiddenProbabilities(
  visibleStates: number[],
  weights: RbmWeights
): number[] {
  const hiddenCount = weights.hBiases.length;
  const hProbs = new Array(hiddenCount).fill(0);

  for (let h = 0; h < hiddenCount; h++) {
    let activation = weights.hBiases[h];
    for (let v = 0; v < visibleStates.length; v++) {
      activation += visibleStates[v] * weights.weights[v][h];
    }
    hProbs[h] = sigmoid(activation);
  }
  return hProbs;
}

/**
 * Compute the activation probability vector for visible units given hidden units
 */
export function computeVisibleProbabilities(
  hiddenStates: number[],
  weights: RbmWeights
): number[] {
  const visibleCount = weights.vBiases.length;
  const vProbs = new Array(visibleCount).fill(0);

  for (let v = 0; v < visibleCount; v++) {
    let activation = weights.vBiases[v];
    for (let h = 0; h < hiddenStates.length; h++) {
      activation += hiddenStates[h] * weights.weights[v][h];
    }
    vProbs[v] = sigmoid(activation);
  }
  return vProbs;
}

/**
 * Sample states stochastically from a probability vector
 */
export function sampleStates(probabilities: number[]): number[] {
  return probabilities.map((p) => (Math.random() < p ? 1.0 : 0.0));
}

/**
 * Execute a single Gibbs step (visible -> hidden -> reconstructed visible)
 */
export function runGibbsStep(
  visibleStates: number[],
  weights: RbmWeights
): {
  hiddenProbs: number[];
  hiddenStates: number[];
  reconstructedProbs: number[];
  reconstructedStates: number[];
} {
  const hiddenProbs = computeHiddenProbabilities(visibleStates, weights);
  const hiddenStates = sampleStates(hiddenProbs);
  const reconstructedProbs = computeVisibleProbabilities(hiddenStates, weights);
  const reconstructedStates = sampleStates(reconstructedProbs);

  return {
    hiddenProbs,
    hiddenStates,
    reconstructedProbs,
    reconstructedStates,
  };
}

/**
 * Calculate the Free Energy F(V) of a visible vector context.
 * Useful for monitoring convergence and checking model equilibrium.
 * Formula: F(v) = - \sum_i v_i a_i - \sum_j \ln(1 + \exp(b_j + \sum_i v_i W_{ij}))
 */
export function calculateFreeEnergy(visibleStates: number[], weights: RbmWeights): number {
  let biasTerm = 0;
  for (let v = 0; v < visibleStates.length; v++) {
    biasTerm += visibleStates[v] * weights.vBiases[v];
  }

  let hTerm = 0;
  for (let h = 0; h < weights.hBiases.length; h++) {
    let innerActivation = weights.hBiases[h];
    for (let v = 0; v < visibleStates.length; v++) {
      innerActivation += visibleStates[v] * weights.weights[v][h];
    }

    // Protection against ln(1+e^x) numerical overflow
    if (innerActivation > 16) {
      hTerm += innerActivation;
    } else if (innerActivation < -16) {
      // contribution of ln(1+e^x) is almost zero
    } else {
      hTerm += Math.log(1.0 + Math.exp(innerActivation));
    }
  }

  return -biasTerm - hTerm;
}

/**
 * Execute a single contrastive divergence training cycle (CD-k) with momentum
 * and weight updates over a batch. Returns the updated weights and batch-aggregated MSE.
 */
export function trainBatchCdK(
  batch: number[][],
  weights: RbmWeights,
  config: RbmConfig,
  currentMomentum: number,
  prevWeightGradient: number[][],
  prevVBiasGradient: number[],
  prevHBiasGradient: number[]
): {
  updatedWeights: RbmWeights;
  mse: number;
  weightGradients: number[][];
  vBiasGradients: number[];
  hBiasGradients: number[];
} {
  const visibleCount = config.visibleCount;
  const hiddenCount = config.hiddenCount;
  const lr = config.learningRate;
  const decay = config.weightDecay;

  // Initialize gradient accumulators
  const avgWDelta = Array.from({ length: visibleCount }, () => new Array(hiddenCount).fill(0));
  const avgVDelta = new Array(visibleCount).fill(0);
  const avgHDelta = new Array(hiddenCount).fill(0);
  let totalMse = 0;

  for (let b = 0; b < batch.length; b++) {
    const v0 = batch[b];

    // CD-k: Positive Phase (v0 -> h0)
    const h0Probs = computeHiddenProbabilities(v0, weights);
    let hkStates = sampleStates(h0Probs);
    let vkProbs = new Array(visibleCount).fill(0);

    // Negative Phase Gibbs sequence
    for (let step = 0; step < config.cdSteps; step++) {
      vkProbs = computeVisibleProbabilities(hkStates, weights);
      const vkStates = sampleStates(vkProbs);
      const hkProbs = computeHiddenProbabilities(vkStates, weights);
      hkStates = sampleStates(hkProbs);
    }
    const hkProbs = computeHiddenProbabilities(vkProbs, weights); // final prediction probability

    // Accumulate gradients
    for (let v = 0; v < visibleCount; v++) {
      for (let h = 0; h < hiddenCount; h++) {
        // gradient weight formula: <v0 h0> - <vk hk>
        avgWDelta[v][h] += v0[v] * h0Probs[h] - vkProbs[v] * hkProbs[h];
      }
      avgVDelta[v] += v0[v] - vkProbs[v];
    }

    for (let h = 0; h < hiddenCount; h++) {
      avgHDelta[h] += h0Probs[h] - hkProbs[h];
    }

    // Accumulate MSE loss between design v0 and reconstruction vkProbs
    let sampleMse = 0;
    for (let v = 0; v < visibleCount; v++) {
      const diff = v0[v] - vkProbs[v];
      sampleMse += diff * diff;
    }
    totalMse += sampleMse / visibleCount;
  }

  // Finalize weight upgrades applying batch scale, momentum, learning rate, and L2 regularization weight decay
  const updatedWeights = Array.from({ length: visibleCount }, () => new Array(hiddenCount).fill(0));
  const updatedVBiases = new Array(visibleCount).fill(0);
  const updatedHBiases = new Array(hiddenCount).fill(0);

  const finalWGrads = Array.from({ length: visibleCount }, () => new Array(hiddenCount).fill(0));
  const finalVGrads = new Array(visibleCount).fill(0);
  const finalHGrads = new Array(hiddenCount).fill(0);

  const batchScale = 1.0 / batch.length;

  for (let v = 0; v < visibleCount; v++) {
    for (let h = 0; h < hiddenCount; h++) {
      const grad = avgWDelta[v][h] * batchScale - decay * weights.weights[v][h];
      const delta = currentMomentum * prevWeightGradient[v][h] + lr * grad;
      updatedWeights[v][h] = weights.weights[v][h] + delta;
      finalWGrads[v][h] = delta;
    }
    const vBiasGrad = avgVDelta[v] * batchScale;
    const vDelta = currentMomentum * prevVBiasGradient[v] + lr * vBiasGrad;
    updatedVBiases[v] = weights.vBiases[v] + vDelta;
    finalVGrads[v] = vDelta;
  }

  for (let h = 0; h < hiddenCount; h++) {
    const hBiasGrad = avgHDelta[h] * batchScale;
    const hDelta = currentMomentum * prevHBiasGradient[h] + lr * hBiasGrad;
    updatedHBiases[h] = weights.hBiases[h] + hDelta;
    finalHGrads[h] = hDelta;
  }

  return {
    updatedWeights: {
      weights: updatedWeights,
      vBiases: updatedVBiases,
      hBiases: updatedHBiases,
    },
    mse: totalMse / batch.length,
    weightGradients: finalWGrads,
    vBiasGradients: finalVGrads,
    hBiasGradients: finalHGrads,
  };
}

/**
 * Structured academic pattern presets
 * Grid dimensions are 8x8 (64 units total).
 */
export const RBM_PATTERNS: PatternData[] = [
  {
    name: "Horizontal Stripes",
    data: [
      1, 1, 1, 1, 1, 1, 1, 1,
      0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 1, 1, 1, 1, 1, 1,
      0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 1, 1, 1, 1, 1, 1,
      0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 1, 1, 1, 1, 1, 1,
      0, 0, 0, 0, 0, 0, 0, 0,
    ],
  },
  {
    name: "Vertical Stripes",
    data: [
      1, 0, 1, 0, 1, 0, 1, 0,
      1, 0, 1, 0, 1, 0, 1, 0,
      1, 0, 1, 0, 1, 0, 1, 0,
      1, 0, 1, 0, 1, 0, 1, 0,
      1, 0, 1, 0, 1, 0, 1, 0,
      1, 0, 1, 0, 1, 0, 1, 0,
      1, 0, 1, 0, 1, 0, 1, 0,
      1, 0, 1, 0, 1, 0, 1, 0,
    ],
  },
  {
    name: "Outer Border Frame",
    data: [
      1, 1, 1, 1, 1, 1, 1, 1,
      1, 0, 0, 0, 0, 0, 0, 1,
      1, 0, 0, 0, 0, 0, 0, 1,
      1, 0, 0, 0, 0, 0, 0, 1,
      1, 0, 0, 0, 0, 0, 0, 1,
      1, 0, 0, 0, 0, 0, 0, 1,
      1, 0, 0, 0, 0, 0, 0, 1,
      1, 1, 1, 1, 1, 1, 1, 1,
    ],
  },
  {
    name: "Glyph Block 'T'",
    data: [
      1, 1, 1, 1, 1, 1, 1, 1,
      0, 0, 0, 1, 1, 0, 0, 0,
      0, 0, 0, 1, 1, 0, 0, 0,
      0, 0, 0, 1, 1, 0, 0, 0,
      0, 0, 0, 1, 1, 0, 0, 0,
      0, 0, 0, 1, 1, 0, 0, 0,
      0, 0, 0, 1, 1, 0, 0, 0,
      0, 0, 0, 1, 1, 0, 0, 0,
    ],
  },
  {
    name: "Handwritten Digit '0'",
    data: [
      0, 0, 1, 1, 1, 1, 0, 0,
      0, 1, 1, 0, 0, 1, 1, 0,
      1, 1, 0, 0, 0, 0, 1, 1,
      1, 1, 0, 0, 0, 0, 1, 1,
      1, 1, 0, 0, 0, 0, 1, 1,
      1, 1, 0, 0, 0, 0, 1, 1,
      0, 1, 1, 0, 0, 1, 1, 0,
      0, 0, 1, 1, 1, 1, 0, 0,
    ],
  },
  {
    name: "Handwritten Digit '1'",
    data: [
      0, 0, 0, 1, 1, 0, 0, 0,
      0, 0, 1, 1, 1, 0, 0, 0,
      0, 0, 0, 1, 1, 0, 0, 0,
      0, 0, 0, 1, 1, 0, 0, 0,
      0, 0, 0, 1, 1, 0, 0, 0,
      0, 0, 0, 1, 1, 0, 0, 0,
      0, 0, 0, 1, 1, 0, 0, 0,
      0, 0, 1, 1, 1, 1, 0, 0,
    ],
  },
  {
    name: "Handwritten Digit '4'",
    data: [
      0, 0, 1, 1, 0, 0, 0, 0,
      0, 1, 1, 1, 0, 0, 0, 0,
      1, 1, 0, 1, 1, 0, 0, 0,
      1, 1, 0, 1, 1, 0, 0, 0,
      1, 1, 1, 1, 1, 1, 0, 0,
      0, 0, 0, 1, 1, 0, 0, 0,
      0, 0, 0, 1, 1, 0, 0, 0,
      0, 0, 1, 1, 1, 1, 0, 0,
    ],
  },
  {
    name: "Diagonal X Shape",
    data: [
      1, 1, 0, 0, 0, 0, 1, 1,
      0, 1, 1, 0, 0, 1, 1, 0,
      0, 0, 1, 1, 1, 1, 0, 0,
      0, 0, 0, 1, 1, 0, 0, 0,
      0, 0, 0, 1, 1, 0, 0, 0,
      0, 0, 1, 1, 1, 1, 0, 0,
      0, 1, 1, 0, 0, 1, 1, 0,
      1, 1, 0, 0, 0, 0, 1, 1,
    ],
  },
];

/**
 * Appends noise manually of a specific level to binary datasets.
 * Useful for pattern reconstruction testing.
 */
export function addNoise(data: number[], level: number): number[] {
  return data.map((pixel) => {
    if (Math.random() < level) {
      return pixel === 1 ? 0 : 1;
    }
    return pixel;
  });
}
