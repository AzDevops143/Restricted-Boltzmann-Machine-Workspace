/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * RBM Weight & Bias State representation
 */
export interface RbmWeights {
  // 2D Array: weights[v][h] where v is visible index, h is hidden index
  weights: number[][];
  // 1D Array: length corresponds to visible nodes count
  vBiases: number[];
  // 1D Array: length corresponds to hidden nodes count
  hBiases: number[];
}

/**
 * Model configuration parameters
 */
export interface RbmConfig {
  visibleCount: number;
  hiddenCount: number;
  learningRate: number;
  weightDecay: number;
  momentum: number;
  cdSteps: number; // k in CD-k
  batchSize: number;
  epochs: number;
}

/**
 * Training logs recorded per epoch during CD-k algorithm execution
 */
export interface TrainingLog {
  epoch: number;
  reconstructionLoss: number;
  averageFreeEnergy: number;
  weightSparsity: number; // calculated fraction of near-zero weights
  hiddenActivationSparsity: number; // fraction of active hidden neurons
}

/**
 * Dataset pattern definition for visualization and training
 */
export interface PatternData {
  name: string;
  data: number[]; // binary flat vector of visibleCount dimension
}

/**
 * Comparative experiment settings for analyzing hyperparameter influence
 */
export interface ComparativeResult {
  clientId: string;
  label: string;
  hiddenCount: number;
  learningRate: number;
  cdSteps: number;
  reconstructionLoss: number;
  energyVariance: number;
  convergenceEpoch: number;
  status: "Completed" | "Pending" | "Running";
}

/**
 * Slide node describing slides for the presentation deck
 */
export interface PresentationSlide {
  id: number;
  title: string;
  category: "Foundations" | "Training" | "Architecture" | "Evaluation" | "Containerization";
  bullets: string[];
  latexBlock?: string;
}

/**
 * Unit test specification for the RBM mathematical runner
 */
export interface RbmTestCase {
  id: string;
  name: string;
  description: string;
  status: "idle" | "passed" | "failed" | "running";
  logs: string[];
  executionTimeMs?: number;
}

/**
 * Microservice structure mimicking Kubernetes replica state
 */
export interface MicroservicePod {
  name: string;
  role: "inference" | "trainer" | "aggregator" | "storage-sync";
  status: "Running" | "Pending" | "Failed";
  cpuUsage: number; // Percentage
  memoryUsage: number; // MB
  restarts: number;
}

/**
 * Cloud storage state simulator
 */
export interface CloudSyncState {
  lastSyncTime: string | null;
  syncedEpochs: number[];
  uploadedBytes: number;
  storageProvider: "Google Cloud Storage" | "Amazon S3" | "Azure Blob Storage";
  bucketName: string;
}
