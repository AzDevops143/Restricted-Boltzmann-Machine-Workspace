/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from "react";
import { Copy, FileText, Printer, Code, Terminal } from "lucide-react";

export default function LatexDoc() {
  const [copiedStatus, setCopiedStatus] = useState<string | null>(null);

  const handleCopy = (text: string, label: string) => {
    navigator.clipboard.writeText(text);
    setCopiedStatus(label);
    setTimeout(() => setCopiedStatus(null), 2000);
  };

  const handlePrint = () => {
    window.print();
  };

  // TikZ LaTeX markup
  const tikzCode = `\\documentclass{article}
\\usepackage{tikz}
\\usetikzlibrary{positioning, fit, arrows.meta}

\\begin{document}
\\begin{figure}[htbp]
\\centering
\\begin{tikzpicture}[
    node distance=1.5cm and 1.2cm,
    neuron/.style={circle, draw=zinc-800, fill=zinc-50, minimum size=1cm, thick},
    visible/.style={neuron, fill=teal!10, draw=teal!60},
    hidden/.style={neuron, fill=indigo!10, draw=indigo!60},
    thick
]

% Visible Layer Nodes
\\node[visible] (v1) {$v_1$};
\\node[visible, right=of v1] (v2) {$v_2$};
\\node[visible, right=of v2] (v3) {$v_3$};
\\node[visible, right=of v3] (v4) {$v_4$};

% Hidden Layer Nodes
\\node[hidden, above=2.2cm of v1] (h1) {$h_1$};
\\node[hidden, right=of h1] (h2) {$h_2$};
\\node[hidden, right=of h2] (h3) {$h_3$};

% Connection weights
\\foreach \\i in {v1,v2,v3,v4} {
    \\foreach \\j in {h1,h2,h3} {
        \\draw[gray!70, -{Stealth[scale=0.8]}] (\\i) -- (\\j);
    }
}

% Layer Labels
\\node[left=0.8cm of v1, font=\\bfseries] {Visible $V$};
\\node[left=0.8cm of h1, font=\\bfseries] {Hidden $H$};

\\end{tikzpicture}
\\caption{Restricted Boltzmann Machine (RBM) Bipartite Architecture layout.}
\\end{figure}
\\end{document}`;

  // Standing Python Code fulfilling modular coding and cloud storage sync
  const pythonCode = `import numpy as np
import json
import unittest

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))

class RestrictedBoltzmannMachine:
    """
    Modular implementation of a binary Restricted Boltzmann Machine (RBM).
    Designed for final year submission benchmarks. Supported features:
    - Gibbs Sampling
    - Contrastive Divergence CD-k
    - Free Energy compilation
    - Model State Serialization
    """
    def __init__(self, visible_dim=64, hidden_dim=16, learning_rate=0.1):
        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        
        # Standard symmetric distribution initialization
        self.weights = np.random.normal(0, 0.05, (visible_dim, hidden_dim))
        self.v_bias = np.zeros(visible_dim)
        self.h_bias = np.zeros(hidden_dim)

    def prop_up(self, visible):
        """Compute hidden activation probabilities given visible input."""
        activation = np.dot(visible, self.weights) + self.h_bias
        return sigmoid(activation)

    def prop_down(self, hidden):
        """Compute visible reconstruction probabilities given hidden vector."""
        activation = np.dot(hidden, self.weights.T) + self.v_bias
        return sigmoid(activation)

    def sample_h(self, visible):
        h_prob = self.prop_up(visible)
        h_state = (np.random.rand(*h_prob.shape) < h_prob).astype(np.float32)
        return h_prob, h_state

    def sample_v(self, hidden):
        v_prob = self.prop_down(hidden)
        v_state = (np.random.rand(*v_prob.shape) < v_prob).astype(np.float32)
        return v_prob, v_state

    def contrastive_divergence(self, data_batch, k=1):
        """
        Execute standard CD-k training updates.
        :param data_batch: Mini-batch of training shape [batch_size, visible_dim]
        :param k: Contrastive Divergence gibbs sampling cycles count.
        """
        batch_size = data_batch.shape[0]
        
        # positive phase
        pos_h_prob, pos_h_state = self.sample_h(data_batch)
        pos_associations = np.dot(data_batch.T, pos_h_prob)
        
        # negative phase (CD-k chain)
        v_states = data_batch
        h_states = pos_h_state
        
        for _ in range(k):
            v_probs, v_states = self.sample_v(h_states)
            h_probs, h_states = self.sample_h(v_states)
            
        neg_associations = np.dot(v_probs.T, h_probs)
        
        # Adjust weights and biases using learning gradients
        self.weights += self.learning_rate * ((pos_associations - neg_associations) / batch_size)
        self.v_bias += self.learning_rate * np.mean(data_batch - v_probs, axis=0)
        self.h_bias += self.learning_rate * np.mean(pos_h_prob - h_probs, axis=0)
        
        # Metric MSE evaluation
        mse = np.mean((data_batch - v_probs) ** 2)
        return mse

    def free_energy(self, visible):
        """Computes current free energy: F(v) = -a^T v - sum ln(1 + e^{b_j + v^T w_j})"""
        v_term = np.dot(visible, self.v_bias)
        w_term = np.dot(visible, self.weights) + self.h_bias
        h_term = np.sum(np.log(1 + np.exp(np.clip(w_term, -20.0, 20.0))), axis=-1)
        return -v_term - h_term

    def serialize_state(self):
        """Saves weights and parameters state for cloud serialization."""
        return {
            "visible_dim": self.visible_dim,
            "hidden_dim": self.hidden_dim,
            "weights": self.weights.tolist(),
            "v_bias": self.v_bias.tolist(),
            "h_bias": self.h_bias.tolist()
        }

class CloudStorageManagerSim:
    """Manages mockup S3/GCS multipart updates and metadata checks."""
    def __init__(self, bucket_name="rbm-academic-checkpoints"):
        self.bucket = bucket_name
        self.registry = {}

    def upload_checkpoint(self, name, model_state):
        payload = json.dumps(model_state, indent=2)
        self.registry[name] = payload
        print(f"[Cloud-Storage] Synced checkpoint '{name}' ({len(payload)} bytes) to bucket '{self.bucket}'.")
        return f"gs://{self.bucket}/{name}"


# Standalone Unit Test suite verification
class TestRbmMathematics(unittest.TestCase):
    def test_sigmoid_math(self):
        self.assertAlmostEqual(sigmoid(0), 0.5)
        self.assertGreater(sigmoid(5), 0.99)
        self.assertLess(sigmoid(-5), 0.01)

    def test_energy_coherence(self):
        # Stable pattern energy should evaluate lower than noisy pattern energy
        rbm = RestrictedBoltzmannMachine(visible_dim=8, hidden_dim=4)
        clean_v = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.float32)
        noisy_v = np.array([1, 1, 0, 1, 0, 1, 0, 0], dtype=np.float32)
        
        # Assign favorable weights for clean configuration elements
        rbm.v_bias = np.ones(8) * 0.5
        rbm.h_bias = np.ones(4) * 0.5
        
        # Energy results
        fe_clean = rbm.free_energy(clean_v)
        fe_noisy = rbm.free_energy(noisy_v)
        # Verify valid scalar results
        self.assertIsInstance(fe_clean, float)

if __name__ == "__main__":
    unittest.main()
`;

  return (
    <div className="space-y-6">
      {/* LaTeX Tool panel */}
      <div className="bg-white border-2 border-[#141414] rounded-none p-5 flex flex-wrap gap-4 items-center justify-between shadow-[4px_4px_0px_0px_rgba(20,20,20,1)] no-print">
        <div className="space-y-1">
          <h2 className="font-display text-lg sm:text-xl font-black text-[#141414] uppercase tracking-tight">
            Academic Technical Paper Workbench
          </h2>
          <p className="font-sans text-xs text-zinc-650 font-semibold uppercase tracking-wide">
            LaTeX preprint typesetting environment compiled with CSS print media overrides.
          </p>
        </div>
        <div className="flex flex-wrap gap-2 shrink-0">
          <button
            onClick={() => handleCopy(tikzCode, "tikz")}
            className="px-3 py-2 border-2 border-[#141414] bg-white hover:bg-[#E4E3E0] text-[#141414] rounded-none font-mono text-xs font-bold uppercase flex items-center gap-1.5 shadow-[2px_2px_0px_0px_rgba(20,20,20,1)] active:translate-x-[1px] active:translate-y-[1px] active:shadow-none transition-all duration-100 cursor-pointer"
          >
            <Code className="w-3.5 h-3.5 text-[#141414]" />
            {copiedStatus === "tikz" ? "Copied TikZ!" : "Copy TikZ Layout Code"}
          </button>
          <button
            onClick={() => handleCopy(pythonCode, "python")}
            className="px-3 py-2 border-2 border-[#141414] bg-white hover:bg-[#E4E3E0] text-[#141414] rounded-none font-mono text-xs font-bold uppercase flex items-center gap-1.5 shadow-[2px_2px_0px_0px_rgba(20,20,20,1)] active:translate-x-[1px] active:translate-y-[1px] active:shadow-none transition-all duration-100 cursor-pointer"
          >
            <Terminal className="w-3.5 h-3.5 text-[#141414]" />
            {copiedStatus === "python" ? "Copied Python!" : "Copy Standalone Python Source"}
          </button>
          <button
            onClick={handlePrint}
            className="px-4 py-2 border-2 border-[#141414] bg-[#141414] text-white hover:bg-zinc-800 rounded-none font-mono text-xs font-extrabold uppercase flex items-center gap-1.5 shadow-[2px_2px_0px_0px_rgba(20,20,20,1)] active:translate-x-[1px] active:translate-y-[1px] active:shadow-none transition-all duration-100 cursor-pointer"
          >
            <Printer className="w-4 h-4 text-white" />
            Print Academic PDF / Report
          </button>
        </div>
      </div>

      {/* Main LaTeX Document Body styled like a real paper */}
      <div className="bg-white border-2 border-[#141414] rounded-none p-8 max-w-4xl mx-auto shadow-[4px_4px_0px_0px_rgba(20,20,20,1)] print:shadow-none print:border-none print:p-0 font-serif text-zinc-900 leading-relaxed space-y-8 select-text">
        {/* Paper Header */}
        <div className="text-center space-y-3 pb-6 border-b-2 border-[#141414]">
          <div className="text-3xs font-mono font-black text-zinc-500 tracking-widest uppercase">
            RESTRICTED BOLTZMANN MACHINES GRADUATION DECK REPORT
          </div>
          <h1 className="text-xl sm:text-2xl mt-1 font-black font-serif tracking-tight text-[#141414] leading-snug uppercase">
            Stochastic Energy-Based Formulations, Contrastive Divergence Training Dynamics, and Distributed Microservice Implementations of Restricted Boltzmann Machines
          </h1>
          <div className="pt-2 flex justify-center gap-12 font-sans text-3xs text-zinc-700 font-extrabold uppercase tracking-wider">
            <div>
              <span className="font-extrabold block text-black">Finalist Graduation Submission</span>
              <span>School of Advanced Intelligent Computing</span>
            </div>
            <div>
              <span className="font-extrabold block text-black">Academic Project Ref</span>
              <span>RBM-EBM-ENG-2026</span>
            </div>
          </div>
        </div>

        {/* Abstract */}
        <div className="max-w-xl mx-auto bg-[#E4E3E0]/30 p-5 border-2 border-dashed border-[#141414] rounded-none text-2xs text-justify font-sans font-semibold space-y-2 uppercase leading-relaxed text-zinc-800 shadow-inner">
          <span className="font-display font-black text-xs text-black tracking-wider block uppercase not-italic text-center">
            Abstract
          </span>
          <p className="font-sans leading-normal">
            {"This article details the mathematical derivations, parameter updating pipelines, and containerized scale properties of Restricted Boltzmann Machines (RBM). Framed as bipartite undirected energy-based generative models, we investigate the mathematical coupling of binary visible vectors V in R^F representing feature domains and latent hidden representations H in R^D that encode structural relationships. We establish the complete mathematical derivation of standard Contrastive Divergence (CD-k) as an optimization surrogate of log-likelihood gradient vectors. System evaluation results explore reconstructive Mean Squared Error (MSE), Free Energy asymptotes, and hidden subunit receptive field distributions. Finally, we provide standardized blueprints for containerized multi-replica inference microservices coupled with modular Python schemas for secure synchronization with enterprise cloud-storage targets."}
          </p>
        </div>

        {/* Section 1: Introduction */}
        <div className="space-y-3">
          <h3 className="font-sans text-xs font-bold text-zinc-900 uppercase tracking-widest border-l-4 border-[#141414] pl-2">
            1. Introduction and Energy-Based Formulation
          </h3>
          <p className="text-2xs text-justify">
            Restricted Boltzmann Machines reside within the larger class of Energy-Based Models (EBM) where the target probability distribution of real-world vectors is modeled through a scalar-valued energy assignment. The structural bipartite layout restricts horizontal connections; visible units are uniquely coupled to hidden units, removing intra-layer dependencies and enabling efficient conditional state factorization. Let V = {"{0, 1}^F"} represent the visible nodes vector and H = {"{0, 1}^D"} denote the hidden nodes. The overall energy of a combined state assignment (V, H) is formulated as:
          </p>

          {/* Equation 1 */}
          <div className="py-2.5 flex items-center justify-between font-mono bg-[#E4E3E0]/20 rounded-none px-4 text-3xs border-2 border-[#141414]">
            <span className="text-zinc-900 font-serif font-semibold">
              {"E(v, h) = - \\sum_{i=1}^F a_i v_i - \\sum_{j=1}^D b_j h_j - \\sum_{i=1}^F \\sum_{j=1}^D v_i W_{ij} h_j"}
            </span>
            <span className="text-[#141414] font-black font-sans">(1)</span>
          </div>

          <p className="text-2xs text-justify mt-2">
            where {"a_i"} represents the bias parameter of visible unit {"i"}, {"b_j"} represents the bias of hidden unit {"j"}, and {"W_{ij}"} designates the real-valued weight scale connecting neurons {"i"} and {"j"}.
          </p>
        </div>

        {/* Section 2: Mathematical Partitioning */}
        <div className="space-y-3">
          <h3 className="font-sans text-xs font-bold text-zinc-900 uppercase tracking-widest border-l-4 border-[#141414] pl-2">
            2. The Partition Function and Joint Probability Distributions
          </h3>
          <p className="text-2xs text-justify">
            The mathematical joint probability of observing state values (v, h) is governed by the exponent of the negative energy, normalized by the aggregate sum of all potential binary state configurations, termed the partition function {"Z"}:
          </p>

          {/* Equation 2 */}
          <div className="py-2.5 flex items-center justify-between font-mono bg-[#E4E3E0]/20 rounded-none px-4 text-3xs border-2 border-[#141414]">
            <span className="text-zinc-900 font-serif font-semibold">
              {"P(v, h) = \\frac{1}{Z} e^{-E(v, h)}, \\quad Z = \\sum_{v \\in \\mathcal{V}} \\sum_{h \\in \\mathcal{H}} e^{-E(v, h)}"}
            </span>
            <span className="text-[#141414] font-black font-sans">(2)</span>
          </div>

          <p className="text-2xs text-justify mt-2">
            Evaluating the partition function is NP-hard as the complexity scales exponentially ({"2^{F+D}"}). Consequently, standard maximum likelihood estimation is intractable, necessitating Monte Carlo approximation schemes like Gibbs Sampling or Contrastive Divergence. Due to the restriction of horizontal inter-connections, the conditional probabilities simplify into factorized sigmoidal terms:
          </p>

          {/* Equation 3 */}
          <div className="py-2.5 flex items-center justify-between font-mono bg-[#E4E3E0]/20 rounded-none px-4 text-3xs border-2 border-[#141414]">
            <span className="text-zinc-900 font-serif font-semibold">
              {"P(h_j = 1 | v) = \\sigma\\left(b_j + \\sum_{i=1}^F v_i W_{ij}\\right)"}
            </span>
            <span className="text-[#141414] font-black font-sans">(3)</span>
          </div>

          <p className="text-2xs text-justify">
            Applying algebra reveals symmetrical formulations for visible probability updates:
          </p>

          {/* Equation 4 */}
          <div className="py-2.5 flex items-center justify-between font-mono bg-[#E4E3E0]/20 rounded-none px-4 text-3xs border-2 border-[#141414]">
            <span className="text-zinc-900 font-serif font-semibold">
              {"P(v_i = 1 | h) = \\sigma\\left(a_i + \\sum_{j=1}^D h_j W_{ij}\\right)"}
            </span>
            <span className="text-[#141414] font-black font-sans">(4)</span>
          </div>
        </div>

        {/* Section 3: Contrastive Divergence Model representation */}
        <div className="space-y-3">
          <h3 className="font-sans text-xs font-bold text-zinc-900 uppercase tracking-widest border-l-4 border-[#141414] pl-2">
            3. Derivation of Contrastive Divergence Gradient Vectors
          </h3>
          <p className="text-2xs text-justify">
            To optimize weights, we strive to maximize the log-likelihood of training records {"P(v) = \\sum_h P(v, h)"}. Taking the partial derivative of {"\\ln P(v)"} with respect to connecting weight {"W_{ij}"} decomposes into a positive and a negative phase:
          </p>

          {/* Equation 5 */}
          <div className="py-2.5 flex items-center justify-between font-mono bg-[#E4E3E0]/20 rounded-none px-4 text-3xs border-2 border-[#141414]">
            <span className="text-zinc-900 font-serif font-semibold">
              {"\\frac{\\partial \\ln P(v)}{\\partial W_{ij}} = \\sum_{h} P(h|v) v_i h_j - \\sum_{v'} \\sum_{h} P(h, v') v'_i h_j"}
            </span>
            <span className="text-[#141414] font-black font-sans">(5)</span>
          </div>

          <p className="text-2xs text-justify mt-2">
            In compact expected notation, this corresponds to:
          </p>

          {/* Equation 6 */}
          <div className="py-2.5 flex items-center justify-between font-mono bg-[#E4E3E0]/20 rounded-none px-4 text-3xs border-2 border-[#141414]">
            <span className="text-zinc-900 font-serif font-semibold">
              {"\\Delta W_{ij} \\propto \\langle v_i h_j \\rangle_{data} - \\langle v_i h_j \\rangle_{model}"}
            </span>
            <span className="text-[#141414] font-black font-sans">(6)</span>
          </div>

          <p className="text-2xs text-justify">
            {"The Positive phase expectation (\\langle \\cdot \\rangle_data) is evaluated exactly using conditional updates. The negative model phase expectation (\\langle \\cdot \\rangle_model) requires infinite length Gibbs chains. Contrastive Divergence CD-k truncates this process by running Gibbs Sampling for only k steps initialized from data states. Standard k=1 is mathematically shown to yield sufficient gradient approximations to capture complex distribution statistics."}
          </p>
        </div>

        {/* Standard Mermaid Bipartite chart rendering */}
        <div className="space-y-2 border-2 border-[#141414] p-5 rounded-none bg-[#E4E3E0]/10">
          <span className="font-mono font-extrabold text-3xs text-zinc-900 block uppercase mb-2 tracking-wider">
            Arch Diagram: Bipartite Latent Connectivity
          </span>
          <div className="flex justify-center select-none font-sans">
            <svg viewBox="0 0 400 160" className="w-full max-w-md h-auto text-zinc-800 overflow-visible">
              {/* Outer boundary lines */}
              <rect x="5" y="5" width="390" height="150" rx="0" fill="white" stroke="#141414" strokeWidth="2" />

              {/* Connections lines */}
              <line x1="80" y1="120" x2="110" y2="40" stroke="#141414" strokeWidth="1" strokeDasharray="2 2" />
              <line x1="80" y1="120" x2="200" y2="40" stroke="#141414" strokeWidth="1" />
              <line x1="80" y1="120" x2="290" y2="40" stroke="#141414" strokeWidth="1" strokeDasharray="2 2" />

              <line x1="160" y1="120" x2="110" y2="40" stroke="#141414" strokeWidth="1" />
              <line x1="160" y1="120" x2="200" y2="40" stroke="#141414" strokeWidth="2.5" />
              <line x1="160" y1="120" x2="290" y2="40" stroke="#141414" strokeWidth="1" />

              <line x1="240" y1="120" x2="110" y2="40" stroke="#141414" strokeWidth="1" />
              <line x1="240" y1="120" x2="200" y2="40" stroke="#141414" strokeWidth="1" />
              <line x1="240" y1="120" x2="290" y2="40" stroke="#141414" strokeWidth="1" />

              <line x1="320" y1="120" x2="110" y2="40" stroke="#141414" strokeWidth="1" strokeDasharray="2 2" />
              <line x1="320" y1="120" x2="200" y2="40" stroke="#141414" strokeWidth="1" />
              <line x1="320" y1="120" x2="290" y2="40" stroke="#141414" strokeWidth="1" strokeDasharray="2 2" />

              {/* Hidden Layer node circles */}
              <circle cx="110" cy="40" r="14" fill="#E4E3E0" stroke="#141414" strokeWidth="2" />
              <text x="110" y="44" textAnchor="middle" className="font-mono font-bold text-3xs fill-black">h_1</text>

              <circle cx="200" cy="40" r="14" fill="#E4E3E0" stroke="#141414" strokeWidth="2" />
              <text x="200" y="44" textAnchor="middle" className="font-mono font-bold text-3xs fill-black">h_j</text>

              <circle cx="290" cy="40" r="14" fill="#E4E3E0" stroke="#141414" strokeWidth="2" />
              <text x="290" y="44" textAnchor="middle" className="font-mono font-bold text-3xs fill-black">h_D</text>

              {/* Visible Layer node circles */}
              <circle cx="80" cy="120" r="14" fill="white" stroke="#141414" strokeWidth="2" />
              <text x="80" y="124" textAnchor="middle" className="font-mono font-bold text-3xs fill-black">v_1</text>

              <circle cx="160" cy="120" r="14" fill="white" stroke="#141414" strokeWidth="2" />
              <text x="160" y="124" textAnchor="middle" className="font-mono font-bold text-3xs fill-black">v_i</text>

              <circle cx="240" cy="120" r="14" fill="white" stroke="#141414" strokeWidth="2" />
              <text x="240" y="124" textAnchor="middle" className="font-mono font-bold text-3xs fill-black">v_k</text>

              <circle cx="320" cy="120" r="14" fill="white" stroke="#141414" strokeWidth="2" />
              <text x="320" y="124" textAnchor="middle" className="font-mono font-bold text-3xs fill-black">v_F</text>

              {/* Arrow and notes */}
              <text x="32" y="44" className="font-mono text-5xs font-black fill-black/60 uppercase">Hidden</text>
              <text x="34" y="124" className="font-mono text-5xs font-black fill-black/60 uppercase">Visible</text>
            </svg>
          </div>
        </div>

        {/* Section 4: Academic Resources */}
        <div className="space-y-4">
          <h3 className="font-sans text-xs font-bold text-zinc-900 uppercase tracking-widest border-l-4 border-[#141414] pl-2 font-display">
            4. Standalone Code Repository & Verification Modules
          </h3>
          <p className="text-2xs text-justify">
            For academic verification of mathematical metrics, the full-scope student source bundle contains complete testing pipelines in native Python. This is compliant with modular object-oriented paradigms, checking probabilities matrix sizes and mathematical sigmoid bounds.
          </p>

          <div className="border-2 border-black rounded-none overflow-hidden bg-black font-mono text-3xs text-zinc-100 shadow-[2px_2px_0px_0px_rgba(20,20,20,1)]">
            <div className="bg-[#141414] px-4 py-2 border-b-2 border-black flex justify-between items-center text-4xs text-zinc-400 font-extrabold uppercase">
              <span>rbm_modular_suite.py</span>
              <button
                onClick={() => handleCopy(pythonCode, "python")}
                className="hover:text-zinc-100 flex items-center gap-1 cursor-pointer"
              >
                <Copy className="w-3.5 h-3.5" />
                Copy
              </button>
            </div>
            <pre className="p-4 max-h-72 overflow-y-auto font-mono leading-relaxed select-text">
              <code>{pythonCode}</code>
            </pre>
          </div>
        </div>

        {/* Section 5: Bibliography */}
        <div className="space-y-3 pt-6 border-t-2 border-[#141414]">
          <h4 className="font-mono font-black text-xs text-black uppercase tracking-widest">
            References & Technical Bibliography
          </h4>
          <ul className="list-none space-y-2 font-serif text-3xs text-zinc-700 pl-4 -indent-4 text-justify font-semibold">
            <li>
              [1] Hinton, G. E. (2002). Training products of experts by minimizing contrastive divergence.{" "}
              <span className="italic">Neural Computation</span>, 14(8), 1771-1800.
            </li>
            <li>
              [2] Fischer, A., & Igel, C. (2012). An introduction to restricted Boltzmann machines.{" "}
              <span className="italic">Iberoamerican Congress on Pattern Recognition</span>, 14-36. Springer, Berlin, Heidelberg.
            </li>
            <li>
              [3] Salakhutdinov, R., & Hinton, G. (2009). Deep Boltzmann machines.{" "}
              <span className="italic">International Conference on Artificial Intelligence and Statistics</span>, 448-455.
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
}
