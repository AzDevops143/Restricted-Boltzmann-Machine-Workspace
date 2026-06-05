/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from "react";
import { PresentationSlide } from "../types";
import { ChevronLeft, ChevronRight, Play, Pause, Grid, BookOpen } from "lucide-react";

export default function PptxSlides() {
  const slides: PresentationSlide[] = [
    {
      id: 1,
      title: "Restricted Boltzmann Machines: Mathematical Formulation",
      category: "Foundations",
      bullets: [
        "Unifying bipartite stochastic neural network framework representing joint feature spaces.",
        "Undirected connections strictly link visible inputs V to discrete latent hidden subunits H.",
        "Energy parameterization is designed to measure modeling confidence: lower assignments map to high probability states.",
        "Joint probability P(v, h) is governed by normalizing partition metrics (Z) scaled via e^{-E(v, h)}."
      ],
      latexBlock: "E(v, h) = - \\sum_i a_i v_i - \\sum_j b_j h_j - \\sum_{i,j} v_i W_{ij} h_j",
    },
    {
      id: 2,
      title: "Stochastic Transitions & Gibbs Sampling",
      category: "Foundations",
      bullets: [
        "Symmetrical bipartite structure deletes horizontal connections, establishing independence within individual layers.",
        "Visible nodes are conditionally independent given the values of hidden node spaces, factorizing conditional probability calculations.",
        "Stochastic processes estimate expectations through sequential node sampling, utilizing sigmoid threshold activations step-by-step.",
        "Asymptotic chains converge to true equilibrium states, mapping sample reconstructions from hidden states back to visible states."
      ],
      latexBlock: "P(h_j = 1 | v) = \\sigma\\left(b_j + \\sum_i v_i W_{ij}\\right)",
    },
    {
      id: 3,
      title: "Gradient Training via Contrastive Divergence (CD-k)",
      category: "Training",
      bullets: [
        "Maximum Likelihood optimization is intractable due to exponential partition sums scaling as O(2^{V+H}).",
        "Contrastive Divergence (CD-k) serves as an proxy, approximating gradients through short Gibbs chains.",
        "Gradient updates calculate differences between positive phases (data-driven associations) and negative phases (reconstructed states).",
        "Standard CD-1 operates efficiently, providing robust parameter updates suitable for pattern extraction and auto-associative completion."
      ],
      latexBlock: "\\Delta W_{ij} \\approx \\eta \\left( \\langle v_i h_j \\rangle_{data} - \\langle v_i h_j \\rangle_{reconstruction} \\right)",
    },
    {
      id: 4,
      title: "Automated Calibration & Model Validation Pipeline",
      category: "Evaluation",
      bullets: [
        "Systematic automated pipelines compare and validate performance indices under strict configurations.",
        "Evaluating Reconstruction loss curves determines training hyperparameter efficiency.",
        "Denoising metrics check pattern reconstruction and repairing under variable added noise (0% to 80%).",
        "Free Energy stability evaluates over epoch states to detect overfitting or partition divergence."
      ],
      latexBlock: "MSE = \\frac{1}{F} \\sum_{i=1}^F (v_i - v'_i)^2",
    },
    {
      id: 5,
      title: "Architectural Scale: Docker Containerization",
      category: "Containerization",
      bullets: [
        "The complete RBM model pipeline follows a clean microservices pattern for decoupled horizontal scaling.",
        "Optimized Docker containerization scripts divide standard builds and training engines cleanly.",
        "Kubernetes orchestration schemas define highly available replica states to balance incoming inference requests safely.",
        "Continuous caching layers speed up inference responses by serving repeated feature requests from hot memory."
      ],
      latexBlock: "gcr.io/academic-rbm-service/rbm-inference:v1.2.0",
    },
    {
      id: 6,
      title: "Distributed Checkpoints & Cloud Synchronization",
      category: "Containerization",
      bullets: [
        "Object-oriented Python modules process continuous cloud syncing protocols securely.",
        "Thread-safe clients automatically upload structured model snapshots to Amazon S3 or Google Cloud Storage buckets.",
        "JSON-serialized metadata registries track loss benchmarks, epoch numbers, and learning rates across deployments.",
        "Automated backup procedures enable rapid service disaster recovery from cloud checkpoints."
      ],
      latexBlock: "gs://rbm-academic-checkpoints/epoch-100-weights.json",
    },
  ];

  const [activeIndex, setActiveIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [showHandouts, setShowHandouts] = useState(false);
  const [playIntervalId, setPlayIntervalId] = useState<NodeJS.Timeout | null>(null);

  // Auto-play controls
  const togglePlay = () => {
    if (isPlaying) {
      if (playIntervalId) clearInterval(playIntervalId);
      setIsPlaying(false);
    } else {
      setIsPlaying(true);
      const id = setInterval(() => {
        setActiveIndex((prev) => (prev + 1) % slides.length);
      }, 3500);
      setPlayIntervalId(id);
    }
  };

  const handleNext = () => {
    setActiveIndex((prev) => (prev + 1) % slides.length);
  };

  const handlePrev = () => {
    setActiveIndex((prev) => (prev - 1 + slides.length) % slides.length);
  };

  const activeSlide = slides[activeIndex];

  return (
    <div className="space-y-6 select-text">
      {/* Control Actions bar */}
      <div className="bg-white border-2 border-[#141414] rounded-none p-5 flex flex-col sm:flex-row sm:items-center justify-between gap-4 shadow-[4px_4px_0px_0px_rgba(20,20,20,1)] no-print">
        <div className="space-y-1">
          <h2 className="font-display text-lg sm:text-xl font-black text-[#141414] uppercase tracking-tight">
            Academic Slideshow Defense Deck
          </h2>
          <p className="font-sans text-xs text-zinc-650 font-semibold uppercase tracking-wide">
            Slide presentations illustrating theoretical formulations, training CD steps, and scaling.
          </p>
        </div>
        <div className="flex flex-wrap gap-2 shrink-0">
          <button
            onClick={() => setShowHandouts(!showHandouts)}
            className="px-3.5 py-2 border-2 border-[#141414] bg-white hover:bg-[#E4E3E0] text-[#141414] rounded-none font-mono text-xs font-bold uppercase flex items-center gap-1.5 shadow-[2px_2px_0px_0px_rgba(20,20,20,1)] active:translate-x-[1px] active:translate-y-[1px] active:shadow-none transition-all duration-100 cursor-pointer"
          >
            {showHandouts ? <BookOpen className="w-3.5 h-3.5 text-[#141414]" /> : <Grid className="w-3.5 h-3.5 text-[#141414]" />}
            {showHandouts ? "View Slides Player" : "View Handouts Grid (6-Slides)"}
          </button>
          {!showHandouts && (
            <button
              onClick={togglePlay}
              className={`px-3.5 py-2 border-2 border-[#141414] rounded-none font-mono text-xs font-bold uppercase flex items-center gap-1.5 shadow-[2px_2px_0px_0px_rgba(20,20,20,1)] active:translate-x-[1px] active:translate-y-[1px] active:shadow-none transition-all duration-100 cursor-pointer ${
                isPlaying
                  ? "bg-amber-100 text-black font-extrabold"
                  : "bg-black text-white hover:bg-zinc-800"
              }`}
            >
              {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
              {isPlaying ? "Pause Auto-Run" : "Start Auto-Run"}
            </button>
          )}
        </div>
      </div>

      {!showHandouts ? (
        /* Presenter Player Card */
        <div className="bg-white border-2 border-[#141414] rounded-none overflow-hidden shadow-[4px_4px_0px_0px_rgba(20,20,20,1)] relative">
          <div className="bg-[#E4E3E0] border-b-2 border-[#141414] px-6 py-3 flex justify-between items-center text-3xs font-mono font-extrabold text-[#141414] uppercase">
            <span className="font-extrabold tracking-wider text-black">
              RBM Academic Slide: Part {activeSlide.id}
            </span>
            <span>Category: {activeSlide.category}</span>
          </div>

          {/* Slide Deck Main content body */}
          <div className="p-8 sm:p-12 min-h-100 flex flex-col justify-between space-y-6">
            <div className="space-y-4">
              <h3 className="font-display text-base sm:text-lg font-black text-black uppercase tracking-wide leading-snug">
                {activeSlide.title}
              </h3>

              {/* Slide bullet lists */}
              <ul className="list-disc pl-5 space-y-3 font-sans text-xs text-zinc-900 leading-relaxed font-semibold text-justify uppercase tracking-wide">
                {activeSlide.bullets.map((b, i) => (
                  <li key={i}>{b}</li>
                ))}
              </ul>
            </div>

            {/* Slide Math formula panel */}
            {activeSlide.latexBlock && (
              <div className="bg-[#E4E3E0]/30 p-5 border-2 border-dashed border-[#141414] rounded-none text-center font-mono text-3xs text-zinc-900 shadow-inner uppercase font-bold">
                <span className="text-zinc-600 font-extrabold block text-4xs mb-1.5 tracking-widest font-sans">
                  Slide Math Equation Reference:
                </span>
                <span className="font-extrabold block break-all text-xs">{activeSlide.latexBlock}</span>
              </div>
            )}
          </div>

          {/* Slide footer buttons */}
          <div className="bg-[#E4E3E0] border-t-2 border-[#141414] px-6 py-4 flex justify-between items-center no-print">
            <button
              onClick={handlePrev}
              className="p-2 border-2 border-[#141414] bg-white hover:bg-[#E4E3E0] text-[#141414] rounded-none cursor-pointer shadow-[2px_2px_0px_0px_rgba(20,20,20,1)] active:translate-x-[1px] active:translate-y-[1px] active:shadow-none transition-all duration-100"
            >
              <ChevronLeft className="w-4 h-4 text-black" />
            </button>
            <span className="font-mono text-xs font-black uppercase text-black">
              Slide {activeSlide.id} of {slides.length}
            </span>
            <button
              onClick={handleNext}
              className="p-2 border-2 border-[#141414] bg-white hover:bg-[#E4E3E0] text-[#141414] rounded-none cursor-pointer shadow-[2px_2px_0px_0px_rgba(20,20,20,1)] active:translate-x-[1px] active:translate-y-[1px] active:shadow-none transition-all duration-100"
            >
              <ChevronRight className="w-4 h-4 text-black" />
            </button>
          </div>
        </div>
      ) : (
        /* Handouts view: 6 grids to read paper summary on 1 scan */
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {slides.map((sld) => (
            <div
              key={sld.id}
              className="bg-white border-2 border-[#141414] rounded-none p-5 shadow-[4px_4px_0px_0px_rgba(20,20,20,1)] space-y-4 flex flex-col justify-between"
            >
              <div className="space-y-3">
                <div className="flex justify-between items-center font-mono text-4xs text-zinc-500 font-extrabold uppercase">
                  <span>Slide {sld.id}</span>
                  <span className="font-extrabold text-[#141414] bg-[#E4E3E0] px-2 py-0.5 border border-[#141414] rounded-none">
                    {sld.category}
                  </span>
                </div>
                <h4 className="font-display text-xs font-black text-black uppercase tracking-tight leading-normal">
                  {sld.title}
                </h4>
                <ul className="list-disc pl-4 space-y-1.5 font-sans text-2xs text-zinc-800 leading-relaxed font-semibold uppercase tracking-wide text-justify">
                  {sld.bullets.map((b, i) => (
                    <li key={i}>{b}</li>
                  ))}
                </ul>
              </div>
              {sld.latexBlock && (
                <div className="bg-[#E4E3E0]/20 p-2.5 border border-[#141414] rounded-none text-center font-mono text-4xs text-[#141414] font-bold uppercase break-all truncate">
                  {sld.latexBlock}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
