/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from "react";
import RbmPlayground from "./components/RbmPlayground";
import PerformanceAnalysis from "./components/PerformanceAnalysis";
import LatexDoc from "./components/LatexDoc";
import PptxSlides from "./components/PptxSlides";
import CloudMicroservice from "./components/CloudMicroservice";
import UnitTestRunner from "./components/UnitTestRunner";
import { Cpu, BarChart3, Presentation, FileText, Database, ShieldCheck, GraduationCap } from "lucide-react";

export default function App() {
  const [activeTab, setActiveTab] = useState<"simulator" | "benchmarks" | "slides" | "paper" | "cloud" | "asserts">("simulator");

  const tabs = [
    {
      id: "simulator" as const,
      label: "Model Simulator",
      icon: Cpu,
      description: "Interactive CD-k & Gibbs playground"
    },
    {
      id: "benchmarks" as const,
      label: "Hyperparameter Analysis",
      icon: BarChart3,
      description: "Comparative pipelines & charts"
    },
    {
      id: "slides" as const,
      label: "Slide Defense Deck",
      icon: Presentation,
      description: "Presentation defense PPTX preview"
    },
    {
      id: "paper" as const,
      label: "LaTeX Technical Paper",
      icon: FileText,
      description: "Academic manuscript printout PDF"
    },
    {
      id: "cloud" as const,
      label: "Scaling & Cloud Sync",
      icon: Database,
      description: "Microservices & checkpoint uploads"
    },
    {
      id: "asserts" as const,
      label: "Math Unit Asserts",
      icon: ShieldCheck,
      description: "Sandboxed QA mathematical metrics"
    },
  ];

  return (
    <div className="min-h-screen bg-[#E4E3E0] flex flex-col justify-between font-sans antialiased text-[#141414] select-text">
      {/* 1. Integrated Header with Submission context and Academic metadata */}
      <header className="bg-white border-b-3 border-[#141414] py-6 px-6 sm:px-12 block no-print">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div className="space-y-2">
            <div className="flex items-center gap-2 font-mono text-3xs font-bold text-zinc-900 uppercase tracking-widest">
              <span>Academic Project Graduation Portfolio</span>
              <span className="w-1.5 h-1.5 bg-[#141414]" />
              <span>FINAL_SUBMISSION</span>
            </div>
            <h1 className="text-3xl lg:text-5xl font-black text-[#141414] font-display uppercase tracking-tighter leading-none">
              Restricted Boltzmann Machines
            </h1>
            <p className="text-xs sm:text-sm font-bold text-zinc-600 font-sans tracking-wide uppercase">
              Theoretical Formulations and Practical Implementations
            </p>
          </div>

          {/* Student details card */}
          <div className="bg-[#E4E3E0] border-2 border-[#141414] rounded-none px-4 py-3 font-mono text-3xs text-[#141414] shadow-[3px_3px_0px_0px_rgba(20,20,20,1)] uppercase shrink-0">
            <div className="font-extrabold text-[#141414]">Candidate Student: srajam696@gmail.com</div>
            <div className="mt-1.5 pt-1.5 border-t border-dashed border-[#141414]/40 text-zinc-600 font-semibold">
              Project Ref: RBM-EBM-ENG-2026 | Course: EBM-604
            </div>
          </div>
        </div>
      </header>

      {/* 2. Primary Navigation Control (Grid or tabs inline) */}
      <nav className="bg-white border-b-2 border-[#141414] px-6 sm:px-12 py-3.5 block no-print">
        <div className="max-w-7xl mx-auto flex flex-col lg:flex-row gap-4 justify-between items-stretch lg:items-center">
          <div className="flex flex-wrap gap-2">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  title={tab.description}
                  className={`px-3 py-2 rounded-none text-xs font-mono font-bold uppercase border-2 border-[#141414] cursor-pointer transition-all duration-100 flex items-center gap-2 ${
                    isActive
                      ? "bg-[#141414] text-white shadow-none translate-x-[1px] translate-y-[1px]"
                      : "bg-white text-[#141414] hover:bg-[#E4E3E0] shadow-[2px_2px_0px_0px_rgba(20,20,20,1)] hover:-translate-x-[1px] hover:-translate-y-[1px] hover:shadow-[3px_3px_0px_0px_rgba(20,20,20,1)]"
                  }`}
                >
                  <Icon className={`w-4 h-4 ${isActive ? "text-amber-400" : "text-[#141414]"}`} />
                  {tab.label}
                </button>
              );
            })}
          </div>

          <div className="font-mono text-3xs text-zinc-600 font-extrabold tracking-widest uppercase flex items-center justify-between lg:justify-end gap-2 border-t border-dashed border-zinc-200 lg:border-t-0 pt-2 lg:pt-0">
            <span>RBM-EBM COGNITIVE ENGINE</span>
          </div>
        </div>
      </nav>

      {/* 3. Main content viewport rendering selected components */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-8">
        <div className="transition-all duration-200">
          {activeTab === "simulator" && <RbmPlayground />}
          {activeTab === "benchmarks" && <PerformanceAnalysis />}
          {activeTab === "slides" && <PptxSlides />}
          {activeTab === "paper" && <LatexDoc />}
          {activeTab === "cloud" && <CloudMicroservice />}
          {activeTab === "asserts" && <UnitTestRunner />}
        </div>
      </main>

      {/* 4. Academic Footer with strict compliance rules */}
      <footer className="bg-white border-t-3 border-[#141414] py-6 px-6 sm:px-12 text-[#141414] font-mono text-3xs font-bold uppercase block no-print mt-auto">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <GraduationCap className="w-5 h-5 text-[#141414]" />
            <span>School of Advanced Intelligent Computing | Final Degree Submission Portfolio</span>
          </div>
          <div className="text-left md:text-right text-zinc-600">
            <span>TypeScript Native Array Pipeline • No-HMR Container • Print overrides active</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
