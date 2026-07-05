# Empirical Comparison of Quantum-Circuit-Generated and Classical Keyed Transforms for Fragile Medical Image Watermarking

**B.Tech. Final Year Major Project Proposal**

---

## Abstract

Medical images are routinely transmitted over open networks, and undetected pixel-level tampering can lead to misdiagnosis or fraudulent record alteration. Fragile watermarking is one existing method to detect such tampering. This project builds a Region-of-Interest (ROI) preserving, block-level fragile watermarking pipeline and uses it to run a controlled comparison between two ways of generating the watermark payload: a quantum circuit implemented in IBM Qiskit, and a classical keyed cryptographic transform (HMAC-SHA256, truncated to match payload size). Both payload-generation methods are embedded using identical Least Significant Bit Replacement (LSBR) logic and evaluated under an identical attack suite, so that any difference in outcome can be attributed to the transform itself rather than to implementation differences.

The project does not claim that quantum computation provides an inherent security advantage; that claim does not hold once a quantum state is measured and reduced to classical bits. Instead, the project asks a narrower, testable question: does the quantum-generated transform behave differently, in measurable ways, than a classical keyed transform of matched complexity. The expected outcome, stated honestly in advance, is that the classical transform will likely perform at least as well on standard cryptographic measures (avalanche behavior, collision rate), since HMAC-SHA256 has decades of analysis behind it. What remains genuinely open is whether the quantum path shows different behavior under hardware noise and under the specific attack types tested, which is the finding this project is designed to establish.

---

## Table of Contents

1. [Introduction and Motivation](#1-introduction-and-motivation)
2. [Objectives](#2-objectives)
3. [Literature Survey](#3-literature-survey)
4. [Methodology and Architecture](#4-methodology-and-architecture)
5. [Threat Model and Attack Suite](#5-threat-model-and-attack-suite)
6. [Evaluation Metrics](#6-evaluation-metrics)
7. [Timeline (8–10 Months)](#7-timeline-810-months)
8. [Expected Outcomes](#8-expected-outcomes)
9. [Deliverables](#9-deliverables)
10. [Risks and Mitigations](#10-risks-and-mitigations)
11. [References](#11-references)

---

## 1. Introduction and Motivation

Medical imaging modalities (ultrasound, MRI, X-ray) are frequently transmitted digitally between hospitals, labs, and insurers. A tampered image in this pipeline is not a minor error; it can change a diagnosis or support fraudulent claims. Fragile watermarking addresses this by embedding a signal into the image that breaks in a detectable way if the image is altered.

This project builds such a watermarking pipeline with one design constraint that is non-negotiable: the diagnostic Region of Interest (ROI) is never touched by the embedding process. Only non-diagnostic background regions carry the watermark. Background regions are split into small blocks (4×4 or 8×8 pixels), and each block carries an independently generated and independently checkable watermark signature.

The specific research contribution of this project is not the watermarking scheme itself, which follows established fragile watermarking design. The contribution is a controlled, side-by-side test of two different ways to generate the watermark payload for each block: a small parameterized quantum circuit, and a classical keyed cryptographic function. Both are plugged into the same embedding and extraction pipeline, so the comparison isolates the effect of the payload-generation method.

### 1.1 Why Not Claim a Quantum Security Advantage

A quantum circuit is a unitary operation. Once it is measured, its output is an ordinary classical bit string, and the only thing protecting that output from being reproduced or inverted is the secrecy of the circuit's parameters — the same assumption a classical keyed function relies on. No property of superposition or entanglement survives the measurement step to provide extra protection. Any proposal that claims otherwise is making a claim that does not hold up under basic scrutiny. This project instead treats the quantum circuit as one candidate transform function among others, and tests it empirically rather than assuming it is better because it is quantum.

---

## 2. Objectives

- **O1.** Build an ROI-preserving, block-level fragile watermarking pipeline with two interchangeable payload-generation paths: classical (Path A) and quantum (Path B).
- **O2.** Fully specify and implement both payload-generation methods with matched payload capacity (same number of bits per block).
- **O3.** Empirically measure and compare avalanche behavior, collision rate, and tamper-detection accuracy (False Positive Rate and False Negative Rate) for Path A and Path B under an identical attack suite.
- **O4.** Characterize the effect of real quantum hardware noise on Path B's determinism, and quantify the false-positive rate this noise introduces on its own, separate from any actual tampering.
- **O5.** Report the comparison honestly, including the case where Path A performs better, which is the statistically expected outcome for raw cryptographic measures.

---

## 3. Literature Survey

### 3.1 Quantum Image Representation and Its Practical Limits

The survey will cover Quantum Image Processing (QIP) representations, in particular the Novel Enhanced Quantum Representation (NEQR):

$$|\psi >=\frac{1}{\sqrt{2^{2n}}}\sum_Y\sum_X|f(X,Y)\otimes |x>\otimes|Y>\quad \text{for} X,Y=0,1,\ldots, 2^n -1$$


This section will include a worked calculation showing why simulating a full 256×256, 16-bit DICOM image on a classical computer requires a statevector of a size that is not feasible on ordinary hardware, which is the reason this project restricts itself to small blocks (4×4 or 8×8, roughly 12–14 qubits per block) rather than whole images.

### 3.2 Classical Fragile Watermarking

Coverage of Least Significant Bit Replacement (LSBR) as the embedding mechanism, and of HMAC-SHA256 and keyed permutations as established, well-analyzed payload-generation functions. This is the baseline the quantum path is measured against, not a lesser alternative included for contrast.

### 3.3 Threat Modeling in Medical Diagnostics

The literature review will establish a realistic threat model for clinical networks: direct pixel tampering, copy-move (block replay) attacks, and oracle-based re-embedding, where an attacker extracts a valid signature from one block and attempts to reuse it on a different, forged block. This section will also cover why block-content tampering in medical images carries higher real-world consequences than in generic image watermarking, which is the applied justification for the project.

---

## 4. Methodology and Architecture

### 4.1 Pipeline Overview

```
[Input 8-bit medical image, max 128x128]
[from a public, de-identified dataset]
        |
        v
Classical ROI segmentation (NumPy)
  - ROI left untouched
  - Background split into 4x4 / 8x8 blocks
        |
        +-- Path A: Classical keyed transform (HMAC-SHA256, truncated)
        |            -> payload bits per block
        |
        +-- Path B: Quantum circuit (Qiskit, ~12-14 qubits per block)
        |            -> NEQR state prep + parameterized circuit
        |            -> measurement -> payload bits per block
        |
        v
Identical LSBR embedding for both paths
        |
        v
Identical extraction + tamper-detection logic for both paths
```

Both paths share every stage of the pipeline except payload generation. This is the actual experimental control: if a difference in results appears, it can be attributed to the transform, not to differences in how the two paths are wired up.

### 4.2 Step 1: Preprocessing and Segmentation

The input is a windowed 8-bit grayscale medical image, drawn from a public, de-identified imaging dataset (e.g. an open ultrasound, MRI, or chest X-ray dataset commonly used in imaging research). No real patient data will be used, which avoids the ethical clearance requirements that would apply to identifiable clinical data. The ROI is segmented and excluded from the embedding pipeline entirely. The background is partitioned into 4×4 or 8×8 blocks.

### 4.3 Step 2: Payload Generation — Path A (Classical Baseline)

Path A computes a keyed HMAC-SHA256 over a message consisting of the block's pixel values concatenated with its spatial coordinates (row and column index within the image), then truncates the output to match the payload capacity of a single block (e.g. 16 bits for a 4×4 block at 1 bit/pixel). Including the coordinates in the HMAC input is what binds the payload to that specific block's location, which is what defeats copy-move and re-embedding attacks on this path.

### 4.4 Step 3: Payload Generation — Path B (Quantum Circuit)

Path B constructs an NEQR-style quantum state for the block's pixel values, then applies a parameterized circuit where the block's spatial coordinates are encoded as rotation angles (or equivalent gate parameters) applied to the circuit before measurement. This is the explicit mechanism by which Path B binds the payload to block location, matching the role that including coordinates in the HMAC input plays in Path A. The circuit is measured to produce a classical bit string of the same length as Path A's payload.

### 4.5 Step 4: Embedding

Both payloads, once generated, are embedded into their respective blocks using identical LSBR logic. This step is shared code between Path A and Path B; it is not re-implemented separately for each.

### 4.6 Step 5: Real Hardware Execution

A reduced version of the Path B circuit (2–3 qubits) will be executed on real IBM Quantum hardware, in addition to the noise-free simulator results. The purpose of this step is to measure the false-positive rate introduced by hardware noise alone, on unmodified blocks, separately from the false-positive and false-negative rates measured under actual tampering. This keeps "noise-driven detection failure" and "tampering-driven detection" as two separate, clearly labeled results rather than one conflated number.

**Contingency:** IBM Quantum's free-tier hardware access has variable queue times and allocation limits. If hardware time is not available within the Month 6 window, a calibrated noise model applied to the local simulator will be used as a substitute, and this substitution will be reported as a limitation in the final manuscript rather than left unstated.

---

## 5. Threat Model and Attack Suite

The following attacks will be applied, identically, to the output of both Path A and Path B:

1. Direct pixel tampering within background blocks.
2. Copy-move: relocating a block (with its embedded payload) to a different position in the image.
3. Oracle-based re-embedding: extracting a valid payload from a legitimate block and inserting it into a different, forged block.
4. Lossy compression (JPEG, at settings representative of typical DICOM wrapper handling).

---

## 6. Evaluation Metrics

- **Avalanche behavior:** fraction of output payload bits that change when one input bit (or pixel value) is flipped, measured over a minimum of 5,000 blocks per path to keep the estimate statistically meaningful at small payload sizes.
- **Collision rate:** fraction of distinct blocks producing an identical payload, measured over the same block sample, with the small payload size (16 bits for a 4×4 block) taken into account when interpreting the result, since collisions become common at this size purely due to the birthday bound, independent of transform quality.
- **False Positive Rate / False Negative Rate:** standard tamper-detection accuracy under the attack suite above, for both paths, with confidence intervals reported given the sample sizes used.
- **PSNR and SSIM:** standard image-quality metrics for the background blocks after embedding, for both paths. (Note: PSNR for the ROI itself is not reported as a result, since the ROI is untouched by design and the value is trivially unbounded — this is a design invariant, not a finding.)

---

## 7. Timeline (8–10 Months)

| Month | Activity |
|---|---|
| 1 | Literature survey; full specification of Path A (HMAC-based, with coordinate binding defined explicitly) and Path B (circuit design, with coordinate-to-parameter binding defined explicitly). Dataset selection (public, de-identified). |
| 2 | Classical pipeline: ROI segmentation, block partitioning, Path A implementation, shared LSBR embedding/extraction module. |
| 3 | Path B implementation in Qiskit on local noise-free simulator; verify embedding/extraction parity with Path A on the same test images. |
| 4–5 | Avalanche and collision-rate measurement for both paths, using the sample sizes specified in Section 6. Interim results write-up. |
| 6 | Real hardware execution of the reduced Path B circuit; measurement of noise-driven false-positive rate in isolation (contingency plan applies if hardware access is delayed). |
| 7 | Full attack suite executed against both Path A and Path B outputs. |
| 8 | Computation of FPR/FNR, PSNR, SSIM, avalanche, and collision statistics for both paths, with confidence intervals. |
| 9 | Comparative analysis write-up, stating the result plainly regardless of which path performs better. |
| 10 | Thesis and manuscript drafting; defense preparation. |

---

## 8. Expected Outcomes

Stated in advance, honestly: Path A (HMAC-SHA256) is expected to perform at least as well as Path B on avalanche and collision metrics, since it is a well-analyzed cryptographic primitive and Path B is not. This is not treated as a failure of the project. The result that matters is whether Path B shows meaningfully different behavior under hardware noise or under specific attack types, which is a question the classical literature does not already answer and which this project is positioned to answer with real measurements rather than assumptions.

---

## 9. Deliverables

- A documented GitHub repository containing: ROI segmentation code, Path A implementation, Path B implementation, the shared LSBR embedding/extraction module, and the attack suite, all under one experimental harness.
- A results report giving avalanche, collision, FPR/FNR, PSNR, and SSIM values for both paths, with the hardware-noise result reported separately.
- A thesis manuscript, structured for undergraduate submission and suitable for submission to a quantum image processing workshop or an applied medical-imaging security venue.

---

## 10. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| IBM Quantum hardware queue delays or allocation limits | Calibrated noise-model substitute on local simulator, reported as a limitation if used. |
| Small payload size (16 bits) makes collision/avalanche estimates noisy | Minimum sample size of 5,000 blocks per measurement, with confidence intervals reported rather than point estimates alone. |
| Real patient data availability/ethics clearance | Use only public, de-identified imaging datasets; no real patient data. |
| Quantum circuit simulation cost at larger block sizes | Restrict to 4×4 and 8×8 blocks (12–14 qubits), consistent with the qubit-budget analysis in Section 3.1. |

---

## 11. References

1. [To be populated during literature survey — NEQR and QIP foundational papers.]
2. [To be populated — classical fragile watermarking / LSBR references.]
3. [To be populated — HMAC-SHA256 / keyed cryptographic transform references.]
4. [To be populated — medical image authentication and threat modeling references.]
5. [To be populated — IBM Qiskit documentation and NISQ hardware noise characterization references.]

# Learning Path: Skills and Resources for the Quantum vs. Classical Watermarking Project

This is organized by what the project actually needs, in the order you'll need it. Each section lists free resources only, checked as current. Skip anything you already know — this is a reference, not a checklist you have to complete top to bottom.

---

## 1. Python and Image Processing Basics (Month 1–2 need)

You need enough NumPy and basic image handling to do ROI segmentation and block partitioning.

- **NumPy official quickstart** — covers array slicing and indexing, which is really all you need for block partitioning: https://numpy.org/doc/stable/user/quickstart.html
- **Real Python — NumPy image processing basics**: https://realpython.com/numpy-tutorial/
- **OpenCV Python tutorials** (for reading images, converting to grayscale, basic thresholding for ROI segmentation): https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
- **scikit-image** (if you want a higher-level, more Python-native alternative to OpenCV for segmentation): https://scikit-image.org/docs/stable/user_guide.html

You don't need deep computer vision knowledge here. Segmenting a fixed ROI (e.g. a rectangular diagnostic region) and splitting the rest into fixed-size blocks is a small amount of array indexing.

---

## 2. Classical Cryptography: HMAC and Keyed Transforms (Month 1–2 need)

This is Path A. You need to understand what HMAC actually does, not just call a library function.

- **Real Python — Hashing in Python (hashlib, hmac)**: https://realpython.com/python-hashlib/
- **Python official docs — `hmac` module**: https://docs.python.org/3/library/hmac.html
- **NIST FIPS 198-1 — the HMAC standard itself** (short, worth reading once so you understand what "keyed" actually secures): https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.198-1.pdf
- **"Avalanche effect" and "collision resistance" — plain explanation** (Wikipedia is fine here, it's a stable reference for definitions, not a citation for your thesis): https://en.wikipedia.org/wiki/Avalanche_effect

The one thing to actually understand, not just implement: why including the block's coordinates as part of the HMAC input is what binds the payload to that block. If you can explain that in one sentence without notes, you're ready to implement Path A.

---

## 3. Quantum Computing Fundamentals (Month 1–3 need)

You do not need a full quantum computing course before starting. You need the basics: qubits, superposition, measurement, and simple gates. Everything past that can be learned as you go, tied to the specific circuit you're building.

- **IBM Quantum Learning — "Basics of quantum information"** (the current, actively maintained IBM course, part of their "Understanding quantum information and computation" series hosted on their platform, now at quantum.cloud.ibm.com after their 2025 platform migration): https://quantum.cloud.ibm.com/learning
- **The same series on YouTube** (if you prefer video first, reading second): search "Qiskit YouTube Understanding quantum information and computation" — the channel is youtube.com/@qiskit
- **Qiskit Global Summer School** — a free, structured two-week program IBM runs annually, with a beginner track for people with no prior Qiskit experience. It's not always running, but recordings from past years are usually left up. Check: https://www.ibm.com/quantum/blog/qiskit-summer-school-2026
- **Qiskit documentation — "Getting started"**: https://docs.quantum.ibm.com/start

You need roughly: what a qubit is, what a quantum circuit diagram means, what measurement does (destroys superposition, gives classical bits), and how to build a circuit with basic gates (X, H, CNOT, and parameterized rotation gates like RY/RZ, since you'll need rotation gates for coordinate binding in Path B). That's a few days of focused study, not a semester.

---

## 4. NEQR Specifically (Month 1–4 need)

This is the actual technique your Path B circuit is built on, so it's worth reading the original source rather than only secondary explanations.

- **Original NEQR paper**: Zhang, Y., Lu, K., Gao, Y., Wang, M. (2013). "NEQR: a novel enhanced quantum representation of digital images." *Quantum Information Processing*, 12(8), 2833–2860. DOI: 10.1007/s11128-013-0567-z. A copy is available via ResearchGate: https://www.researchgate.net/publication/257641933_NEQR_A_novel_enhanced_quantum_representation_of_digital_images
- **Working NEQR-in-Qiskit code examples** — genuinely useful since it shows a working implementation rather than only the math: https://github.com/t11s/NEQR-examples
- **"Quantum Image Processing in Practice" (Agaian et al.)** — has a clear chapter on quantum image representations including NEQR, useful if the original paper's notation is hard going at first: available via Wiley Online Library (your institution's library access may be needed, check with your supervisor).

Read the original paper for the state-preparation equation your proposal already cites, then go straight to the GitHub code to see how it's actually built in Qiskit. Going code-first after a first read of the paper is faster than trying to fully absorb the paper's math in isolation.

---

## 5. Real Hardware Execution (Month 6 need)

- **IBM Quantum Platform** (where you request time on real hardware): https://quantum.cloud.ibm.com/ — sign-up gives a small free monthly allocation of execution time on real QPUs.
- **Qiskit Runtime documentation** (Sampler/Estimator primitives, how jobs are actually submitted and queued): https://docs.quantum.ibm.com/run
- **IBM's noise and error mitigation documentation** — relevant for understanding what you're actually measuring when you look at hardware-induced false positives: https://docs.quantum.ibm.com/guides/error-mitigation-and-suppression-techniques

Check queue times before committing to a hardware-run date in your timeline — this varies and is the single most likely scheduling risk in the whole project, as already flagged in your proposal's risk table.

---

## 6. Statistics for the Comparison (Month 8–9 need)

You need this to report FPR/FNR, avalanche, and collision rates honestly, with confidence intervals rather than bare numbers.

- **Real Python — confidence intervals in Python**: https://realpython.com/python-confidence-intervals/
- **StatQuest (YouTube channel) — "Confidence Intervals" and "Hypothesis Testing"** — genuinely clear, no-jargon explanations if statistics isn't your strong area: https://www.youtube.com/@statquest
- **scipy.stats documentation** (for computing the actual intervals in code): https://docs.scipy.org/doc/scipy/reference/stats.html

The specific thing to get right: at a 16-bit payload size, your sample size (the proposal specifies a minimum of 5,000 blocks) determines how much you can trust a measured rate. Don't skip the confidence interval step — a bare percentage at this scale is close to meaningless on its own.

---

## 7. Datasets (Month 1 need)

Public, de-identified medical imaging data, so you avoid any patient-data ethics clearance requirement.

- **MedMNIST** — a standardized collection of public medical imaging datasets (X-ray, ultrasound, OCT, CT), pre-processed and free to use for research/educational purposes under Creative Commons licensing. Install via `pip install medmnist`. Official site: https://medmnist.com/ (Note: MedMNIST's own documentation states the dataset is not intended for actual clinical use, which is fine for this project since you only need realistic-looking images to test a watermarking pipeline, not to make any diagnostic claims.)
- **NIH Chest X-ray dataset** (larger, full-resolution X-rays, if you want higher-resolution test images than MedMNIST's default small size): https://www.kaggle.com/datasets/nih-chest-xrays/data

MedMNIST's default images are quite small (28×28), so for your 128×128 windowed input you may want MedMNIST's larger-size variant (MedMNIST+, up to 224×224) or the NIH dataset directly, then crop/resize to your working size.

---

## 8. LaTeX for the Thesis (Month 9–10 need)

- **Overleaf's free LaTeX tutorials** — the most beginner-friendly starting point, and lets you write without installing anything locally: https://www.overleaf.com/learn
- **"Not So Short Introduction to LaTeX2e"** — the standard free reference document nearly every academic recommends: https://tobi.oetiker.ch/lshort/lshort.pdf

If your `.tex` proposal file compiles already (it does — we tested it), you have most of the structure you'll need for the thesis; it's mostly a matter of extending the same document with your actual results.

---

## Suggested Order

1. Python/NumPy image basics + HMAC (can be done in parallel, roughly week 1–3)
2. Quantum computing fundamentals (week 2–4, can overlap with the above)
3. NEQR paper + Qiskit code examples (week 4–6)
4. Build Path A, then Path B, testing both independently before combining
5. Statistics resources — read before Month 8, not during, so you're not learning confidence intervals under deadline pressure
6. Dataset setup — do this early (Month 1), it's a common thing people leave too late and then lose a week to license/format issues
7. LaTeX — start the thesis document skeleton early (even empty section headers), not just in Month 9

None of this needs to be mastered in the deep, research-level sense. You need working competence: enough to implement, test, and explain what you built and why, in plain terms, to your supervisor and eventually a reviewer.
