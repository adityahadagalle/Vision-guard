# Dataset Selection, Legal Licensing Audit & Ingestion Security

**Document Version:** 1.2.0 (Audited Patch)  
**Status:** Canonical Evaluation Benchmark & Legal Licensing Specification  
**Date:** November 2026  
**Scope:** Rigorous legal licensing audit of candidate computer-vision datasets, legal facts vs. project policies, binary CIFAR-10 ingestion security, and hardware sizing for NVIDIA RTX 3050 (4GB VRAM) / CPU fallback.

---

## 1. Executive Licensing & Redistribution Policy

VisionGuard strictly separates **verifiable legal copyright facts** from **conservative project policies**:

```text
┌────────────────────────────────────────────────────────────────────────┐
│ Legal Facts & Authoritative Terms                                      │
│   • CIFAR-10: Research grant from Toronto; commercial rights           │
│     unresolved (parent 80M Tiny Images retracted by MIT in 2020).      │
│   • GTSRB: Non-commercial research grant; commercial requires consent. │
│   • COCO Annotations: CC BY 4.0 (Commercial & redistribution allowed). │
│   • COCO Images: Individual Flickr photographers (Heterogeneous terms).│
│   • Pascal VOC: Academic research and educational grant only.          │
├────────────────────────────────────────────────────────────────────────┤
│ VisionGuard Conservative Project Policy                                │
│   • Conservative Stance: The platform WILL NOT bundle, mirror, or      │
│     redistribute raw third-party image datasets in its repository or   │
│     Docker images without confirmed, explicit redistribution rights.   │
│   • Automated Download Scripts: End-user fetches data directly from    │
│     authoritative official sources via scripts with license prompts.   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. In-Depth Dataset Legal Audits

### 2.1 CIFAR-10 & CIFAR-100
1. **Official Source:** Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton (University of Toronto) — [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html).
2. **Explicit Legal License:** **UNVERIFIED / Research Grant Only.** No formal open-source license (such as MIT, Apache-2.0, or CC-BY) is granted on the official site.
3. **Copyright Ownership:** Sourced from the larger "80 Million Tiny Images" dataset collected in 2008 by Antonio Torralba, Rob Fergus, and William T. Freeman at MIT. Individual image copyrights belong to original web photographers (scraped from Google Images).
4. **Research vs. Commercial Restrictions:**
   - *Research Use:* Permitted by University of Toronto ("free to use for research purposes").
   - *Commercial Use:* **Unresolved and unverified.** There is no commercial grant. (Note: MIT officially retracted the parent 80 Million Tiny Images dataset in 2020 due to ethical concerns; CIFAR-10 remains widely used in academic ML security benchmarks, but commercial rights are unresolved).
5. **Redistribution Status:** Authoritative terms do not explicitly authorize third-party redistribution.
6. **Attribution Requirements:** Citation of Krizhevsky (2009) *Learning Multiple Layers of Features from Tiny Images*.
7. **Image vs. Annotation Terms:** Unified distribution format.
8. **VisionGuard Project Policy:** VisionGuard will **not bundle** CIFAR-10; users fetch the dataset directly using `scripts/data/download_benchmarks.py`.

---

### 2.2 GTSRB (German Traffic Sign Recognition Benchmark)
1. **Official Source:** Institut für Neuroinformatik (INI), Ruhr-Universität Bochum (Stallkamp et al., 2011/2012) — [https://benchmark.ini.rub.de/](https://benchmark.ini.rub.de/).
2. **Explicit Legal License:** **Custom Non-Commercial Research Agreement** (or CC BY-SA 4.0 depending on data repository mirror).
3. **Copyright Ownership:** Institut für Neuroinformatik, Ruhr-Universität Bochum, Germany.
4. **Research vs. Commercial Restrictions:**
   - *Research Use:* Freely permitted for scientific, non-commercial, and educational research.
   - *Commercial Use:* **Unresolved without consent.** Commercial use requires prior written consent from INI Ruhr-Universität Bochum.
5. **Redistribution Status:** Redistribution requires retention of copyright notices and attribution; commercial redistribution is restricted without consent.
6. **Attribution Requirements:** Must cite Stallkamp et al. (IJCNN 2011 / Neural Networks 2012).
7. **Image vs. Annotation Terms:** Images and CSV annotation files share identical terms under the INI agreement.
8. **VisionGuard Project Policy:** VisionGuard will **not bundle** raw GTSRB image files; automated download scripts fetch from official INI mirrors.

---

### 2.3 COCO 2017 (Common Objects in Context)
1. **Official Source:** Microsoft COCO Consortium (Lin et al., 2014) — [https://cocodataset.org/](https://cocodataset.org/).
2. **Explicit Legal Licenses (Two Distinct Records):**
   - **COCO Annotations:** **Creative Commons Attribution 4.0 International (CC BY 4.0)**.
   - **COCO Images:** **Individual Flickr User Licenses** (heterogeneous terms).
3. **Copyright Ownership:**
   - *Annotations:* Copyright © COCO Consortium.
   - *Images:* Copyright belongs to individual Flickr photographers. COCO does not own image copyrights.
4. **Research vs. Commercial Restrictions:**
   - *Annotations:* Commercial use, modification, and redistribution are **legally permitted** under CC BY 4.0 with attribution.
   - *Images:* Dependent on individual photo terms (some CC BY, some CC BY-NC, some All Rights Reserved).
5. **Redistribution Status:**
   - *Annotations:* Can be legally redistributed with CC BY 4.0 attribution.
   - *Images:* Bulk image redistribution is not granted by the COCO Consortium and is subject to Flickr terms.
6. **Attribution Requirements:** Must cite Lin et al., *Microsoft COCO: Common Objects in Context*, ECCV 2014.
7. **VisionGuard Project Policy:** Synthetic test annotation JSON files may be included as unit-test fixtures with attribution; raw images are **never bundled**.

---

### 2.4 Pascal VOC 2007 (Visual Object Classes Challenge)
1. **Official Source:** University of Oxford / University of Leeds (Everingham et al.) — [http://host.robots.ox.ac.uk/pascal/VOC/](http://host.robots.ox.ac.uk/pascal/VOC/).
2. **Explicit Legal License:** **Academic Research Use Only** (No standardized open-source license grant).
3. **Copyright Ownership:** University of Oxford, University of Leeds, and individual Flickr photographers.
4. **Research vs. Commercial Restrictions:**
   - *Research Use:* Permitted solely for academic research and education.
   - *Commercial Use:* **Unresolved and generally unauthorized.** Sourced from Flickr under academic challenge policy.
5. **Redistribution Status:** Redistribution of full database mirrors requires permission from VOC organizers.
6. **Attribution Requirements:** Must cite Everingham et al., *The PASCAL Visual Object Classes Challenge*, IJCV 2010.
7. **VisionGuard Project Policy:** VisionGuard will **not bundle** raw VOC data; user-executed scripts required.

---

## 3. Dataset Summary Matrix: Legal Facts vs. Project Policy

| Dataset | Primary Source | Verified Legal Status | Commercial Rights Status | Images vs. Annotations Terms | VisionGuard Conservative Policy |
|---|---|---|---|---|---|
| **CIFAR-10** | Univ. of Toronto | Research grant only | **Unresolved / Unverified** | Unified | **No Bundling;** User script downloads directly |
| **GTSRB** | INI Ruhr-Univ. Bochum | Non-commercial research agreement | **Requires Written Consent** | Unified | **No Bundling;** User script downloads directly |
| **COCO 2017** | COCO Consortium | Annotations: CC BY 4.0; Images: Flickr | **Annotations: YES; Images: Mixed** | **Separate Records** | **Annotations Fixtures Allowed; Raw Images NEVER Bundled** |
| **Pascal VOC 2007** | Oxford / Leeds | Academic use only | **Unresolved / Unauthorized** | Unified | **No Bundling;** User script downloads directly |

---

## 4. Secure CIFAR-10 Ingestion Specification (Binary Distribution)

### 4.1 Insecurity of Python Distribution
The commonly distributed `cifar-10-python.tar.gz` package serializes image batches via Python `pickle` (`cPickle` dictionaries). Ingesting pickled datasets introduces identical arbitrary code execution risks as loading untrusted PyTorch checkpoints.

### 4.2 Mandatory Official Binary Distribution
VisionGuard strictly specifies the official **binary distribution** (`cifar-10-binary.tar.gz`):
- **Archive Contents:** 5 training binary files (`data_batch_1.bin` to `data_batch_5.bin`) and 1 test binary file (`test_batch.bin`).
- **Archive Integrity Verification:** The SHA-256 digest of `cifar-10-binary.tar.gz` is verified against `configs/datasets/checksums.json` before unpacking.
- **Binary Record Layout:** Exactly 3,073 contiguous bytes per image record:
  - Byte 0 (1 byte): Integer class label ($0 \dots 9$).
  - Bytes 1–1024 (1024 bytes): Red channel ($32 \times 32$ pixels, row-major order).
  - Bytes 1025–2048 (1024 bytes): Green channel ($32 \times 32$ pixels, row-major order).
  - Bytes 2049–3072 (1024 bytes): Blue channel ($32 \times 32$ pixels, row-major order).
- **Tensor Shape & Layout Conversions:**
  - Raw binary reading yields a flat 3072-byte buffer reshaped into `(3, 32, 32)` channel-first (CHW) format.
  - An explicit transposition to `(32, 32, 3)` height-width-channel (HWC) format is performed when interfacing with image processing libraries (Pillow, OpenCV) or visualization tools:
    $$\mathbf{X}_{\text{HWC}} = \text{transpose}(\mathbf{X}_{\text{CHW}}, (1, 2, 0)) \implies (32, 32, 3)$$
- **Safe Ingestion Implementation:**
  ```python
  import numpy as np


  def load_cifar10_binary_batch(filepath: str) -> tuple[np.ndarray, np.ndarray]:
      with open(filepath, "rb") as f:
          buf = f.read()
      # 10,000 images * 3,073 bytes = 30,730,000 bytes
      data = np.frombuffer(buf, dtype=np.uint8).reshape(10000, 3073)
      labels = data[:, 0]
      # Native raw shape is (10000, 3, 32, 32) CHW; transpose to (10000, 32, 32, 3) HWC
      images_chw = data[:, 1:].reshape(10000, 3, 32, 32)
      images_hwc = images_chw.transpose(0, 2, 3, 1)
      return images_hwc, labels
  ```
- **Security Guarantee:** Executed via raw byte manipulation with **zero unpickling** (`pickle.load` is never invoked).
- **Identity vs. Authenticity Disclaimer:**
  - Locally calculated SHA-256 establishes **artifact identity** (confirming that local bytes match an expected digest).
  - It does **NOT establish authenticity** unless compared with a trusted independent source (e.g., official upstream cryptographic checksums published on an authentic academic domain).

---

## 5. Hardware Sizing & Unverified Estimates

> **AUDIT MANDATE:**  
> All memory footprints and execution runtimes in this section are **UNVERIFIED ESTIMATES** based on theoretical computational complexity. They must **not** be presented as measured VisionGuard benchmark results until Phase 2 formal benchmarking is executed.

- **GPU Target:** NVIDIA GeForce RTX 3050 Laptop GPU (4,096 MiB VRAM).
- **CPU Target:** x86_64 Multi-core processor (AVX2 supported), 16 GB RAM.

| Benchmark Dataset | Format & Scale | Approximate Storage | Estimated Peak VRAM ($B=32$) | Estimated Execution Profile |
|---|---|---|---|---|
| **CIFAR-10 (Binary)** | Raw binary ($60,000$ imgs, $32 \times 32$) | $\sim 170$ MB | *UNVERIFIED ESTIMATE:* $< 1$ GB | Fast local iteration; suitable for repeated poison runs. |
| **GTSRB** | Folders/CSV ($51,839$ imgs, resized to $64 \times 64$) | $\sim 300$ MB | *UNVERIFIED ESTIMATE:* $< 1.2$ GB | Safety-critical traffic sign backdoor testing. |
| **COCO-Val 2017** | JSON + Images ($5,000$ imgs, $640 \times 480$) | $\sim 1$ GB | *UNVERIFIED ESTIMATE:* $< 1.8$ GB ($B=16$) | Evaluates absolute pixel bounding boxes and complex scenes. |
| **Pascal VOC 2007** | XML/TXT + Images ($5,011$ imgs, $500 \times 375$) | $\sim 1$ GB | *UNVERIFIED ESTIMATE:* $< 1.8$ GB ($B=16$) | Evaluates normalized YOLO/VOC bounding boxes. |
