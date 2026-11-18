# VisionGuard Data, Model & Receipt Format Specifications

**Document Version:** 1.1.0 (Audited)  
**Status:** Canonical Data Format Standard  
**Date:** November 2026  
**Scope:** Strict structural schemas, format-specific coordinate validation contracts, safe PyTorch checkpoint loading policies, and canonical Ed25519 inference receipts.

---

## 1. Supported Computer Vision Dataset Layouts & Coordinate Contracts

Format-specific validation rules must **never** be mixed or cross-applied.

### 1.1 Folder-per-Class Classification Dataset
```text
dataset_root/
├── provenance.json          # (Optional) Provenance manifest
├── class_a/
│   ├── img_001.jpg
│   └── img_002.png
└── class_b/
    ├── img_003.jpg
    └── img_004.jpg
```
- Subfolder names represent ground-truth category strings.
- Only `.jpg`, `.jpeg`, and `.png` files are ingested; hidden OS files are ignored.

---

### 1.2 COCO 2017 Format (Object Detection)
```text
dataset_root/
├── provenance.json          # (Optional) Provenance manifest
├── annotations/
│   ├── instances_train.json
│   └── instances_val.json
└── images/
    ├── train/
    │   └── 000000000001.jpg
    └── val/
        └── 000000000002.jpg
```

#### COCO Coordinate Validation Specification
In the COCO format, bounding boxes are defined by the 4-tuple:
$$\text{bbox} = [x_{\min}, y_{\min}, w, h]$$

> **CRITICAL COCO COORDINATE RULES:**
> 1. **Absolute Pixel Units:** Coordinates in COCO are **absolute floating-point pixel values**, NOT normalized.
> 2. **Top-Left Origin:** $(x_{\min}, y_{\min})$ represents the top-left coordinate of the bounding box in image pixel space.
> 3. **Positive Dimensions:** Width $w$ and height $h$ must be strictly positive:
>    $$w > 0.0, \quad h > 0.0$$
> 4. **Image Canvas Boundaries:** Given image dimensions $W_{\text{image}}$ and $H_{\text{image}}$ (extracted from `images[].width` and `images[].height` and verified against actual image file headers):
>    $$x_{\min} \ge 0.0, \quad y_{\min} \ge 0.0$$
>    $$x_{\min} + w \le W_{\text{image}}$$
>    $$y_{\min} + h \le H_{\text{image}}$$
> 5. **Declared Bbox Convention:** If an external dataset declares alternative bbox conventions (e.g., `[x_min, y_min, x_max, y_max]`), this must be explicitly specified in dataset metadata; default parser strictly assumes `[x_min, y_min, width, height]`.
> 6. **Foreign Key Integrity:** Every `annotation.image_id` must reference a valid `images[].id`. Every `annotation.category_id` must reference a valid `categories[].id`.

---

### 1.3 YOLO Format (Object Detection)
```text
dataset_root/
├── provenance.json          # (Optional) Provenance manifest
├── dataset.yaml             # Class names and split pointers
├── images/
│   ├── train/
│   │   └── img_001.jpg
│   └── val/
│       └── img_002.jpg
└── labels/
    ├── train/
    │   └── img_001.txt
    └── val/
        └── img_002.txt
```

#### YOLO Coordinate Validation Specification
In the YOLO format, each object instance is a line-delimited text record:
$$\text{record} = \langle \text{class\_id} \rangle \quad \langle x_{\text{center}} \rangle \quad \langle y_{\text{center}} \rangle \quad \langle w \rangle \quad \langle h \rangle$$

> **CRITICAL YOLO COORDINATE RULES:**
> 1. **Normalized Floating-Point Units:** Coordinates in YOLO are strictly **normalized to the range $[0.0, 1.0]$** relative to image width and height.
> 2. **Box Center Origin:** $(x_c, y_c)$ represents the geometric center of the bounding box.
> 3. **Positive Dimensions:** Normalized width and height must satisfy:
>    $$0.0 < w \le 1.0, \quad 0.0 < h \le 1.0$$
> 4. **Boundary Range Constraints:** Box boundaries must strictly lie within the $[0.0, 1.0]$ unit interval:
>    $$x_c - \frac{w}{2} \ge 0.0, \quad x_c + \frac{w}{2} \le 1.0$$
>    $$y_c - \frac{h}{2} \ge 0.0, \quad y_c + \frac{h}{2} \le 1.0$$
> 5. **Class ID Integrity:** $\text{class\_id}$ must be an integer in $\{0, 1, \dots, K-1\}$, mapped directly against `dataset.yaml` class declarations.
> 6. **Parsing & Missing-Label Conditions:**
>    - If an image exists with an **empty** `.txt` label file, it represents a valid negative sample (image containing zero objects of interest).
>    - If an image exists with a **missing** `.txt` label file (file absent from directory), this is flagged as an `ORPHANED_IMAGE_ERROR`.
>    - If a `.txt` label file exists without an accompanying image file, it is flagged as an `ORPHANED_LABEL_ERROR`.

---

### 1.4 CIFAR-10 Official Binary Ingestion Format (`cifar-10-binary.tar.gz`)
To eliminate arbitrary code execution risks from Python pickled datasets (`cifar-10-python.tar.gz`), VisionGuard mandates the official binary distribution:
```text
cifar-10-batches-bin/
├── data_batch_1.bin         # Training batch 1 (10,000 samples)
├── data_batch_2.bin         # Training batch 2 (10,000 samples)
├── data_batch_3.bin         # Training batch 3 (10,000 samples)
├── data_batch_4.bin         # Training batch 4 (10,000 samples)
├── data_batch_5.bin         # Training batch 5 (10,000 samples)
├── test_batch.bin           # Test batch (10,000 samples)
├── batches.meta.txt         # Class labels listing
└── provenance.json          # (Optional) Provenance manifest
```

#### Binary Record Layout Specification
Each binary file contains contiguous fixed-size records of exactly **3,073 bytes per image**:
- **Byte 0 (1 byte):** Unsigned 8-bit integer representing the ground-truth class label in $\{0, 1, \dots, 9\}$.
- **Bytes 1–1024 (1024 bytes):** Red channel pixel intensities ($32 \times 32$ matrix in row-major order).
- **Bytes 1025–2048 (1024 bytes):** Green channel pixel intensities ($32 \times 32$ matrix in row-major order).
- **Bytes 2049–3072 (1024 bytes):** Blue channel pixel intensities ($32 \times 32$ matrix in row-major order).

#### Tensor Layout & Transformation Contract
- **Native Memory Ingestion:** Ingested via direct buffer reading (`numpy.frombuffer`) yielding a tensor shape of `(3, 32, 32)` in **channel-first (CHW)** layout.
- **Explicit Conversion for Image Libraries:** For libraries requiring height-width-channel format (Pillow, OpenCV, matplotlib, or HWC visual transforms), VisionGuard mandates an explicit transposition:
  $$\mathbf{X}_{\text{HWC}} = \text{transpose}(\mathbf{X}_{\text{CHW}}, (1, 2, 0)) \implies (32, 32, 3)$$
- **Zero Unpickling:** Ingestion is performed strictly through primitive byte parsing without executing Python object serialization.

#### Artifact Identity vs. Authenticity
- A locally calculated SHA-256 digest of `cifar-10-binary.tar.gz` establishes **artifact identity** (confirming that local bytes match an expected hash).
- It does **NOT establish authenticity** unless explicitly matched against an independent, authoritative external source (e.g., official University of Toronto cryptographic checksums published on an authentic academic domain or an Ed25519-signed release receipt).

---

## 2. Image File Physical Guardrails

| Constraint | Limit / Requirement | Violation Action | Security Rationale |
|---|---|---|---|
| **Supported Formats** | JPEG / JPG, PNG | Immediate rejection (`unsupported_format`) | Prevents unvetted image codecs from triggering buffer overflows. |
| **Magic Byte Signatures** | JPEG: `\xFF\xD8\xFF`<br>PNG: `\x89PNG\r\n\x1a\n` | Rejection (`corrupted_magic_bytes`) | Prevents polyglot or disguised executable files. |
| **Max Image Pixels** | 40,000,000 pixels ($8000 \times 5000$) | Rejection (`decompression_bomb_risk`) | Prevents memory allocation exhaustion attacks. |
| **Max File Size** | 250 MB per file | Rejection (`file_size_exceeded`) | Prevents disk exhaustion DoS. |
| **Color Spaces** | RGB, RGBA, Grayscale (1-channel) | Converted to RGB | Normalizes input tensors for PyTorch processing. |

---

## 3. Provenance Manifest Specification (`provenance.json`)

To enable VisionGuard's **Contributor-Level Risk Tracing**, datasets may supply an optional `provenance.json` manifest:

```json
{
  "manifest_version": "1.0.0",
  "dataset_name": "Traffic-Safety-2026",
  "provenance_records": [
    {
      "relative_path": "images/train/img_001.jpg",
      "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      "contributor_id": "contractor_batch_04",
      "source_id": "sensor_front_camera_02",
      "batch_id": "batch_2026_11_alpha",
      "upload_session_id": "session_9941"
    }
  ]
}
```
*Rule:* Missing provenance records remain `null`. The system **never** fabricates or guesses contributor identifiers.

---

## 4. PyTorch Checkpoint Security & Ingestion Policy

### 4.1 Strict Loading Prohibition
> **MANDATORY SECURITY DIRECTIVE:**  
> `torch.load(..., weights_only=False)` is **STRICTLY PROHIBITED** on all untrusted model files. Standard PyTorch loading invokes Python's `pickle` engine, allowing arbitrary code execution via `__reduce__` exploit payloads.

### 4.2 Limitations of `weights_only=True` & Relevant PyTorch Advisories
- PyTorch's `weights_only=True` (introduced in PyTorch 1.13 and made default in 2.6) restricts unpickling to a whitelist of primitive types and tensors.
- **`weights_only=True` is NOT a complete security sandbox:**
  - Upstream vulnerabilities in PyTorch unpickling and C++ archive parsing have demonstrated that type restrictions and memory corruptions occur:
    - **CVE-2025-32434:** PyTorch deserialization bypass in `torch.load(weights_only=True)` allowing arbitrary object instantiation in versions $< 2.5.1 / 2.6.0$. Runtime must be $\ge 2.6.0$ or patched baseline.
    - **CVE-2026-24747 (GHSA-63cw-57p8-fm3p):** PyTorch C++ unpickler integer overflow during tensor buffer calculation in versions $\le 2.9.1$. Patched in $\ge 2.10.0$. Causes `weights_only` unpickler memory corruption and potential arbitrary code execution.
    - **GHSA-v32g-mr63-3j83:** PyTorch advisory published November 10, 2026. Reported affected versions $\ge 2.14.0$, reported patched in $2.15.0$. Identifies potential arbitrary file read during `torch.load(weights_only=True)`. No CVE currently assigned; noted as a newly published AI-assisted advisory with no package listed, tracked for upstream confirmation. Current project environment uses `torch==2.15.0+cpu` (meeting the 2.15.0 patch baseline and exceeding 2.10.0), which must not be downgraded.
  - An attacker can construct a valid tensor-only checkpoint with astronomical dimensions (e.g., shape $[10^6, 10^6, 10^6]$), causing immediate host memory exhaustion (Denial of Service) during tensor allocation.
- **Python AST parsing or pickle opcode inspection cannot guarantee safety:** The pickle bytecode is a Turing-complete stack machine; obfuscated instruction sequences cannot be statically proven benign.

### 4.3 Mandatory Checkpoint Sandboxing Architecture & Minimum Conditions
> **IMPORTANT SECURITY STATUS:**  
> The OS isolation mechanisms specified below represent **architectural specifications**, not implemented security guarantees. Until verified in runtime testing, untrusted checkpoints cannot be treated as contained.

VisionGuard specifies the following **minimum conditions for permitting untrusted checkpoint deserialization**:
1. **Strict Process Isolation:** Non-elevated child process running in an unprivileged execution context (Windows `Low Integrity Level` / `AppContainer`; Linux unshared user namespace `unshare -U`), completely separated from the host supervisor process.
2. **Complete Network Isolation:** Network sockets disabled (loopback disabled or network namespace `unshare -n` with zero routes).
3. **Restricted Filesystem Access:** Read-only handle to candidate checkpoint file; write access restricted strictly to an isolated temporary scratch directory (`tmpfs`) purged upon process exit. Zero access to host secrets or system directories.
4. **Strict Resource Quotas:** Memory limit capped via Windows Job Objects or Linux cgroups v2 (`memory.max`) to 2 GB max, with a 30-second hard execution timeout terminated via `SIGKILL`.

**Capability Gating on Sandbox Absence:**  
If any of these minimum isolation conditions (network isolation, restricted filesystem access, or process isolation) cannot be proven on the host platform, VisionGuard **strictly refuses to deserialize untrusted checkpoints** and reports:
- `status: UNAVAILABLE`
- `reason: "Host platform lacks proven process sandboxing, network severance, or filesystem isolation controls required for untrusted checkpoint deserialization"`

5. **Post-Load State-Dict Verification:**
   - Verifies that the loaded object is strictly a `dict` mapping string keys to `torch.Tensor` instances.
   - Rejects any checkpoint containing custom non-tensor Python objects.
   - Validates tensor dimensions against expected architecture contracts ($W \times H \times C \le 2048 \times 2048 \times 3$).

### 4.4 Future Safertensors Adoption (Pure Tensor Storage)
VisionGuard will evaluate `safetensors` (HuggingFace format: pure JSON header + raw binary buffer, zero executable code, zero unpickling) as a primary storage format for model weights while remaining 100% PyTorch-native via `safetensors.torch.load_file()`.

---

## 5. Inference Receipt & Canonical Serialization Standard

### 5.1 Receipt Schema (`InferenceReceipt`)
```json
{
  "receipt_version": "1.0.0",
  "sequence_id": 10452,
  "event_id": "c7a8b89e-5e92-411a-9f44-98442886f77a",
  "timestamp": "2026-11-10T18:45:00.123456Z",
  "input_digest": "3a7bd3e2360a3d29eea436fcfb7e44c735d117c42d1c1835420b6b9942dd4f1b",
  "model_digest": "4b227777d4dd1fc61c6f884f48641d02b4d121d3fd328cb08b5531fcacdabf8a",
  "config_digest": "e2b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "output_payload": {
    "predicted_class": 3,
    "confidence": 0.9842,
    "logits": [0.001, 0.002, 0.005, 0.984, 0.004, 0.001, 0.001, 0.001, 0.001, 0.000]
  },
  "previous_receipt_hash": "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
  "signer_key_id": "ed25519_key_node_01"
}
```

### 5.2 Canonical Serialization Rules (RFC 8785 JCS)
To ensure that digital signatures and SHA-256 hashes are reproducible across different platforms, JSON serialization follows **RFC 8785**:
1. **Key Ordering:** Object keys are sorted lexicographically by UTF-8 byte order at all levels.
2. **Whitespace:** Zero insignificant whitespace (use `,` and `:` without surrounding spaces).
3. **Encoding:** Strict UTF-8 without Byte Order Mark (BOM).
4. **Number Representation:** Floats formatted using standard IEEE 754 representation without trailing exponent zeros.
5. **Signing:** Ed25519 signature computed over canonical UTF-8 bytes using Python `cryptography` library.
