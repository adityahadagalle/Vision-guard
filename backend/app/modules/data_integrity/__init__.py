"""Data Integrity module.

Provides dataset ingestion, quarantine, deterministic snapshots, and integrity detectors:
- Ingestion subsystem (quarantine, format adapters, snapshot manager)
- Tier-1 detectors (CorruptImageDetector, ExactDuplicateDetector, AnnotationGeometryValidator)
- Future tiers: Perceptual hashing (Tier-2), label noise/backdoor detection (Tier-3).
"""
