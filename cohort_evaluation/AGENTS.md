# Quantitative evaluation handoff

Read README.md. Keep code self-contained; do not import parent scripts or access
rosie_preprocess. Never change raw images, caches, checkpoints or prior run files.
Read saved OOF prediction arrays only. Verify patient roles, exact measured/xy
matching and input provenance before comparing. Use common TRAINING thresholds
for all models and report threshold origin. Do not fit decision rules on outer
patients and label them validated; ROC-derived operating points are diagnostic.
Preserve paired-patient differences and defined counts. Bootstrap uncertainty is
conditional on predictions, not a full retraining/nested-CV result. Historical
cross-architecture comparisons are descriptive, not isolated method effects.
Use new outputs, source snapshots and completion manifest. Run tests before use.
Report actual remote versus synthetic local evaluation honestly. No GPU required.
