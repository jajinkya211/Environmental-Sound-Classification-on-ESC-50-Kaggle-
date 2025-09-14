Environmental Sound Classification on ESC-50 (Kaggle)

End-to-end ESC-50 pipeline featuring a robust feature stack (log-mel + PCEN), a strong spectrogram CNN, a pretrained waveform backbone (wav2vec-style) with attentive pooling and partial unfreezing, 5-fold cross-validation, test-time augmentation (TTA), ensembling, rich diagnostics, and exportable checkpoints.

Highlights

Data & splits: ESC-50 (2,000 clips, 50 classes, 5 official folds). Uses esc50.csv and filepaths from the Kaggle dataset.

Audio I/O: Resample to 16 kHz, 5-second mono segments.

Features: Power MelSpectrogram (64 bins) → log-mel (dB) + PCEN (with robust fallback: torchaudio.functional, torchaudio.transforms.PCEN, then librosa.pcen).

Augmentations: SpecAugment (time/freq masking), light waveform noise/gain; optional mixup in feature space.

Models

Model A (Spec-CNN): compact CNN with SE, adaptive pooling.

Model B (Pretrained): torchaudio bundle (wav2vec2/Hubert/WavLM) + attentive temporal pooling; partial unfreezing of last N transformer blocks; discriminative LRs.

Model A Pro: BC-ResNet-8 (depthwise + SE + residual), stronger aug + mixup; low-memory training (channels_last + grad accumulation).

Evaluation: Loss/accuracy curves, confusion matrices, per-class accuracy charts, t-SNE embeddings, Top-1 & Top-3 metrics.

CV: 5-fold runner for Model A with per-fold bars and mean±std summary.

TTA & Ensemble: Time-shift TTA for spectrogram and waveform models, weighted A+B ensembling.

Exports: TorchScript (if supported) or state_dict, plus per-file prediction CSVs.

Results (from the provided runs)

Model A (Spec-CNN, fold-1): Top-1 ≈ 0.45, Top-3 ≈ 0.68

Model A — 5-fold CV: Top-1 0.412 ± 0.033, Top-3 0.673 ± 0.033

Model B Upgraded (fold-1): Top-1 0.6725 (no TTA)

Model A Pro (fold-1): Top-1 up to ~0.525 (low-mem run later at 0.4675)

TTA (fold-1):

A Pro: Top-1 0.463, Top-3 0.677

B Up: Top-1 0.660, Top-3 0.840

Ensemble (A Pro 0.4 / B Up 0.6, TTA): Top-1 0.708, Top-3 0.860

Artifacts:

Per-fold CSV: /kaggle/working/modelA_cv_metrics.csv

Fold-1 predictions (TTA/ensemble): /kaggle/working/preds_fold1_tta_ensemble_withB.csv

Exports in /kaggle/working/exports/ (TorchScript or state_dict)

Notes: Scores vary slightly with seeds, aug strength, and unfreezing depth.

Repository / Notebook Structure

Single Kaggle-style notebook with the following sections (cells):

Bootstrap & dataset probe – reads esc50.csv, maps class labels, inspects audio paths.

Feature pipeline (ALL-IN-ONE) – MelSpec, log-mel, PCEN fallback, SpecAugment, dataset class, fold splitter, previews.

Loaders & preview – safe collate (time padding), train/val/test loaders per fold.

Model A (Spec-CNN) + trainer – training loop, curves, confusion matrix, worst-classes chart.

Model B (Pretrained) + waveform loaders – attentive pooling, frozen backbone baseline.

Unified evaluation – Top-1/Top-3, per-class bars, CM, t-SNE for both models.

5-Fold CV (Model A) – runs all official folds, aggregates mean±std, saves CSV + bars.

Upgrade Model B – partial unfreezing (last N blocks), discriminative LRs, longer training.

Upgrade Model A (BC-ResNet-8) – stronger aug + mixup + cosine schedule; low-mem variant and OOM-recovery utilities.

Extras – TTA (spec + wave), A+B ensemble, CSV saves, model exports.


Tips & Tweaks

Model B: tune unfreeze_last_n (e.g., 4–8); keep head LR > backbone LR.

TTA: narrow shifts (e.g., [-0.1, -0.05, 0, 0.05, 0.1]) can help stability.

Model A Pro: try width=48 and epochs=18–20 if VRAM allows; otherwise keep low-mem + accumulation.

Ensemble: sweep weights on val logits to pick wA/wB, then apply to test.



Acknowledgments

Dataset: ESC-50 (use per the dataset’s license/terms).

Pretrained audio backbones: torchaudio.pipelines.
