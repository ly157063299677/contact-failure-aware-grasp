# Source references

The method and experiment settings in this repository were transcribed from the supplied manuscript
and supplementary DOCX files, read on 2026-10-16. The supplementary DOCX was newer than its PDF and
was used for the detailed settings. Their SHA-256 identifiers are retained in `source_manifest.json`.
The documents themselves are not redistributed in the repository.
The final source comparison found unchanged manuscript body text and an updated Table S3
specifying the outcome-only PointNet–Finger MLP for the Top-k baseline; this update is
included in the code and instructions.

Official upstream sources checked for integration context:

- [Dexonomy repository](https://github.com/JYChen18/Dexonomy)
- [GraspADMM integration guide](https://github.com/JYChen18/Dexonomy/blob/main/getting_started/3_ADMM.md)
- [GraspADMM paper](https://arxiv.org/abs/2603.13832)
- [PyTorch 2.7 BCEWithLogitsLoss](https://docs.pytorch.org/docs/2.7/generated/torch.nn.BCEWithLogitsLoss.html)
- [MuJoCo 3.3.3 Python bindings](https://mujoco.readthedocs.io/en/3.3.3/python.html)

The native ADMM optimization, assets and original experiment state are external dependencies.
No claim is made that the current upstream main branch is the exact revision used in the manuscript.
Record the actual upstream commit and asset checksums when connecting a physical backend.
