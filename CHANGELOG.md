# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-11-17

### Added
- `ckpt-doctor check` for model folders, single `.safetensors` files, `*.safetensors.index.json` files
  and Hugging Face cache repos (by repo id, repo folder or snapshot folder).
- File layer (CD101-CD114): Git LFS pointer, HTML page, empty/tiny file, header limits, JSON problems,
  duplicate keys, unknown dtype, offset holes, truncation with exact missing bytes, trailing bytes.
- Index layer (CD201-CD211): missing index or shards, tensors missing from or extra in shards, wrong or
  duplicated shard, `total_size` mismatch, unsafe paths.
- Config layer (CD301-CD391): layer count, tied embeddings, dtype, layer-number gaps.
- Cache layer (CD401-CD407): `.incomplete` blobs, broken symlinks, dangling refs, tree-listing sizes.
- Hash layer (CD501-CD503): `--hash`, `--manifest`, `--write-manifest` (sha256sum format).
- Fix commands for findings, `--json` (schema in `docs/report.schema.json`), `--sarif`, `--strict`,
  exit codes 0/1/2.
- English and Korean output: `--lang`, `CKPT_DOCTOR_LANG`, `LC_ALL`/`LANG`.
- `ckpt-doctor explain CODE` and `ckpt-doctor sample DIR [--check]` (1 clean + 13 broken synthetic samples).
- Python API: `check_path()`.

### Fixed
- CD302 no longer fires for models whose output head is `embed_out.weight`, `output.weight` or `output_layer.weight`.
- Fix commands for single-file and index inputs use the parent folder for `--local-dir`.
- CD110/CD111 messages say "data region" instead of "file".
- Layer count check understands bare `h.N.*` (canonical GPT-2) names.
