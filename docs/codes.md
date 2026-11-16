# Finding codes

Every finding has a stable code. `ckpt-doctor explain CODE` prints the same text.
The severity shown is the default; a few codes lower it in context.

## Input

| Code | Severity | Meaning | Why it happens | How to fix |
|---|---|---|---|---|
| `CD001` | warning | No .safetensors files found | The folder has no .safetensors files (maybe the model ships .bin or .gguf weights, or you pointed at the wrong folder). | Point ckpt-doctor at the folder that holds the .safetensors files. |

## File layer (safetensors headers)

| Code | Severity | Meaning | Why it happens | How to fix |
|---|---|---|---|---|
| `CD101` | error | Git LFS pointer instead of weights | The repository was cloned without git-lfs, so a ~130 byte text stub replaced the weights. Loaders report this as 'header too large'. | Run git lfs install, then git lfs pull --include for this file. |
| `CD102` | error | HTML page saved as weights | A download returned an error or login page (expired token, gated model, proxy) and it was saved under the weight's name. Loaders report 'header too large'. | Delete the file and download it again with a valid token. |
| `CD103` | error | File too small / empty header | An interrupted upload or download left a stub (for example 16 zero bytes). | Download the file again. |
| `CD104` | error | Header length above the 100 MB limit | The file is not safetensors at all, or its first bytes are damaged. | Check what the file really contains (head -c 200) and download it again. |
| `CD105` | error | Header longer than the file | The file was cut off inside the header. | Download the file again. |
| `CD106` | error | Duplicate key in header JSON | Two entries share one name; the safetensors format forbids it and loaders reject the file. | Re-export the checkpoint with a current safetensors version. |
| `CD107` | error | Header is not valid JSON | The header bytes are damaged or the file was written by a broken tool. | Download the file again, or re-save the checkpoint. |
| `CD108` | error | Unknown tensor dtype | The file uses a dtype this version does not know, or the header is corrupt. | Check the file with the library that wrote it; re-download if unexpected. |
| `CD109` | error | Hole or overlap in the data region | Tensor byte ranges must tile the data region without gaps or overlaps. | Re-save the checkpoint or download the file again. |
| `CD110` | error | Data truncated | The download stopped early or the disk filled up. | Download this file again; only the broken shard needs a re-download. |
| `CD111` | error | Extra bytes after the last tensor | The file was appended to or two files were concatenated. | Download the file again. |
| `CD112` | error | Malformed header entry | Header fields have the wrong type; the file is damaged or was written by a faulty tool. | Download the file again, or re-save the checkpoint. |
| `CD114` | error | Offsets do not match shape x dtype | The header disagrees with itself; the file is damaged. | Download the file again. |

## Index layer (shard index vs headers)

| Code | Severity | Meaning | Why it happens | How to fix |
|---|---|---|---|---|
| `CD201` | error | Shard set without index | A sharded checkpoint needs its .safetensors.index.json, and some shards (or the index) were not downloaded. | Download the missing shards and the index file. |
| `CD202` | error | Index points to a missing shard | The shard was never downloaded, was deleted, or moved. | Download the missing shard. |
| `CD203` | error | Tensor in index but not in shard | The index is stale (written for different shards) or the shard was replaced. | Rebuild the index or download matching index and shards from the same revision. |
| `CD204` | error | Tensor in shard but not in index | The index is stale or incomplete; loaders will never load this tensor. | Rebuild the index or download matching index and shards from the same revision. |
| `CD205` | error | Index points to the wrong shard | Shards were renamed or re-split after the index was written. | Rebuild the index from the shard headers. |
| `CD206` | error | Tensor stored in several shards | Two shards overlap, often from mixing files of two exports. | Keep one consistent set of shards from a single export. |
| `CD207` | error | total_size does not match | total_size is the sum of tensor bytes (not file sizes); a mismatch means the index belongs to other weights. | Use the index that was exported with these shards. |
| `CD208` | warning | Index without metadata.total_size | The index was written by a script that omitted the metadata block. | Add metadata.total_size (sum of tensor bytes) to the index. |
| `CD209` | info | Shard not referenced by an index | A leftover file from another export sits in the folder; loaders ignore it. | Delete it if it is stale. |
| `CD210` | error | Unsafe shard path in index | Shard names must be plain relative file names. | Fix the index entry. |
| `CD211` | error | Index file unusable | The index is not valid JSON or has no weight_map object. | Download the index file again. |

## Config layer (config.json vs tensors)

| Code | Severity | Meaning | Why it happens | How to fix |
|---|---|---|---|---|
| `CD301` | error | Layer count differs from config | config.json and the weights come from different models, or layers were removed without updating config. | Set the layer count in config.json to match the weights, or use the matching weights. |
| `CD302` | error | lm_head missing while embeddings are untied | The save script dropped the output head; loading will report missing keys. | Save lm_head.weight too, or set tie_word_embeddings to true if the head shares the embedding. |
| `CD303` | warning | Config dtype differs from stored dtype | The weights were saved or cast to another precision than config.json says. | Update config.json dtype, or convert the weights. This only changes the load-time default. |
| `CD304` | warning | Gap in layer numbering | Some layers are missing from the checkpoint. | Check that all shards are present and the export was complete. |
| `CD305` | warning | config.json missing | The folder has no model configuration. | Download config.json from the same revision. |
| `CD306` | warning | config.json unreadable | The file is damaged or not a JSON object. | Download config.json again. |
| `CD390` | info | Layer naming not recognised | Only model.layers.N, encoder.layer.N, transformer.h.N, blocks.N and layers.N are understood. | No action needed; ckpt-doctor stays silent rather than guess. |
| `CD391` | info | lm_head present although tied | The save script wrote the shared weight twice. | Harmless; drop the extra tensor to save space. |

## Hugging Face cache layer

| Code | Severity | Meaning | Why it happens | How to fix |
|---|---|---|---|---|
| `CD401` | warning | .incomplete blob in the cache | A download was interrupted; the cache keeps the partial file. | List what hf cache would clean up with hf cache prune --dry-run, or delete the .incomplete file after re-downloading. |
| `CD402` | error | Broken symlink in the snapshot | The blob was deleted or the cache was copied without its blobs folder. | Download the file again into the cache. |
| `CD403` | error | Ref points to a missing snapshot | The snapshot folder was removed or the download never finished. | Download that revision again. |
| `CD404` | error | File from the tree listing is absent | The download is incomplete. | Download the file again. |
| `CD405` | error | File size differs from the tree listing | The file is truncated or was modified after the download. | Download the file again. |
| `CD406` | info | Tree listing format not supported | A newer huggingface_hub wrote a format this version does not know. | Update ckpt-doctor. |
| `CD407` | warning | Tree listing unreadable | The trees json file is damaged. | Delete it; it is only a cache of the repository listing. |

## Hash layer (sha256)

| Code | Severity | Meaning | Why it happens | How to fix |
|---|---|---|---|---|
| `CD501` | error | sha256 differs from the baseline | The bytes changed since the baseline was taken (bit rot, partial overwrite, bad copy) while the header is still valid. | Download the file again, or restore it from a good copy. |
| `CD502` | info | No sha256 baseline for this file | Without a manifest or a cached tree listing there is nothing to compare against. | Create a baseline with --write-manifest while the folder is known to be good. |
| `CD503` | error | Manifest lists a missing file | The file was deleted or moved after the manifest was written. | Restore the file. |
