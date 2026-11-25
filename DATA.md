# Data sources and obligations

No dataset is bundled with this repository. `tools/download_data.py` fetches
files from ModelScope into `data/raw/` (git-ignored) after printing the license
notices. You must review and comply with each source's terms.

## Datasets used by the default pipeline

| Key | Dataset | License | Size | Notes |
|---|---|---|---|---|
| `lccc` | [OmniData/LCCC](https://www.modelscope.cn/datasets/OmniData/LCCC) | **LCCC Custom (research use)** | ~357 MB zip | Human Weibo conversations; the base of the corpus |
| `slim` | [AI-ModelScope/Slim-LCCC-zh](https://www.modelscope.cn/datasets/AI-ModelScope/Slim-LCCC-zh) | Apache-2.0 | ~2.7 MB | Cleaned subset derived from LCCC |
| `muice` | [Moemuu/Muice-Dataset](https://www.modelscope.cn/datasets/Moemuu/Muice-Dataset) | **CC-BY-NC-4.0 (non-commercial)** | ~0.9 MB | Role-play chat; opt-in via `--include-nc` |

```bash
python tools/download_data.py --accept-license            # LCCC + Slim-LCCC
python tools/download_data.py --accept-license --include-nc  # also Muice (NC)
```

The script verifies a minimum size and prints the SHA-256 of each file. It does
not bypass access controls; it uses the public ModelScope dataset API.

## Obligations

1. **Do not redistribute the downloaded files.** They are not covered by this
   repository's AGPL license. Share the download instructions instead.
2. **LCCC is a research-use corpus.** Review the CDial-GPT / LCCC terms before
   using it, and before publishing a model trained on it. Whether a trained
   model counts as a derivative work is a legal question for you and your
   counsel; this project takes no position and grants no rights to the data.
3. **Personal information.** LCCC is collected from public social media. It can
   contain personal data and offensive language. Do not build services that
   expose or profile individuals from this data.
4. **Muice is non-commercial (CC-BY-NC-4.0).** Mixing it in makes the resulting
   corpus (and arguably the model) subject to non-commercial restrictions. It is
   disabled by default for this reason.
5. **Your own corpus.** If you use `--custom`, you are responsible for its
   rights, consent, and privacy compliance.

## Citation

If you use the default corpus, cite the LCCC paper:

```bibtex
@inproceedings{wang2020lccc,
  title     = {A Large-scale Chinese Short-Text Conversation Dataset},
  author    = {Wang, Yida and Ke, Pei and Zheng, Yinhe and Huang, Kaili and Jiang, Yong and Zhu, Xiaoyan and Huang, Minlie},
  booktitle = {NLPCC},
  year      = {2020},
  note      = {arXiv:2008.03946}
}
```

## Why no data is included

- Copyright: datasets carry their own licenses, some non-commercial.
- Privacy: social-media text can contain personal information.
- Size: the raw archive is hundreds of megabytes.

The repository stays small and redistributable; the pipeline stays reproducible
because every source is a public, versioned download.
