| Method | nDCG@10 | MRR@10 | Recall@10 | Recall@100 |
|---|---:|---:|---:|---:|
| BM25 | 0.648 | 0.617 | 0.764 | 0.880 |
| Dense: nomic-embed-text (no task prefix) | 0.687 | 0.650 | 0.824 | 0.950 |
| Hybrid RRF: BM25 + nomic-embed-text (no task prefix) | 0.712 | 0.678 | 0.846 | 0.960 |
| Dense: nomic-embed-text (task prefixes) | 0.696 | 0.657 | 0.839 | 0.940 |
| Hybrid RRF: BM25 + nomic-embed-text (task prefixes) | 0.706 | 0.669 | 0.845 | 0.955 |
| Dense: mxbai-embed-large (no task prefix) | 0.726 | 0.690 | 0.860 | 0.968 |
| Hybrid RRF: BM25 + mxbai-embed-large (no task prefix) | 0.731 | 0.698 | 0.861 | 0.980 |
| Dense: mxbai-embed-large (task prefixes) | 0.730 | 0.692 | 0.872 | 0.975 |
| Hybrid RRF: BM25 + mxbai-embed-large (task prefixes) | 0.737 | 0.705 | 0.864 | 0.977 |
