# Final Over-Smoothing Mechanism Results (PAD4_007_08)

## Environment

TensorFlow was installed only in the local repository environment `<repo-root>\.venv_cfc_tf`. The existing conda/base environment was not modified.

## Node-Level Over-Smoothing Diagnostic

Plain-GCN within-molecule mean node cosine similarity increased with depth:

- 3 layers: 0.455 (95% CI 0.452-0.458)
- 5 layers: 0.558 (95% CI 0.554-0.562)
- 7 layers: 0.644 (95% CI 0.638-0.650)
- Controlled GCN-CfC encoder, 3-layer GCN before CfC: 0.455 (95% CI 0.452-0.458)

This supports the over-smoothing diagnosis: deeper GCN propagation makes atom-level representations within the same molecule increasingly similar. The controlled GCN-CfC model uses the 3-layer GCN encoder before CfC refinement, so its node-level similarity remains at the 3-layer level rather than following the 5/7-layer Plain-GCN trend. We do not report a post-CfC node-similarity curve because CfC is applied after graph pooling and produces graph-level states, not atom-level states. A 5- or 7-layer GCN-CfC variant would be a different ablation in which the GCN encoder itself is deepened; that test would measure the deepened encoder's over-smoothing, not a direct node-level effect of CfC.

## CfC Graph-Level Refinement

Across three CfC seeds, test performance and latent separation were:

- Accuracy: 0.9901 +/- 0.0000
- F1: 0.9796 +/- 0.0000
- ROC-AUC: 0.9995 +/- 0.0000
- Nearest opposite-label distance: 2.515 before CfC -> 3.702 after CfC
- Silhouette score: 0.443 before CfC -> 0.748 after CfC

## Activity-Discordant Analogue Pairs

For the seed-7 CfC run, 9 of 10 selected structurally similar opposite-label pairs had larger distances after CfC refinement. The mean post/pre distance ratio was 1.569 +/- 0.431.

## Key Output Files

- `final_node_similarity_curve_controlled.png`
- `final_node_similarity_summary_controlled.csv`
- `final_cfc_seed_metrics.csv`
- `final_cfc_seed_metrics_summary.csv`
- `final_pair_distance_seed7.csv`
- `cfc_on_plain_gcn3/latent_pairs_pre_cfc.png`
- `cfc_on_plain_gcn3/latent_pairs_post_cfc.png`
- `cfc_on_plain_gcn3/pair_distance_comparison_with_cfc.png`
