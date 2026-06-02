# Manuscript Text for the GCN-CfC Over-Smoothing Analysis

## Methods: Over-Smoothing Diagnostic

To examine whether the graph encoder suffered from over-smoothing, we quantified the similarity of atom-level hidden states within each molecular graph. For each test molecule, the final atom embeddings before graph-level pooling were extracted from either the Plain-GCN baseline or the GCN encoder used in GCN-CfC. Molecules containing fewer than two atoms were excluded from this diagnostic. For molecule \(i\) with \(n_i\) atoms and atom embeddings \(z_{i1}, ..., z_{in_i}\), the within-molecule node similarity was defined as:

\[
S_i = \frac{2}{n_i(n_i - 1)} \sum_{p < q} \cos(z_{ip}, z_{iq})
\]

The complementary node dispersion score was defined as \(D_i = 1 - S_i\). Higher \(S_i\) indicates stronger homogenization of atom-level representations, whereas higher \(D_i\) indicates better retention of atom-level heterogeneity. Plain-GCN models with 3, 5, and 7 graph convolution layers were trained using the same train/validation/test split and atom descriptors as the main model. The GCN-CfC model was evaluated at the graph-encoder output immediately before global pooling and is therefore shown as an encoder-depth reference rather than as a CfC-time-dependent node curve. This distinction is important because the CfC module operates after graph-level pooling and does not perform additional atom-level message passing.

## Results: Diagnostic Interpretation

The over-smoothing diagnostic directly tests whether deeper graph propagation causes atom-level hidden states within the same molecule to collapse toward similar vectors. If the Plain-GCN curves show a monotonic increase in \(S_i\) from 3 to 7 layers, this supports the expected over-smoothing behavior of repeated graph convolution. In contrast, if the GCN-CfC encoder retains a lower \(S_i\) and higher \(D_i\), the result supports the design rationale that GCN-CfC limits node-level message passing to preserve local molecular detail, while transferring additional nonlinear refinement to the graph-level CfC module.

In the PAD4 test set, the Plain-GCN baseline showed a clear depth-dependent increase in within-molecule node similarity: \(S_i = 0.455\) for 3 layers, \(0.558\) for 5 layers, and \(0.644\) for 7 layers. The corresponding node dispersion score decreased from \(0.545\) to \(0.442\) and \(0.356\), respectively. This trend indicates progressive atom-level representation homogenization as graph propagation depth increases. In the controlled GCN-CfC setting, the graph encoder was kept at 3 layers before the CfC refinement step, giving \(S_i = 0.455\) and \(D_i = 0.545\) at the atom-embedding level. Thus, GCN-CfC should be interpreted as preserving the shallow encoder's node-level heterogeneity and then adding CfC-based graph-level refinement, not as producing a separate post-CfC node-embedding curve.

## Methods: Analogue-Pair Latent-Space Analysis

To evaluate whether the coupled model improves discrimination among structurally similar molecules, analogue pairs were selected from the PAD4 test set using Morgan fingerprints. Candidate pairs were required to have opposite activity labels, Tanimoto similarity at least 0.65, and no more than 25% relative difference in atom count. If fewer than the target number of pairs were found, the Tanimoto and atom-count thresholds were relaxed in predefined steps, and the final thresholds were recorded. Each molecule was additionally annotated with a reproducible SMARTS-based electrophilic/covalent-warhead flag to support interpretation of active/inactive or warhead-containing analogue pairs. For each selected pair, graph-level representations were extracted before CfC refinement and, when TensorFlow/CfC weights were available, after CfC refinement. The pairwise Euclidean distance in each latent space, the ratio of post-CfC to pre-CfC pair distance, the label silhouette score, and the nearest opposite-label distance were reported.

## Results: Analogue-Pair Interpretation

A larger distance between opposite-label analogue pairs after CfC refinement, together with an increased silhouette score or nearest opposite-label distance, would indicate that the CfC module improves graph-level discrimination among molecules that are topologically similar but functionally different. This analysis is complementary to the node-level over-smoothing diagnostic: the node-level analysis evaluates whether local atom representations remain heterogeneous, whereas the analogue-pair analysis evaluates whether graph-level refinement improves separability in the downstream decision space.

Across three CfC random seeds, the final GCN-CfC classifier achieved test accuracy \(0.990\), F1 \(0.980\), and ROC-AUC \(0.999\) on PAD4_007_08. More importantly for the mechanistic analysis, graph-level latent separation improved after CfC refinement: the mean nearest opposite-label distance increased from \(2.515\) before CfC to \(3.702\) after CfC, and the label silhouette score increased from \(0.443\) to \(0.748\). For the selected activity-discordant analogue pairs, 9 of 10 pairs showed increased Euclidean distance after CfC refinement in the seed-7 run, with a mean post/pre distance ratio of \(1.57 \pm 0.43\). These results support the interpretation that CfC improves graph-level discrimination among structurally similar molecules while avoiding additional atom-level message passing.

## Discussion Paragraph

Over-smoothing is a known limitation of deep graph convolutional networks. Repeated graph propagation acts as a form of Laplacian smoothing, progressively reducing differences among node representations and causing atom-level hidden states within a molecular graph to become more homogeneous. In molecular screening, this effect is undesirable because subtle local changes, such as electrophilic motifs, ring substitutions, and local heteroatom arrangements, may carry important information for distinguishing active, inactive, and covalent-confounded chemotypes.

The GCN-CfC architecture addresses this issue through a division of labor rather than by using CfC to directly reverse node-level over-smoothing. The GCN encoder is kept responsible for local and mesoscopic molecular topology and is evaluated before excessive node-level propagation can erase atom-specific detail. After global pooling, the CfC module receives the graph-level representation \(h_G\) and applies an input-dependent closed-form continuous-time transformation. This step refines the global molecular representation through learned nonlinear dynamics, but it does not add further node-level message passing. Thus, additional representational capacity is introduced at the graph level without requiring deeper GCN stacks.

This decoupling explains why GCN-CfC can preserve sensitivity to local molecular structure while still learning a highly discriminative global representation for PAD4 screening. The diagnostic node-similarity analysis tests preservation of atom-level heterogeneity, and the analogue-pair latent-space analysis tests whether structurally similar but activity-discordant molecules become more separable after CfC refinement. Together, these results support the mechanistic interpretation that GCN-CfC mitigates over-smoothing pressure by stopping node-level graph propagation at a controlled depth and shifting subsequent refinement to an efficient continuous-time graph-level module.

## Suggested Figure Captions

**Figure Sx. Node-level over-smoothing diagnostic.** Mean within-molecule atom-embedding cosine similarity was computed on the PAD4 test set for Plain-GCN models with 3, 5, and 7 graph convolution layers and for the 3-layer GCN encoder used in the controlled GCN-CfC model. Higher values indicate stronger homogenization of atom-level representations. The GCN-CfC value is shown only at the encoder depth because CfC operates after graph pooling and does not generate new atom-level states. Error bands indicate 95% confidence intervals across molecules.

**Figure Sy. Latent-space separation of activity-discordant analogue pairs.** Structurally similar PAD4 test-set molecule pairs with opposite labels were selected by Morgan fingerprint Tanimoto similarity and annotated for SMARTS-defined electrophilic/covalent-warhead motifs. Paired molecules are connected by lines in the latent space before CfC refinement and, where CfC weights are available, after CfC refinement. Increased pairwise separation after CfC indicates improved graph-level discrimination among structurally similar but functionally different molecules.

**Figure Sz. Pairwise distance comparison for selected analogue pairs.** Euclidean distances between opposite-label analogue pairs were measured in the GCN graph-level representation space and, when available, in the final GCN-CfC representation space. A distance ratio greater than 1 indicates that CfC refinement increased separation of the selected activity-discordant pair.

## Reference Anchors

- Kipf and Welling, ICLR 2017: graph convolution formulation.
- Li, Han, and Wu, AAAI 2018: GCNs as Laplacian smoothing and over-smoothing.
- Oono and Suzuki, ICLR 2020: loss of expressive power in deep GNNs.
- Zhao and Akoglu, ICLR 2020: PairNorm and over-smoothing diagnostics.
- Hasani/Lechner et al., Nature Machine Intelligence 2022: closed-form continuous-time neural networks.
