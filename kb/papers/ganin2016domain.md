---
id: ganin2016domain
title: "Domain-Adversarial Training of Neural Networks"
authors: [Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, François Laviolette, Mario Marchand, Victor Lempitsky]
year: 2016
venue: JMLR 2016 journal article (read as arXiv v4)
url: https://jmlr.org/papers/v17/15-239.html
arxiv: 1505.07818
doi:
pdf_sha256: 39349d71ca79d44b9cfb24be39ceb5802d58f4e004287655b7df49337938bb59
tags: [domain-adaptation, unsupervised-domain-adaptation, adversarial-training, gradient-reversal, representation-learning, domain-invariance, theory, sim-to-real]
status: read
read: 2026-11-21
---

# Domain-Adversarial Training of Neural Networks

<!-- Read from arXiv v4 (2016-05-26), 35 PDF pages (printed numbers match PDF pages). Read in
full: theory and method (Sec. 1-4, pp. 1-13), toy, PAD and deep image experiments (Sec. 5.1.1-5.1.2,
5.1.5, 5.2, pp. 13-26), conclusion (pp. 30-31). Sentiment (Sec. 5.1.3-5.1.4, pp. 16-18) and person
re-id (Sec. 5.3, pp. 26-30) were read but only headline results are recorded, per the brief.
References (pp. 31-35) skimmed. Values read off figures are marked. -->

## TL;DR

DANN learns features that predict source labels but can't tell source from target: a domain
classifier on the features is trained normally while a gradient reversal layer makes the extractor
maximise its loss; motivated by the Ben-David H-divergence bound; strong on digit/sign/Office DA.

## Problem

Unsupervised domain adaptation: labeled samples from a source distribution D_S and unlabeled
samples from a target D_T (same label space, shifted inputs, e.g. synthetic vs real images or
reviews of different products); the goal is a classifier with low target risk (Sec. 1, p. 2;
Sec. 3, p. 5). Earlier DA methods mostly worked on fixed features; the authors want adaptation
inside representation learning, so features are both discriminative and domain-invariant (p. 2).

## Method

- Theory (Sec. 3, pp. 5-7). The H-divergence d_H(D_S, D_T) is twice the largest gap, over
  classifiers η in H, between how often η says "1" on source and on target inputs (Def. 1, p. 6);
  for a symmetric H its empirical version is 2(1 - min_η [error of η separating the two samples])
  (Eq. 1, p. 6). The Proxy A-distance PAD = 2(1 - 2ε), ε the test error of a domain classifier
  (linear SVM or MLP) trained on the pooled samples (Eq. 2-3, p. 6). Theorem 2 (from Ben-David
  et al. 2006, p. 7) bounds target risk by source risk + d̂_H(S, T) + complexity terms + β, with
  β ≥ inf_η* [R_S(η*) + R_T(η*)] the joint risk of the best single classifier.
- Shallow DANN (Sec. 4.1, pp. 7-10): one hidden layer G_f, a softmax label predictor G_y and a
  logistic domain regressor G_d on the hidden layer (Eq. 4-7). The regulariser is the domain
  regressor's best (negated) loss over source and target (Eq. 8), a surrogate for d̂_H; training
  seeks a saddle point of E = source label loss - λ · domain loss (Eq. 9) by SGD (Alg. 1, p. 10).
- Deep DANN (Sec. 4.2, pp. 11-13, Fig. 1 p. 12): feature extractor G_f(θ_f), label predictor
  G_y(θ_y), domain classifier G_d(θ_d). Saddle point of
  E(θ_f, θ_y, θ_d) = mean source L_y - λ · (mean source L_d + mean target L_d) (Eq. 10-12);
  updates θ_f ← θ_f - μ(∂L_y/∂θ_f - λ ∂L_d/∂θ_f), θ_y ← θ_y - μ ∂L_y/∂θ_y,
  θ_d ← θ_d - μλ ∂L_d/∂θ_d (Eq. 13-15). Implemented with a gradient reversal layer (GRL)
  between G_f and G_d: R(x) = x forward, dR/dx = -I backward (Eq. 16-17, p. 13), with λ weighting
  L_d in the pseudo-objective (Eq. 18, p. 13); Fig. 1 says it multiplies the gradient by "a
  certain negative constant" (p. 12); for the theory to apply the label
  predictor's class must be contained in the domain classifier's (H_y ⊆ H_d, p. 11).
- Reverse validation for picking λ and other hyperparameters without target labels (Sec. 5.1.2,
  p. 16): train on source + unlabeled target, self-label the target, train a reverse classifier
  on that, and score it on a held-out source split.
- Deep image setup (Sec. 5.2.2, pp. 20-22, Fig. 4 p. 21): small CNNs for digits and signs,
  ImageNet AlexNet fine-tuned for Office with the domain classifier on a 256-d bottleneck after
  fc7 (Tzeng et al.'s architecture, its domain mean-based regularisation replaced by the domain
  classifier, p. 26). Domain head x→1024→1024→2 (x→100→2 for MNIST), choices the authors call
  "arbitrary" (p. 20). SGD, momentum 0.9, μ_p = 0.01/(1 + 10p)^0.75 with p the training progress
  from 0 to 1; adaptation factor λ_p = 2/(1 + exp(-10p)) - 1 ramped from 0 to 1, applied only to
  the feature extractor's reversed gradient (domain classifier uses λ = 1); batch 128, half source
  and half target.
- Baselines (Sec. 5.2.1, p. 20): source-only (lower bound), train-on-target (upper bound) and
  Subspace Alignment (SA, Fernando et al. 2013), whose one hyperparameter was tuned on target
  test accuracy, favouring the baseline. All four methods' last layers were retrained with a linear
  SVM on extracted features.

## Assumptions

- A1: Unsupervised domain adaptation: a labeled source sample drawn i.i.d. from D_S and an unlabeled target sample drawn i.i.d. from the target's input marginal; no target labels are used for training (Sec. 3, p. 5)
- A2: Source and target share one label space Y of L classes, i.e. the same task (Sec. 3, p. 5)
- A3: There are exactly two domains, source and target, separated by a binary domain classifier; more than two domains or multiple sources are not treated (Sec. 3, p. 5, Eq. 2, p. 6, Sec. 4.2, p. 11)
- A4: Both domains live in one input space X, so the same feature extractor takes either domain's inputs; only the distribution over X shifts (Sec. 3, p. 5)
- A5: Theorem 2's conditions: a hypothesis class of finite VC dimension of binary classifiers (multiclass needs a class symmetric under label permutations), with i.i.d. source and target samples (Sec. 3.1, p. 6, Sec. 3.3, p. 7)
- A6: Some single classifier in the class has low risk on both domains (low β); otherwise the bound allows no low target risk (Sec. 3.3, p. 7)
- A7: For Theorem 2 to apply to DANN, the domain classifier's hypothesis class contains the label predictor's (H_y ⊆ H_d) (Sec. 4.2, p. 11)
- A8: Making the marginal feature distributions indistinguishable does not pair source and target inputs of different classes, i.e. aligning marginals also aligns the class-conditional structure (reader's inference, Sec. 3.3, p. 7)
- A9: The model is a feed-forward network trained by backpropagation and SGD, with a domain classifier attached to the feature layer through a gradient reversal layer (identity forward, gradient multiplied by -λ backward) (Sec. 4.2, pp. 11-13, Eq. 16-18)
- A10: Deep training reaches a usable saddle point with plain SGD when the feature extractor's adaptation factor is ramped from 0 to 1 over training, so the domain signal is weak early, and every batch mixes source and target samples (Sec. 5.2.2, pp. 21-22)
- A11: Unlabeled target data are abundant and available during training (the method's stated premise; fully transductive on Office) (Sec. 5.2.4, p. 26)
- A12: Shallow experiments choose hyperparameters by reverse validation, which needs only source labels and self-labeled target data (Sec. 5.1.2, p. 16, Sec. 5.1.3, p. 17)
- A13: The deep digit and sign comparison is set up without target labels for DANN (no hyperparameter search) but tunes the SA baseline on target test accuracy, and retrains every method's last layer with a linear SVM (Sec. 5.2.1-5.2.2, pp. 20-21)
- A14: Adaptation needs source-trained features that do not already separate the domains strongly, so a narrow source (MNIST) toward a more diverse target (SVHN) fails (reader's inference, Sec. 5.2.4, p. 25)
- A15: The paired domains differ in appearance (synthetic vs real, backgrounds, cameras, products) while the class of an input stays recognisable in both, so one labeler can serve both (reader's inference, Sec. 5.2.4, p. 23, Fig. 6, p. 24)
- A16: Sentiment data are Amazon reviews of 4 product types encoded as unigram and bigram bag-of-words vectors with binary labels (Sec. 5.1.3, p. 16)
- A17: PAD is measured with linear SVM domain classifiers trained on half of the pooled representations, and grows with representation size (Sec. 5.1.5, p. 19)
- A18: Office starts from an ImageNet-pretrained AlexNet fine-tuned in Tzeng et al.'s architecture, on small domains, and is compared against previously published numbers (Sec. 5.2.2, p. 20, Sec. 5.2.4, p. 26, Table 3, p. 24)
- A19: Digit and sign feature extractors are small CNNs of two or three conv layers taken from prior work (Sec. 5.2.2, p. 20, Fig. 4, p. 21)
- A20: The synthetic sources were built to resemble the targets: Syn Numbers with variation chosen manually to simulate SVHN, and Syn Signs simulating various imaging conditions (Sec. 5.2.4, p. 23, p. 25)
- A21: Re-id treats each camera-network dataset as a domain and learns descriptors with a siamese network on source pairs; target images are used without probe-gallery correspondences (Sec. 5.3.1-5.3.2, pp. 27-28)
- A22: Every experiment is supervised prediction from fixed, i.i.d. datasets (classification or descriptors), not sequential decision-making with rewards or data that change during training (reader's inference, Sec. 3, p. 5, Sec. 5, p. 13)

## Claims

- C1: Theorem 2 (from Ben-David et al. 2006): with probability 1 - δ, for every η in a class of VC dimension d, target risk ≤ source empirical risk + d̂_H(S, T) + β + terms of order sqrt((d log n + log(1/δ))/n); target risk can be low only when β is low, i.e. when some classifier does well on both domains [A1, A2, A3, A4, A5, A6] (Sec. 3.3, p. 7).
- C2: The bound suggests choosing a representation that trades off low source risk against low empirical H-divergence; DANN's domain-classifier loss is a surrogate for d̂_H [A1, A2, A3, A4, A5, A6, A7] (Sec. 3.3, p. 7, Sec. 4.1, pp. 8-9, Eq. 8).
- C3: Deep DANN is standard backpropagation/SGD plus a gradient reversal layer, and can be added to almost any feed-forward network trained by backpropagation [A9] (Sec. 4.2, pp. 12-13, Eq. 16-18, Sec. 6, p. 31).
- C4: Without the reversal (i.e. minimising the domain loss through the features too), SGD would make features dissimilar across domains to help the domain classifier [A9] (Sec. 4.2, p. 12).
- C5: On a two-moons toy (target rotated 35°, 300 unlabeled target points, 15 hidden units, λ = 6), shallow DANN classifies source and target points correctly while a standard network does not fully adapt; in a PCA of the hidden layer, DANN's target points mix with the source while the standard network leaves target-only clusters, and DANN's domain regressor fails to separate the domains [A1, A2, A3, A4, A9, A22] (Fig. 2, Sec. 5.1.1, pp. 13-15).
- C6: Headline sentiment result (Amazon reviews, 12 pairs of 4 domains, shallow models): DANN has the best accuracy on 9 of 12 pairs on raw features (one a tie with NN, E→K .854; SVM wins the other 3), and a pairwise Poisson binomial test gives it a probability of .87 of outperforming a standard NN and .83 an SVM; on mSDA features, .92 and .88 [A1, A2, A3, A4, A9, A11, A12, A16, A22] (Table 1, p. 17, Sec. 5.1.3-5.1.4, pp. 16-18).
- C7: DANN's learned representations have a lower PAD than the raw input on every pair, and a lower PAD than a standard NN's hidden layer on most pairs (figure reading: most points below the diagonal, a few near it) [A1, A2, A3, A4, A9, A11, A12, A16, A17, A22] (Fig. 3a, Fig. 3b, Sec. 5.1.5, pp. 18-19).
- C8: mSDA representations have a higher PAD than the raw data, which the authors note "seems to contradict" the theory, while DANN on top of mSDA lowers the PAD [A1, A2, A3, A4, A9, A11, A12, A16, A17, A22] (Fig. 3c, Sec. 5.1.5, p. 19).
- C9: MNIST→MNIST-M accuracy: source-only .5225, SA .5690, DANN .7666 (52.9% of the gap to train-on-target .9596) [A1, A2, A3, A4, A8, A9, A10, A11, A13, A15, A19, A22] (Table 2, Sec. 5.2.4, pp. 23-24).
- C10: Syn Numbers→SVHN: source-only .8674, SA .8644 (a drop), DANN .9109 (79.7% of the gap to .9220) [A1, A2, A3, A4, A8, A9, A10, A11, A13, A15, A19, A20, A22] (Table 2, Sec. 5.2.4, pp. 23-24).
- C11: SVHN→MNIST: source-only .5490, SA .5932, DANN .7385 (42.6% of the gap to .9942); the reverse direction MNIST→SVHN is a failure case in which DANN does not improve on the non-adapted ≈0.25 accuracy [A1, A2, A3, A4, A8, A9, A10, A11, A13, A14, A15, A19, A22] (Table 2, pp. 24-25, Sec. 5.2.4).
- C12: The authors attribute the MNIST↔SVHN asymmetry to features: a CNN trained on MNIST separates the two domains strongly, an SVHN-trained one mixes them much more [A14, A15, A19] (Sec. 5.2.4, p. 25).
- C13: Syn Signs→GTSRB (43 classes): source-only .7900, SA .8165, DANN .8865 (46.4% of the gap to .9980) [A1, A2, A3, A4, A8, A9, A10, A11, A13, A15, A19, A20, A22] (Table 2, Sec. 5.2.4, pp. 24-25).
- C14: Semi-supervised signs (430 labeled real images plus ≈31,000 unlabeled): Syn + Real Adapted reaches the lowest validation error (figure reading: about 0.065 vs about 0.075 for Syn + Real, 0.12 for Syn Adapted, about 0.19 Syn and 0.20 Real at 5·10^5 batches); the authors leave thorough verification to future work [A2, A3, A4, A9, A10, A11, A19, A20, A22] (Fig. 7, Sec. 5.2.4, p. 25).
- C15: On Office (fully transductive), DANN gets .730 A→W, .964 D→W and .992 W→D, above the listed GFK, SA, DLID, DDC and DAN numbers (SA was not evaluated fully transductively, and some baseline numbers are reproduced from Long and Wang 2015); over source-only (.642, .961, .978) the gain is large on A→W and small on the other two [A1, A2, A3, A4, A8, A9, A10, A11, A15, A18, A22] (Table 3, p. 24, Sec. 5.2.4, p. 26).
- C16: On Office, target accuracy slightly degrades late in training while source accuracy keeps rising; turning off the domain classifier makes this far more apparent, so the authors conclude it acts as a regulariser [A1, A2, A3, A4, A9, A10, A11, A18, A22] (Sec. 5.2.4, p. 26).
- C17: t-SNE of features shows adaptation brings source and target distributions much closer (MNIST→MNIST-M, Syn Numbers→SVHN), and the authors observe a strong correspondence between this overlap and adaptation success [A1, A2, A3, A4, A9, A10, A11, A15, A19, A22] (Fig. 5, p. 22, Sec. 5.2.3, pp. 22-23).
- C18: Headline re-id result: replacing the label predictor with a siamese descriptor predictor, domain-adversarial training improves CMC curves across eight dataset pairs per the caption, considerably for pairs involving PRID; the curves are close on several pairs, touch or cross near ranks 30-50 on Whole CUHK→PRID and CUHK/p1→PRID, and the adapted curve falls below the baseline at higher ranks on CUHK/p1→VIPeR (figure reading) [A3, A4, A9, A11, A21, A22] (Fig. 9, Sec. 5.3.3, pp. 28-29).
- C19: The paper presents re-id as evidence that the approach applies beyond classification, to descriptor learning [A9, A21] (Sec. 5.3.3, p. 28, Sec. 6, p. 31).

## Implementation notes

- GRL: forward R(x) = x, backward dR/dx = -I (source: Eq. 16-17, p. 13); λ weights L_d (source: Eq. 18, p. 13); in the deep experiments this amounts to scaling the reversed gradient into G_f by -λ_p while the domain classifier uses λ = 1 (source: Sec. 5.2.2, p. 21, footnote 6, p. 22).
- λ schedule λ_p = 2/(1 + exp(-γp)) - 1, γ = 10, not tuned; the authors say it makes the domain classifier less sensitive to noisy signal early in training (source: Sec. 5.2.2, p. 21).
- lr μ_p = μ0/(1 + αp)^β, μ0 = 0.01, α = 10, β = 0.75, momentum 0.9, a schedule tuned for convergence and low source error; no annealing on MNIST↔SVHN to avoid a poor local minimum (source: Sec. 5.2.2, p. 21; Sec. 5.2.4, p. 23).
- Losses: logistic (softmax) loss for labels, binomial cross-entropy for the domain; SVHN model uses dropout and an ℓ2-norm constraint (source: Sec. 5.2.2, p. 21).
- Inputs mean-subtracted; batches of 128, half source and half target (source: Sec. 5.2.2, pp. 21-22).
- Sentiment: 5000-d bag-of-words, hidden 50 or 100, lr 1e-3, λ among 9 values from 1e-2 to 1 picked by reverse validation (source: Sec. 5.1.3, p. 17).
- Re-id: DML siamese net (Yi et al. 2014), domain head x→500→1, fixed lr 0.001, momentum 0.9, dropout 0.5, 128 source + 128 target per batch, 20k or 50k iterations (source: Sec. 5.3.2-5.3.3, p. 28).
- Code: shallow DANN at graal.ift.ulaval.ca/dann (p. 10); Caffe GRL at sites.skoltech.ru/compvision/projects/grl/ (p. 13) (source: the paper's footnotes; not checked).
- The paper notes that GANs' saturating-sigmoid fix is "directly applicable" (source: Sec. 2, p. 4); it does not report using it.
- Number of seeds and variance of the image results are not reported (source: Table 2-3, p. 24).
- Survey carry-over (reader's note): for N games, a multi-class domain classifier over game ids behind one GRL is the natural extension; the paper only treats two domains (source: assumption).
- Survey carry-over (reader's note): Theorem 2's β term means invariance helps only if one labeler or policy is good on all games; matching marginals can align the wrong states (source: assumption, from Theorem 2, p. 7).

## Limitations / open questions

- Two domains, one shared label space; target labels never used but source labels required
  (Sec. 3, p. 5). Multi-source is not treated.
- Failure case MNIST→SVHN: invariance from a narrow source domain does not help (Sec. 5.2.4,
  p. 25). PAD on mSDA contradicts the bound's intuition (Fig. 3c, p. 19).
- Hyperparameters: domain-head sizes "arbitrary", λ schedule untuned; SA tuned on target test
  data (Sec. 5.2.1-5.2.2, pp. 20-21). No variance or seed counts for image results.
- Theorem 2: target risk can be low only when β is low, i.e. when some classifier has low risk
  on both domains (Sec. 3.3, p. 7). Open question (reader's inference, not in the paper): β
  involves target risk, so it can't be checked without target labels.
- Open question (survey lens): DANN aligns marginal feature distributions, not states with the
  same meaning; across many games with different state distributions, does adversarial alignment
  collapse semantically different states? Conditional or class-aware alignment may be needed.
- Open question: the adversarial objective is a saddle point; the paper reports no instability
  beyond the λ ramp and the no-annealing MNIST↔SVHN case; how does it behave with many domains?

## Related

- related [[gupta2017invariant]]: learns a shared feature space across two robots from paired states, naming an adversarial objective like this one as an alternative to pairs.
- related [[liu2018context]]: aligns viewpoints by translating frames between contexts, using time-aligned demos instead of an adversary.
- related [[zhang2020correspondence]]: unpaired cross-domain maps for control from GAN losses (adversarial domain discriminators) plus dynamics cycle-consistency, mapping one domain into the other rather than learning shared invariant features.
- related [[serrano2024crossdomain]]: survey of cross-domain RL transfer, where adversarial feature alignment is one family.
- related [[liu2025prototypes]]: cross-domain representation pretraining for RL, a different route to features shared across domains.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
