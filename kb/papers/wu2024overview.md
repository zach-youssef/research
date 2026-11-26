---
id: wu2024overview
title: Towards audio language modeling - an overview
authors: [Haibin Wu, Xuanjun Chen, Yi-Cheng Lin, Kai-wei Chang, Ho-Lam Chung, Alexander H. Liu, Hung-yi Lee]
year: 2024
venue: arXiv preprint
url: https://arxiv.org/abs/2402.13236
arxiv: 2402.13236
doi:
pdf_sha256: ac48068ad3d5452cc2a429a17c7c96e7e9a7f2ad580559f72ee1ac2b618f6cc1
tags: [survey, audio-tokenizer, neural-codec, rvq, audio-language-model, speech, music, general-audio]
status: read
read: 2026-11-26
---

# Towards audio language modeling - an overview

## TL;DR

A 4-page overview (v1, Feb 2024) of neural audio codecs used as audio tokenizers: compares six open
codecs (15 released variants) by method, discriminators and training data, then 11 codec-based
language models by task and codec. Descriptive only: no benchmark numbers, no non-speech evaluation.

## Problem

Codecs built for transmission were found to work as tokenizers for audio LMs, and many codecs and
codec-based LMs appeared in about three years without a review comparing them (Sec. I, p. 1).

## Method

A survey, no experiments. Sec. II (pp. 1-3) covers the codecs SpeechTokenizer, AcademiCodec,
AudioDec, DAC, EnCodec and FunCodec (plus SoundStream and SoundStorm in the overview text): a
method overview, a comparison of tricks (Table II, p. 3) and of codebook count, sample rate, bit
rate and training data (Table I, p. 3). Sec. III (pp. 3-4) describes the codec-LM pipeline (Fig. 2,
p. 3: context codes plus codec codes into an LM, decoder back to audio) and 11 codec-based LMs
(Table III, p. 4). Sec. IV (p. 4) is a 3-line conclusion. The paper has no section on open
problems and no results tables beyond these comparisons.

## Assumptions

- A1: The paper reports no measurements of its own; everything is a description of the cited papers' designs and claims (Sec. II-III, pp. 1-4).
- A2: Table I's caption defines "BPS" as bit rate in bits per second, but the printed values (1.5 to 24) match the codecs' kbps rates, so the caption's unit is taken to be wrong and the values read as kbps (reader's inference, Table I, p. 3).
- A3: The comparison is of open-source codecs as of Feb 2024 (v1) (Sec. II, p. 1).

## Claims

- C1: Neural codecs, first built to compress audio for transmission, were recently found to be suitable tokenizers that turn continuous audio into discrete codes for audio LMs [A3] (p. 1).
- C2: Of the six compared open codecs (15 released variants), the text says SpeechTokenizer, AudioDec and FunCodec use only English speech (though Table I lists FunCodec variants F5-F6 as trained on 25k hours of English and Mandarin data), AcademiCodec on bilingual speech (AISHELL, LibriTTS, VCTK), while DAC and EnCodec include speech, music and general audio [A3] (Sec. II-C, Table I, p. 3).
- C3: Table I lists 16 kHz to 44.1 kHz sample rates; DAC's variants D1-D3 (16, 24 and 44.1 kHz; 12, 32 and 9 codebooks; BPS-column values 6, 24 and 8) share one training-data cell (Common Voice, DAPS, VCTK, MUSDB, Jamendo, AudioSet), so the table does not attribute data to single variants; EnCodec variants E1-E5 go from 2 to 32 codebooks at 1.5-24 on the BPS column [A2, A3] (Table I, p. 3).
- C4: DAC is described as keeping high fidelity across general audio, music and speech, and the authors credit mainly the periodic (snake) activation, plus factorised L2-normalised RVQ codes, quantizer dropout and refined losses [A1, A3] (Sec. II-A, p. 2).
- C5: DAC found that MSD plus MPD discriminators alone give blurry audio with artifacts, and added a multi-scale multi-band STFT discriminator to improve phase modelling and cut aliasing [A1] (Sec. II-B, p. 2).
- C6: SpeechTokenizer distils HuBERT layer-9 semantic tokens into the first RVQ layer so content sits there and paralinguistic information in later layers; FunCodec found adding semantic tokens helps quality, most with the residual method; both use K-means initialisation of codebooks for better code use [A1, A3] (Sec. II-B, p. 2, Table II, p. 3).
- C7: AcademiCodec's group-residual VQ (several parallel RVQ groups) is claimed to reconstruct well with few codebooks and so a low bit rate and shorter token sequences for LMs [A1] (Sec. II-A, p. 2).
- C8: EnCodec adds a small transformer for entropy coding of the quantized units, which cuts bandwidth and speeds encoding/decoding [A1] (Sec. II-B, pp. 2-3).
- C9: Codec-based LMs for sound and music use SoundStream (AudioGen, MusicLM, AudioLM) or EnCodec (MusicGen, UniAudio, VALL-E family) codes; AudioGen handles audio continuation from text, MusicGen music generation and stereo generation, UniAudio covers speech, sound, music and singing tasks [A1, A3] (Sec. III-A, Table III, p. 4).
- C10: UniAudio speeds autoregressive prediction with a multi-scale transformer, a large global transformer for first-layer codes and a small local one for the other layers; its codec is a revised EnCodec [A1] (Sec. III-A, p. 4).
- C11: Discrete units from quantising self-supervised speech features (GSLM, pGSLM, dGSLM, TWIST) hold rich acoustic and linguistic content but lack speaker and paralinguistic information, per the survey's citations [A1] (Sec. III-A, p. 4).
- C12: An ideal codec should keep content while preserving paralinguistic and speaker information, and a universal audio LM should generalise across speech, music and general audio (stated as the aim, with the arms race "still ongoing") [A1] (Sec. I, p. 1).

## Implementation notes

- No code, hyperparameters or evaluation protocol are given; the paper only points to the cited codec papers (source: Sec. II, p. 1).
- Token rates are not stated in frames per second; derive them from each codec's own paper (source: assumption).

## Limitations / open questions

- The paper states no limitations and no open problems; its conclusion only hopes the comparison inspires future work (Sec. IV, p. 4).
- Open question: no reconstruction or downstream metrics are compared, so which codec holds up on non-speech audio (sound effects, music) cannot be read from this paper (reader's inference).
- Open question: game audio (chiptune, FM synthesis, short effects) is not covered by any training set listed in Table I.

## Related

- related [[he2021masked]]: MAE-style masked audio modelling is an alternative to codec tokens for audio inputs, not discussed here.
- Not on main yet: sibling notes on SoundStream, EnCodec, DAC, AudioLM, SNAC, WavTokenizer, Moshi (branch papers-audio-codecs), and the other two surveys in this batch.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
