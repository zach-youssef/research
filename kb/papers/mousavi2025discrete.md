---
id: mousavi2025discrete
title: Discrete Audio Tokens: More Than a Survey!
authors: [Pooneh Mousavi, Gallil Maimon, Adel Moumen, Darius Petermann, Jiatong Shi, Haibin Wu, Haici Yang, Anastasia Kuznetsova, Artem Ploujnikov, Ricard Marxer, Bhuvana Ramabhadran, Benjamin Elizalde, Loren Lugosch, Jinyu Li, Cem Subakan, Phil Woodland, Minje Kim, Hung-yi Lee, Shinji Watanabe, Yossi Adi, Mirco Ravanelli]
year: 2025
venue: TMLR 2025 (survey and benchmark)
url: https://arxiv.org/abs/2506.10274
arxiv: 2506.10274
doi:
pdf_sha256: 355f03ab30a28a7a10b8240948f6ba74b4759fe7026859215e71b1a75969c832
tags: [survey, benchmark, audio-tokenizer, neural-codec, rvq, fsq, music, general-audio, sound-effects, token-rate, audio-language-model]
status: read
read: 2026-11-26
---

# Discrete Audio Tokens: More Than a Survey!

## TL;DR

Taxonomy plus a benchmark of 7 open tokenizers (EnCodec, DAC, WavTokenizer, ...) on speech, music
and AudioSet sound: reconstruction, DASB probing, audio LMs, ablations. Domain match critical
(ablations mixed); fidelity doesn't predict LM quality; bitrate helps reconstruction and music LMs.

## Problem

Surveys of audio tokens cover speech and single aspects; none compares tokenizers across speech, music and
general audio under consistent conditions, and tokenizers are trained on mixed data, rates and domains (Sec. 1, p. 3).

## Method

- Taxonomy (Sec. 2, Fig. 3, Table 1, pp. 4-14): quantizer (k-means, RVQ, GVQ, SVQ, FSQ, multi-scale RVQ,
  cross-scale RVQ, PQ; fixed vs adaptive bitrate), encoder-decoder (CNN, CNN+RNN, Transformer, CNN+T;
  time or time-frequency domain), training (joint vs separate; GAN, diffusion, masked prediction, feature
  matching; auxiliary disentanglement, semantic distillation, supervised semantic), streamability, target domain.
- Benchmark tokenizers (Table 2, p. 15): EnCodec (24 kHz SMA, 32 kHz music, 16 kHz audio), DAC (44/24/16 kHz),
  SpeechTokenizer, Mimi, Discrete WavLM, SQ-Codec, WavTokenizer (24 kHz SMA and speech-only), from released checkpoints
  (Sec. 3, p. 16). S/M/A = speech/music/audio training domains.
- Reconstruction (Sec. 3.1, pp. 16-18): LibriSpeech test-clean (speech), MUSDB (~10 h music, 44.1 kHz), AudioSet test (~55 h);
  signal metrics SDR, SI-SNR, perceptual VISQOL, SingMOS (non-speech) and PESQ, UTMOS, WER, speaker similarity (speech) (Table 3, p. 16).
- Downstream (Sec. 3.2, pp. 18-22): DASB, frozen tokens, lightweight heads, weighted sum over codebooks, 3 seeds; sound event
  classification (ESC-50), audio separation (FUSS), music genre (GTZAN), music separation (MUSDB) (Table 6, p. 19).
- Acoustic LMs (Sec. 3.3, pp. 22-32): 357M Qwen-2.5 speech LMs (SALMon, ZeroSpeech); VALL-E TTS on LibriTTS; AudioCraft 300M
  text-to-audio (about 4,050 h, AudioCaps test, 10 s clips) and text-to-music (FMA 702 h; MusicCaps, FMA test) (pp. 23-31).
- Ablations (Sec. 4, pp. 33-36): DAC backbone in ESPnet-Codec, RVQ vs SVQ vs FSQ, distillation, 16 vs 44.1 kHz, trained on 1k h of speech,
  audio or music or all three (Table 14, p. 34).

## Assumptions

- A1: Tokenizers are the authors' released checkpoints, each trained by its authors on different data, rates and domains, so the benchmark compares these checkpoints and not the designs (Sec. 3, p. 16).
- A2: Non-speech reconstruction is measured on MUSDB (music) and AudioSet clips (YouTube sound), not on game or synthesised audio (Sec. 3.1, p. 16).
- A3: Downstream heads are light and frozen-token, with tokens projected to 1024 dimensions; results are averaged over 3 seeds (Sec. 3.2, p. 19).
- A4: Audio LMs are moderate size (357M speech, ~300M audio and music) with limited compute and data, and generative metrics depend on each tokenizer's vocoder (Sec. 3.3, p. 23, pp. 28-29).
- A5: Audio and music LMs use autoregressive decoding with the MusicGen delay pattern, top-k sampling and classifier-free guidance, not masked or diffusion models (Sec. 3.3.3-3.3.4, pp. 27-30).
- A6: Token rate (tokens per second of audio, defined p. 17) equals frame rate times number of codebooks used (reader's inference, Table 4, p. 17).
- A7: ESC-50 and GTZAN are small training sets for the DASB probes (Sec. 3.2, p. 21).

## Claims

- C1: Of the 52 tokenizers in Table 1, 8 are marked for all of speech, music and audio (DAC, EnCodec, HILCodec, SD-Codec, SemantiCodec, SNAC, WavTokenizer, X-Codec); the table marks most others speech-only, with a few music-only (HARP-Net, SingOMD) [] (Table 1, p. 6).
- C2: Frame rates in Table 2: EnCodec 75 Hz (24 kHz), 50 Hz (32 kHz music, 16 kHz audio); DAC 86/75/50 Hz at 44/24/16 kHz; Mimi 12.5 Hz; WavTokenizer 75 Hz (multi-domain) or 40 Hz (speech); params 14.9M (Enc-SMA-24) to 331.9M (Discrete WavLM); MACs per second of audio 3.4G (WT-S-24) to 147G (DAC-44) [A1] (Table 2, p. 15).
- C3: Reconstruction, general audio (AudioSet): quality drops as bitrate drops; EnCodec 24 kHz gives SDR -1.29/4.28/7.72 dB and VISQOL 3.94/4.25/4.36 at 1.5/6/24 kbps (150/600/2400 tokens/s); DAC has negative SI-SNR (-11.55 to -9.91) yet VISQOL 3.99-4.59; WavTokenizer SMA-24 at 0.98 kbps (75 tokens/s) has SDR -4.55, VISQOL 3.96 [A1, A2, A6] (Table 5, p. 18).
- C4: Reconstruction, music (MUSDB): EnCodec 24k SDR 2.16/7.32/11.04 dB, VISQOL 4.05/4.38/4.50; DAC SDR 1.75/4.82/5.56; WavTokenizer SMA-24 (0.98 kbps) SDR -14.30, VISQOL 3.64, and speech-only WT-S-24 (0.52 kbps, 40 tokens/s) VISQOL 3.33, SingMOS 2.42, the worst of all rows [A1, A2, A6] (Table 5, p. 18).
- C5: Signal-level metrics (SDR, SI-SNR) are unreliable indicators for DAC, WavTokenizer and Discrete WavLM, which are not trained for waveform fidelity yet sound plausible on perceptual metrics (UTMOS, DNSMOS, VISQOL, SingMOS); the authors say models optimised for perceptual or downstream goals can have low signal reconstruction yet sound good [A1] (Sec. 3.1, pp. 17-18).
- C6: Speech reconstruction: Enc-SMA-24 at 24 kbps has PESQ 3.71, WER 2.77; DAC-SMA-24 at 24 kbps PESQ 4.45, STOI 0.99, WER 2.72; SpeechTokenizer 4 kbps WER 2.41, Mimi 4.4 kbps PESQ 3.38; SQ-Codec 3 kbps WER 2.37 (lowest), UTMOS 3.90; WavTokenizer 0.98 kbps WER 8.10; ground truth WER 2.83 [A1] (Table 4, p. 17).
- C7: In DASB speech tasks the semantic Discrete WavLM is best on most discriminative tasks and SpeechTokenizer second; acoustic tokenizers (EnCodec, DAC, WavTokenizer, SQ-Codec) need more data and tuning; no discrete tokenizer reaches the continuous WavLM baseline on ASR (4.07 vs best 4.78 WER clean) or intent classification (86.1 vs 78.73) [A1, A3] (Table 7, p. 21, Sec. 3.2, pp. 20-21).
- C8: DASB on non-speech: sound event classification (ESC-50) tops out at 37.0% (EnCodec 8Q) versus 92.9% continuous; music genre (GTZAN) at 70.33% (EnCodec 2Q) versus 87.0% continuous; more codebooks lower genre accuracy (EnCodec 39.67% at 32Q); the authors say RVQ tokenizers struggle most on small datasets such as ESC-50 and GTZAN, with high bitrate amplifying this, and that larger downstream heads helped acoustic tokenizers converge in unreported tuning [A1, A3, A7] (Table 9, p. 22, Sec. 3.2, p. 21).
- C9: DASB separation: best discrete SI-SDRi is 9.53 dB for audio (EnCodec 8Q, continuous 15.07) and 1.98 for music (EnCodec 8Q, continuous 13.29); high-bitrate (32Q) tokens separate far worse (audio -1.73, music -11.72), polyphonic music being hardest; the mixture itself scores -16.5 dB (audio) and -7.71 (music) [A1, A3] (Table 9, p. 22, Sec. 3.2, p. 20).
- C10: Across DASB, more codebooks (2, 8, 32) improve reconstruction but often reduce downstream performance, and medium bitrate usually balances best; earlier RVQ codebooks carry more phonetic information, later ones redundancy [A1, A3] (Sec. 3.2, p. 20).
- C11: In speech LMs (SALMon, ZeroSpeech), HuBERT 25 Hz is best on semantic tasks (sBLIMP 60.89), purely acoustic tokenizers EnCodec and DAC show negligible semantic ability (sBLIMP about 51), Discrete WavLM leads acoustic consistency on average (gender 92.0%, sentiment 70.0%, speaker 86.5%, though WT-SMA-24 at 78.5 and Mimi at 77.0/71.5 beat it on sentiment); upweighting the semantic stream (x100) lifts Mimi sBLIMP from 52.25 to 60.17; sentiment-acoustic alignment stays near chance for all [A1, A4] (Table 10, p. 24, Sec. 3.3.1, pp. 24-25).
- C12: TTS (VALL-E on LibriTTS): best UTMOS is the ESPnet EnCodec retrained on LibriTTS only (Enc-S-24, 3.77) while the released multi-domain EnCodec gets 2.31 and DAC 11.71 dWER; best dWER Discrete WavLM 4.32, then WavTokenizer 4.67; SQ-Codec failed to converge; the authors say that with high-data regimes and in-domain speech, acoustic tokenizers such as EnCodec can be competitive with semantic ones [A1, A4] (Table 11, p. 27, Sec. 3.3.2, p. 27).
- C13: Text-to-audio LM (AudioCaps FAD, lower better): the audio-only EnCodec 16 kHz (4 codebooks, 50 Hz) has the best text-conditioned FAD (1.955) and unconditional FAD (3.548); WavTokenizer SMA-24 (1 codebook, 75 tokens/s) is second on text-conditioned FAD (2.594) despite the weakest reconstruction apart from the music-only EnCodec (recon FAD 5.018 vs 12.611 for Enc-M-32); Enc-SMA-24 FAD 3.771; DAC-44 6.929; SQ-Codec 7.733; music-only EnCodec 10.110 [A1, A4, A5] (Table 12, p. 29).
- C14: The authors conclude that for sound-effect generation domain-specific tokenizer training matters (general-audio EnCodec best FAD) and that best reconstruction does not mean best LM modelling; a single-stream tokenizer simplifies LM modelling and may close the gap with more compute [A1, A4, A5] (Sec. 3.3.3, p. 29).
- C15: Text-to-music (MusicCaps FAD): music-only EnCodec 32 kHz (4 codebooks) best text-conditioned FAD 4.264 (DAC-44 8.398, Enc-SMA-24 11.173, DAC-24 9.403, SQ 14.211, WavTokenizer 17.050); on FMA-test DAC-44 has the lowest FAD (8.115) and Enc-M-32 the best KLD and CLAP; for music, higher sample rate and bitrate help, contrasting with audio and speech where higher bitrate was harder to model [A1, A4, A5] (Table 13, p. 31, Sec. 3.3.4, pp. 30-32).
- C16: For music only, unconditional generation (continuation from a 2 s prompt) beats text-conditioned on both MusicCaps and FMA, attributed to the music setup's higher CFG dropout (30% vs 10% for audio) and, for FMA, poor text prompts; for text-to-audio Table 12 shows the opposite for 5 of 7 tokenizers, e.g. text-conditioned vs unconditional FAD 3.771 vs 5.996 (Enc-SMA-24), 1.955 vs 3.548 (Enc-A-16), 2.594 vs 4.441 (WT-SMA-24) [A4, A5] (Table 13, Sec. 3.3.4, pp. 30-31, Table 12, p. 29).
- C17: No tokenizer wins on all axes: performance is task- and domain-dependent, and some tokenizers with poor signal fidelity do well downstream [A1] (Fig. 4, p. 32, Sec. 3.4, pp. 32-33).
- C18: Controlled ablation (DAC backbone, 1k h per domain): reconstruction peaks when test domain matches training domain, and training on all three domains is worse on each domain than domain-specific training; RVQ beats SVQ and FSQ overall (SVQ worst, e.g. UTMOS 2.19 vs 3.35 for RVQ-S at 16 kHz); the text says FSQ at 16 kHz surpasses RVQ on UTMOS and DNSMOS, but in Table 15 FSQ-S beats RVQ-S only on DNSMOS (3.21 vs 3.16), not UTMOS (3.29 vs 3.35), while beating RVQ-A and RVQ-3 on UTMOS [A1] (Table 15, p. 35, Sec. 4.2, p. 36).
- C19: Ablation: RVQ models trained at 44.1 kHz improve on most metrics even evaluated at 16 kHz (e.g. RVQ-S SDR 4.08 to 8.24, UTMOS 3.35 to 3.62, though WER rises 2.04 to 2.63), while FSQ degrades on several metrics at higher rate; SSL distillation from a speech model helps speech metrics but its benefit is limited by the teacher's domain in audio and music [A1] (Table 15, p. 35, Table 16, p. 35, Sec. 4.2, p. 36).
- C20: Open problems listed: generalisation at scale, joint signal fidelity and semantic utility, fair consistent evaluation, a gap between reproduced and reported results, semantic distillation beyond speech (music, general audio underexplored), discrete vs continuous representations, unified tokenizers across domains, and trustworthiness (deepfakes) [] (Sec. 5, pp. 36-37).
- C21: Streamability needs causal convolutions or attention masks; many SSL tokenizers use non-causal encoders; per the survey's citations, codec sizes span fewer than 2M parameters (LPCNet, real time at 1.6 kbps) through about 14M (EnCodec, 1.5 kbps) to 159M (BigCodec, 1.04 kbps) [] (Sec. 2.5, p. 14).

## Implementation notes

- Checkpoints: the released ESPnet-Codec models for the ablations (HF collection espnet/codec-survey-pre-trained-models) and the authors' tokenizer database site (source: p. 34, p. 1).
- Compute: ablations 2xGH200 about 48 h; music LMs 4xA100-40GB 48 h; audio LM 2xA100-80GB 48 h; TTS 1xA100 96 h (source: Table 18, p. 53).
- Audio LM recipe: AudioCraft, T5-Base conditioning, 100k steps, batch 128 x 10 s clips, mix-up, top-k 250 (audio); music 200k steps, batch 32 x 10 s, CFG dropout 30% (source: Sec. 3.3.3-3.3.4, pp. 28-30).
- Token rate per second equals frame rate times codebooks used; Table 4 lists it directly (150, 600, 2400 for EnCodec 2/8/32 codebooks) (source: Table 4, p. 17).
- No result is for game audio, chiptune or FM-synth sound; how it would behave is unmeasured (source: assumption).

## Limitations / open questions

- The authors say the models and datasets are moderately sized; they argue the key findings follow larger-scale trends (Sec. 5, p. 37).
- Generative audio metrics (FAD, KLD, CLAP) depend on the vocoder as well as the LM, so poor scores may reflect vocoder limits (Sec. 3.3.3, p. 28).
- The text says the music-specific EnCodec is best on all metrics and datasets and that WavTokenizer has the best reconstruction FAD, but Table 13 shows DAC-44 with the lowest FAD on FMA text-conditioned and on reconstruction in both sets (reader's observation, Table 13, p. 31).
- The ablation text says multi-domain training is worse on each domain than domain-specific training, but on non-speech audio Table 16 is mixed: at 16 kHz the music-trained RVQ-M beats the audio-trained RVQ-A on AudioSet SDR (0.65 vs -0.59) and SI-SNR (-4.07 vs -5.56), and RVQ-3 also beats it on SDR (0.14), RVQ-A winning only VISQOL (4.22); at 44.1 kHz RVQ-3 beats RVQ-A on audio SDR (3.99 vs 3.65), SI-SNR (0.70 vs 0.48) and VISQOL (3.83 vs 3.78), and RVQ-M on music SDR (8.54 vs 8.33) (reader's observation, Table 16, p. 35).
- The paper's bitrate statements depend on the setting: for LM generation higher bitrate helps music and is harder to model for audio and speech (p. 32), while in DASB probing and separation 32 codebooks hurt music too (genre 39.67%, separation -11.72 dB) and medium bitrate is usually best (reader's observation, Table 9, p. 22, p. 20).
- Table 2 marks SQ-Codec as trained on speech, music and audio while Table 1 marks it speech only (reader's observation, p. 6, p. 15).
- Open question: no single-stream tokenizer evaluated here was trained on synthesised or very short sound effects; effect on game audio is unmeasured.
- Open question (reader's): DASB sound-event accuracy (<=37% vs 92.9% continuous) is measured with lightweight probes on the small ESC-50 set, which the authors say hurts RVQ tokens most; whether the gap would persist with more data or a larger head, and whether a continuous audio encoder is the better policy input, is not tested.

## Related

- related [[wu2024overview]]: earlier short overview of codecs and codec-based LMs.
- related [[guo2025discrete]]: speech-only token review with its own reconstruction benchmark; cited by this paper.
- related [[he2021masked]]: MAE-style audio features are a continuous alternative to the discrete tokens benchmarked here.
- Not on main yet: EnCodec, DAC, SoundStream, SNAC, WavTokenizer, Moshi/Mimi notes (branch papers-audio-codecs); audio-in-RL notes (papers-audio-agents).

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
