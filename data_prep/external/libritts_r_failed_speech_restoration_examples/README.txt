These text files contain file paths where speech restoration by Miipher [1] may have failed. Speech restoration is not always perfect, so some phonemes may be lost or changed during the restoration process.

The authors of the LibriTTS-R paper [2] ran automatic speech recognition (ASR) on all LibriTTS-R samples and created these lists of samples with a word error rate (WER) above a certain threshold. The experiments in the LibriTTS-R paper were conducted using these files that may have failed to be restored. However, the files included in these lists are likely to have uncorresponding transcripts and waveforms. Therefore, we recommend excluding them during model training.

In addition, some of the original LibriTTS samples do not align the transcript and waveform, such as mentioned in LibriTTSLabel [3] used in ESPnet. We recommend that users re-execute force alignment for LibriTTS-R.

[1] Y. Koizumi, et al., "Miipher: A Robust Speech Restoration Model Integrating Self-Supervised Speech and Text Representations," WASPAA 2023.
[2] Y. Koizumi, et al., "LibriTTS-R: Restoration of a Large-Scale Multi-Speaker TTS Corpus," INTERSPEECH 2023.
[3] https://github.com/kan-bayashi/LibriTTSLabel 
