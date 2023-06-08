import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import numpy as np


class Realwav:
  def __init__(self):
    MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
    self.processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    self.model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

  
  def sound_and_sentence(self,sound,sentence):
    inputs = self.processor(sound, sampling_rate=16_000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = self.model(inputs.input_values, attention_mask=inputs.attention_mask).logits
    
    probs = torch.softmax(logits, dim=-1)
    ref_ids = self.processor(text=sentence)["input_ids"]
    scores = []
    ref_count = 0
    pred_ids = torch.argmax(logits[0], dim=-1)
    for seq_idx in range(pred_ids.shape[0]):
      if pred_ids[seq_idx] != 0:
        print(f"position of the word {sentence[ref_count]}: {seq_idx}")
        ref_id = ref_ids[ref_count]
        conf_score = probs[0, seq_idx, ref_id].tolist()
        scores.append(conf_score)
        print(conf_score)
        ref_count += 1
        if ref_count >= len(ref_ids):
          break

    sentence_score = np.mean(scores)

    return sentence_score