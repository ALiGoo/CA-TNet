import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class ToneNetDatasetWordEmbeddingRhythm(Dataset):
    def __init__(self, data, all_S_data, used_rhythm_feature_nums, random_jointed_s=False):
        self.data = data
        self.all_S_data = all_S_data
        self.used_rhythm_feature_nums = used_rhythm_feature_nums
        self.random_jointed_s = random_jointed_s
    
    def __getitem__(self, idx):
        f0 = torch.tensor(self.data[idx]['f0'], dtype=torch.float32).unsqueeze(dim=-1)
        syllable_token = torch.tensor(self.data[idx]['syllable_token'], dtype=torch.long)
        syllable_boundary = torch.tensor(self.data[idx]['syllable_boundary'], dtype=torch.long)
        word_token = torch.tensor(self.data[idx]['word_token'], dtype=torch.long)
        word_boundary = torch.tensor(self.data[idx]['word_boundary'], dtype=torch.long)
        syllable_idx = torch.tensor(self.data[idx]['syllable_idx'], dtype=torch.long)
        syllable_duration_feature = torch.tensor(self.data[idx]['syllable_duration_feature'], dtype=torch.float32)
        syllable_duration_token = torch.tensor(self.data[idx]['syllable_duration_token'], dtype=torch.long)
        tone = torch.tensor(self.data[idx]['tone'], dtype=torch.long)

        if self.random_jointed_s:
            auxiliary_data = random.choice(self.all_S_data)
            auxiliary_f0 = torch.tensor(auxiliary_data['f0'], dtype=torch.float32).unsqueeze(dim=-1)
            auxiliary_syllable_token = torch.tensor(auxiliary_data['syllable_token'], dtype=torch.long)
            auxiliary_syllable_boundary = torch.tensor(auxiliary_data['syllable_boundary'], dtype=torch.long)
            auxiliary_syllable_boundary = (auxiliary_syllable_boundary + syllable_boundary[-1]) % 2
            auxiliary_word_token = torch.tensor(auxiliary_data['word_token'], dtype=torch.long)
            auxiliary_word_boundary = torch.tensor(auxiliary_data['word_boundary'], dtype=torch.long)
            auxiliary_word_boundary = (auxiliary_word_boundary + word_boundary[-1]) % 2
            auxiliary_syllable_idx = torch.tensor(auxiliary_data['syllable_idx'], dtype=torch.long)
            auxiliary_syllable_idx = auxiliary_syllable_idx + syllable_token.size(0)
            auxiliary_syllable_duration_feature = torch.tensor(auxiliary_data['syllable_duration_feature'], dtype=torch.float32)
            auxiliary_syllable_duration_token = torch.tensor(auxiliary_data['syllable_duration_token'], dtype=torch.long)
            auxiliary_tone = torch.ones(len(auxiliary_data['tone']), dtype=torch.long) * -100

            f0 = torch.cat([f0, auxiliary_f0])
            syllable_token = torch.cat([syllable_token, auxiliary_syllable_token])
            syllable_boundary = torch.cat([syllable_boundary, auxiliary_syllable_boundary])
            word_token = torch.cat([word_token, auxiliary_word_token])
            word_boundary = torch.cat([word_boundary, auxiliary_word_boundary])
            syllable_idx = torch.cat([syllable_idx, auxiliary_syllable_idx])
            syllable_duration_feature = torch.cat([syllable_duration_feature, auxiliary_syllable_duration_feature])
            syllable_duration_token = torch.cat([syllable_duration_token, auxiliary_syllable_duration_token])
            tone = torch.cat([tone, auxiliary_tone])

        return f0, syllable_token, syllable_boundary, word_token, word_boundary, syllable_idx, syllable_duration_feature, syllable_duration_token, tone
    
    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):        
        f0, syllable_token, syllable_boundary, word_token, word_boundary, syllable_idx, syllable_duration_feature, syllable_duration_token, tone = zip(*batch)

        inputs = dict()
        inputs['f0'] = pad_sequence(f0, batch_first=True)
        inputs['syllable_token'] = pad_sequence(syllable_token, batch_first=True)
        inputs['syllable_boundary'] = pad_sequence(syllable_boundary, batch_first=True)
        inputs['word_token'] = pad_sequence(word_token, batch_first=True)
        inputs['word_boundary'] = pad_sequence(word_boundary, batch_first=True)
        inputs['syllable_idx'] = pad_sequence(syllable_idx, batch_first=True, padding_value=-1)
        inputs['syllable_duration_feature'] = pad_sequence(syllable_duration_feature, batch_first=True)[:,:,:self.used_rhythm_feature_nums]
        inputs['syllable_duration_token'] = pad_sequence(syllable_duration_token, batch_first=True)
        tone = pad_sequence(tone, batch_first=True, padding_value=-100)

        return inputs, tone
