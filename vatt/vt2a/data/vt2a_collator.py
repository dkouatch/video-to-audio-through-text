import torch
import numpy as np
import copy
from transformers import LlamaTokenizer, GemmaTokenizer


class VT2A_Collator:

    def __init__(self, tokenizer_path="/pscratch/sd/x/xiuliu/ltu/pretrained_mdls/vicuna/", padding=True, max_length=None, pad_to_multiple_of=None, return_tensors="pt", label_pad_token_id=-100):
        
        if tokenizer_path == "/pscratch/sd/x/xiuliu/ltu/pretrained_mdls/vicuna/":
            self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
            self.tokenizer.pad_token_id = (
                0  # unk. we want this to be different from the eos token
            )
        else:
            self.tokenizer = GemmaTokenizer.from_pretrained(tokenizer_path)
            
        self.tokenizer.padding_side = "left"
        self.data = []
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id
        self.return_tensors = return_tensors

    def __call__(self, batch):
        text_ids = [{"input_ids": b["input_ids"]} for b in batch]
        return_tensors = self.return_tensors
        new_text_ids = self.tokenizer.pad(
            text_ids,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        # print("AAA", batch[0]["input_ids"], len(batch[0]["input_ids"]))
        # print("LLL", batch[0]["attention_mask"], len(batch[0]["attention_mask"]))
        # adjust the attention mask
        ori_att_mask = new_text_ids["attention_mask"]
        # print("XXX", ori_att_mask[0], len(ori_att_mask[0]))
        video_att_mask = torch.ones([ori_att_mask.shape[0], 10]) # batch size [2, 10]
        new_attention_mask = torch.concat([video_att_mask, ori_att_mask], dim=1)
        # print("HHH", new_attention_mask[0], len(new_attention_mask[0]))

        return {
            # 'file_id': [b["file_id"] for b in batch],
            'audio_tokens': torch.stack([b['audio_tokens'] for b in batch]),
            'video_inputs': torch.stack([b['video_inputs'] for b in batch]),
            "input_ids": new_text_ids["input_ids"],
            "attention_mask": new_attention_mask 
        }