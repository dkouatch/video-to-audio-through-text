import torch
import torch.nn as nn
import numpy as np
from peft import (
    LoraConfig,
    get_peft_model
)
from transformers import GemmaModel
from vt2a.modules.vt2a_mlm_alibi_uni_encodec import VT2AModel
import pytorch_lightning as pl
from vt2a.util import instantiate_from_config
from torch.optim.lr_scheduler import LambdaLR
from collections import OrderedDict


class V2TA(pl.LightningModule):

    def __init__(self, base_model_path, start_model_path, audio_cb_path,
                 latent_dim=128,
                 n_codebooks=4,
                 num_tokens=2048,
                 audio_seq_len=500,
                 visual_seq_len=500,
                 dim=1024,
                 num_layers=12,
                 num_heads=16,
                 dec_num_layers=12,
                 dec_num_heads=16,
                 weight_init='gaussian',
                 depthwise_init='current',
                 zero_bias_init=True,
                 mask_ratio_min=0.5,
                 mask_ratio_max=1.0,
                 mask_ratio_mu=0.75,
                 mask_ratio_std=0.25,
                 scheduler_config=None,
                 monitor=None,
                 ignore_key=[],
                 ckpt_path=None, #"/path/to/vatt_llama.ckpt",
                 finetune_llm=False
                 ):
        super().__init__()
        self.finetune_llm = finetune_llm
        self.base_model_path = base_model_path
        self.start_model_path = start_model_path
        self.initialize_encoder()
        self.eval_flag = False

        self.audio_token_decoder = VT2AModel(audio_cb_path, 
                                             latent_dim, n_codebooks, num_tokens, audio_seq_len, 
                                             visual_seq_len, dim, num_layers, num_heads, dec_num_layers, dec_num_heads, 
                                             weight_init, depthwise_init, zero_bias_init,
                                             mask_ratio_min, mask_ratio_max, mask_ratio_mu, mask_ratio_std, encoder_dim=2048)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.1)

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            base_keys = self.init_from_ckpt(ckpt_path, ignore_keys=ignore_key)
            base_params = list(filter(lambda kv: kv[0] in base_keys, self.named_parameters()))
            self.base_params = [i[1] for i in base_params]
        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config
        
        embed = torch.load("/path/to/pretrained_mdls/gemma_2b/word_embeds.pth")
        self.embed_tokens = nn.Embedding.from_pretrained(embed)
        self.embed_tokens.cpu()
        for name, param in self.embed_tokens.named_parameters():
            param.requires_grad = False
        self.embed_tokens.eval()
    
    def forward(self, x):
        self.embed_tokens.cpu()
        input_ids, video_input, attention_mask, audio_tokens = x["input_ids"], x["video_input"], x["attention_mask"], x["audio_tokens"]
        with torch.no_grad():
            inputs_embeds = self.embed_tokens(input_ids.cpu())
            v_hidden_states = self.encoder.model.encode_all_states(inputs_embeds=inputs_embeds.cuda(), video_input=video_input, attention_mask=attention_mask)[0]
        logits, t_masked = self.audio_token_decoder(audio_tokens, v_hidden_states.float(), attention_mask.float())
        return logits, t_masked
    
    def forward_encoder(self, x, embed_tokens):
        input_ids, video_input, attention_mask = x["input_ids"], x["video_input"], x["attention_mask"]
        inputs_embeds = embed_tokens(input_ids.cpu())
        v_hidden_states = self.encoder.model.encode_all_states(inputs_embeds=inputs_embeds, video_input=video_input, attention_mask=attention_mask)[0]
        return v_hidden_states

    
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
        return keys

    def initialize_encoder(self):
        device_map = "auto"
        self.encoder = GemmaModel.from_pretrained(
            self.base_model_path,
            load_embedding=False,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map=device_map,
        )
        for key, param in self.encoder.named_parameters():
            print(key, np.prod(param.size()))

        lora_r, lora_alpha, lora_dropout = 16, 32, 0.0
        lora_target_modules = ["q_proj", "v_proj"]

        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.encoder = get_peft_model(self.encoder, config)
        state_dict = torch.load(self.start_model_path, map_location='cpu')
        new_state_dict = OrderedDict()
        for key, val in state_dict.items():
            new_state_dict["base_model.model." + key.split(".model.model.")[-1]] = val
        msg = self.encoder.load_state_dict(new_state_dict, strict=False)
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False
        self.encoder.eval()
    
    def get_input(self, batch, bs=None, train=False):
        audio_tokens = batch["audio_tokens"]
        img_embs = batch["video_inputs"].half()
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"].half()
        audio_tokens = audio_tokens.to(memory_format=torch.contiguous_format).long()
        audio_tokens = audio_tokens.to(self.device)
        img_embs = img_embs.to(self.device)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        return {"audio_tokens": audio_tokens, "video_input": img_embs, "input_ids": input_ids, "attention_mask": attention_mask}

    def training_step(self, batch):
        self.eval_flag = False
        inputs = self.get_input(batch, train=True)
        logits, labels = self(inputs)
        total_loss = self.loss(logits, labels)
        log_dict_tot = {
            "train/total_loss": total_loss.clone().detach().mean()
        }
        self.log_dict(log_dict_tot, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)
        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return total_loss

    def on_train_start(self):
        self.optimizers().param_groups = self.optimizers()._optimizer.param_groups

    def validation_step(self, batch, batch_idx):
        self.eval_flag = True
        inputs = self.get_input(batch)
        logits, labels = self(inputs)
        total_loss = self.loss(logits, labels)
        log_dict_tot = {
            "val/total_loss": total_loss.clone().detach().mean()
        }
        self.log("val/total_loss", log_dict_tot["val/total_loss"],
                prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        params = []
        for name, param in self.named_parameters():
            if name.startswith('model.quantizer'):
                print(name)
                continue
            if param.requires_grad:
                params.append(param)
        opt = torch.optim.AdamW(
            params, lr=lr, betas=(0.9, 0.999), weight_decay=1e-6,
        )
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            # print("Setting up Noam Scheduler...")
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt
    
    def get_shift_value_from_step(self):
        wm_up_steps = self.scheduler_config['params']['warm_up_steps'][0]
        cur_step = self.global_step
        cycle_lengths = self.scheduler_config['params']['cycle_lengths'][0]
        if cur_step < wm_up_steps:
            return 1.0 - 0.5 * cur_step / wm_up_steps
        return  0.5 - 0.5 * (cur_step - wm_up_steps) / (cycle_lengths - wm_up_steps)
    
    def gaussian_mean_cosine_step(self):
        cycle_lengths = self.scheduler_config['params']['cycle_lengths'][0]
        mean = 0.25 + (0.95 - 0.25) * np.sin( self.global_step / cycle_lengths * np.pi / 2)
        return mean

    @torch.no_grad()
    def generate_audio(self, x, mask_temperature=25.5, temp=1.0, cfg_coef=5.):
        self.embed_tokens.cpu()
        input_ids, video_input, attention_mask = x["input_ids"], x["video_input"], x["attention_mask"]
        inputs_embeds = self.embed_tokens(input_ids.cpu())
        v_hidden_states = self.encoder.model.encode_all_states(inputs_embeds=inputs_embeds.cuda(), video_input=video_input, attention_mask=attention_mask)[0]
        out = self.audio_token_decoder.generate_audio_sample(v_hidden_states.float(), attention_mask.float(), mask_temperature=mask_temperature, sampling_temperature=temp, cfg_coef=cfg_coef)
        return out