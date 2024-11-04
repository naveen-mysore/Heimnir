import torch
import torch.nn as nn
import os
import json
from tqdm import tqdm

from models.llmbased.backbones.llama import Llama
from models.llmbased.timeseries.timellm.embeddings.patch_embeddings import PatchEmbedding
from models.llmbased.timeseries.timellm.layers.StandardNorm import Normalize
from models.llmbased.timeseries.timellm.layers.mapping_layer import MappingLayer
from models.llmbased.timeseries.timellm.layers.output_projection import FlattenHead
from models.llmbased.timeseries.timellm.layers.reprogramming_layer import ReprogrammingLayer
from models.llmbased.timeseries.timellm.utils.prompt_gen import PromptGen


class TimeLLM(nn.Module):
    def __init__(self):
        super(TimeLLM, self).__init__()
        self.configs = self._load_configs()
        self.device = None

        self.backbone = None
        self.mapping_layer = None

        self.pred_len = self.configs['model']['pred_len']['value']
        self.seq_len = self.configs['model']['seq_len']['value']
        self.patch_len = self.configs['model']['patch_len']['value']
        self.stride = self.configs['llm']['stride']['value']
        self.n_heads = self.configs['llm']['n_heads']['value']
        self.d_ff = self.configs['model']['d_ff']['value']
        self.d_llm = self.configs['llm']['d_llm']['value']

        self.normalize_layers = Normalize(7, affine=False)
        self.patch_embedding = PatchEmbedding(32, self.patch_len, self.stride, 0.1)
        self.reprogramming_layer = ReprogrammingLayer(32, self.n_heads, self.d_ff, self.d_llm)
        self.output_projection = FlattenHead(self.configs)
        self.prmpt = PromptGen()

        self.train_epochs = self.configs['learning']['train_epochs']['value']
        self.train_loader = None

    def set_data(self, data_loader):
        self.train_loader = data_loader

    def move_to_device(self, device):
        self.to(device)
        # Move all dependent modules to the device
        self.normalize_layers.to(device)
        self.patch_embedding.to(device)
        self.reprogramming_layer.to(device)
        self.output_projection.to(device)
        self.prmpt.to(device)
        self.backbone.to(device)

    def _load_configs(self):
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs.json")
        with open(config_path) as f:
            configs = json.load(f)
        return configs

    def __run_inference_on_llama(self, llama_enc_out, n_vars):
        dec_out = self.backbone(llama_enc_out)
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        return dec_out

    def _normalize_data(self, x_enc):
        return self.normalize_layers(x_enc, 'norm')

    def _denormalize_data(self, data):
        dec_out = self.normalize_layers(data, 'denorm')
        print(f"dec_out shape {dec_out.shape}")
        return dec_out

    def set_backbone_as_llama(self):
        self.backbone = Llama()
        self.mapping_layer = MappingLayer(self.backbone.get_vocab_size())

    def _get_prompt_embeddings(self, x_enc):
        # Extract trends and information from input and Generate prompt and get embeddings
        prompt = self.prmpt.generate_prompt(x_enc, self.configs["task"]["description"], self.pred_len, self.seq_len)
        prompt_embeddings = self.prmpt.tokenize_prompt_and_get_prompt_embeddings(prompt,
                                                                                 self.backbone.get_tokenizer(),
                                                                                 self.backbone.get_model(),
                                                                                 x_enc.device)
        return prompt_embeddings

    def _get_source_embeddings(self):
        word_embeddings = self.backbone.get_model().get_input_embeddings().weight
        source_embeddings = self.mapping_layer(word_embeddings.permute(1, 0)).permute(1, 0)
        print(f"source embeddings shape {source_embeddings.shape}")
        return source_embeddings

    def _get_reprogramming_output(self, x):
        source_embeddings = self._get_source_embeddings()
        enc_out = self.reprogramming_layer(x, source_embeddings, source_embeddings)
        return enc_out

    def forward(self, x_enc):
        # normalize the input
        print(f"x_enc shape {x_enc.shape}")
        x_enc = self._normalize_data(x_enc)

        prompt_embeddings = self._get_prompt_embeddings(x_enc)
        print(f"prompt embedding shape {prompt_embeddings.shape}")

        # Get patch embeddings
        patch_embedding, n_vars = self.patch_embedding(x_enc.to(torch.float32))
        print(f"patch embeddings shape {patch_embedding.shape}")

        # Target embeddings, Source embeddings, and Value embeddings
        # Get source embeddings
        reprogrammed_embeddings = self._get_reprogramming_output(patch_embedding)
        print(f"enc_out shape {reprogrammed_embeddings.shape}")

        llama_enc_out = torch.cat([prompt_embeddings, reprogrammed_embeddings], dim=1)

        # Run inference on llama
        dec_out = self.__run_inference_on_llama(llama_enc_out, n_vars)
        print(f"dec_out shape {dec_out.shape}")

        dec_out = self.output_projection(dec_out)
        print(f"output after projection shape {dec_out.shape}")

        dec_out = self._denormalize_data(data=dec_out)
        print(f"dec_out shape {dec_out.shape}")

        return dec_out

    def train_time_llm(self):
        for epoch in range(self.train_epochs):
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(self.train_loader)):
                print(batch_x.shape)
                device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
                batch_x = batch_x.to(device)
                outputs = self.forward(batch_x)
                print(outputs.shape)
                outputs = outputs[:, -self.pred_len:, -1:]
                print(outputs.shape)
                if i == 1:
                    break
