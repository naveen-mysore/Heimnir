import torch
import torch.nn as nn


class PromptGen(nn.Module):
    def __init__(self):
        super(PromptGen, self).__init__()

    def tokenize_prompt_and_get_prompt_embeddings(self, prompt, tokenizer, llm_model, device):
        prompt = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = llm_model.get_input_embeddings()(prompt.to(device))  # (batch, prompt_token, dim)
        return prompt_embeddings

    def generate_prompt(self, x_enc, description, pred_len, seq_len):
        def calcute_lags(_x_enc):
            q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
            k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
            res = q_fft * torch.conj(k_fft)
            corr = torch.fft.irfft(res, dim=-1)
            mean_value = torch.mean(corr, dim=1)
            _, lags = torch.topk(mean_value, 5, dim=-1)
            return lags

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompts = []
        # here batch is B*N( since N is 1 was us) this is batch
        for _batch in range(x_enc.shape[0]):
            min_values_str = str(min_values[_batch].tolist()[0])
            max_values_str = str(max_values[_batch].tolist()[0])
            median_values_str = str(medians[_batch].tolist()[0])
            lags_values_str = str(lags[_batch].tolist())
            prompt_for_one_sequence = (
                f"<|start_prompt|>Dataset description: {description}"
                f"Task description: forecast the next {str(pred_len)} steps given the previous {str(seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[_batch] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )
            prompts.append(prompt_for_one_sequence)
        return prompts

