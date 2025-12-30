import torch
import torch.nn as nn
from transformers import Qwen2ForCausalLM
from typing import Any, Dict, List, Optional, Tuple, Union
class Classifier(nn.Module):
    def __init__(
        self,
        embedding_dim=64,
        projection_dim=64,
        num_classes=2,
    ):
        super().__init__()
        self.proj1 = nn.Linear(embedding_dim, projection_dim)
        self.gelu1 = nn.GELU()
        self.classification = nn.Linear(projection_dim, num_classes)

        #torch.nn.init.kaiming_normal_(self.classification.weight, mode='fan_in', nonlinearity='gelu')
        # 正确初始化（匹配各层的激活函数）
        
        torch.nn.init.kaiming_normal_(
            self.classification.weight, 
            mode='fan_in', 
            nonlinearity='linear' 
        )


    def forward(self, x):
        x = self.proj1(x)
        x = self.gelu1(x)
        x = self.classification(x)
        return x
class SATDAutoModel(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        # 添加一个自定义的 nn.Linear 层
        self.Classifier = Classifier(embedding_dim=config.hidden_size,projection_dim=128,num_classes=2)
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

        >>> model = Qwen2ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        embedding_token_idx = (input_ids == 151667).nonzero(as_tuple=False) - torch.Tensor([0, 1]).to(input_ids.device).to(torch.int32)
        embedding_token_features = []
        for i in range(embedding_token_idx.size(0)):
            embedding_token_features.append(hidden_states[embedding_token_idx[i][0], embedding_token_idx[i][1], :])
        embedding_token_features = torch.stack(embedding_token_features, dim=0)
        return self.Classifier(embedding_token_features)
        
    