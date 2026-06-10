import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model

class GenPromptEmb(nn.Module):
    def __init__(
        self,
        data_path = '23p75__90p50',
        model_name = "gpt2",
        #model_name = "meta-llama/Meta-Llama-3-8B-Instruct",
        device = 'cuda:0',
        input_len = 96,
        d_model = 768,
        layer = 12,
        divide = 'train'
    ):  
        super(GenPromptEmb, self).__init__()
        self.data_path = data_path
        self.device = device
        self.input_len =  input_len
        self.model_name = model_name
        self.d_model = d_model
        self.layer = layer
        self.len = self.input_len-1
        self.variable_names = {
          
           
           '23p75__90p50' : ['date','u10','v10','sp','swvl1','swvl4','z','tcc','r','rmm1','rmm2','sst','tp','Temperature(C)'],
 
        
        }
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2Model.from_pretrained(model_name).to(self.device)

    def _prepare_prompt(self, input_template, in_data, in_data_mark, i, j):
        # Time series value
        values = in_data[i, :, j].flatten().tolist()
        values_str = ", ".join([f"{value:.5f}" for value in values]) 
        
        # Last token
        trends = torch.sum(torch.diff(in_data[i, :, j].flatten()))
        trends_str = f"{trends.item():0f}"
        
        
        # Date
        if self.data_path in ['FRED', 'ILI']:
            start_date = f"{int(in_data_mark[i,0,2]):02d}/{int(in_data_mark[i,0,1]):02d}/{int(in_data_mark[i,0,0]):04d}"
            end_date = f"{int(in_data_mark[i,self.len,2]):02d}/{int(in_data_mark[i,self.len,1]):02d}/{int(in_data_mark[i,self.len,0]):04d}"
        elif self.data_path in ['ETTh1', 'ETTh2', 'ECL']:
            start_date = f"{int(in_data_mark[i,0,2]):02d}/{int(in_data_mark[i,0,1]):02d}/{int(in_data_mark[i,0,0]):04d} {int(in_data_mark[i,0,4]):02d}:00"
            end_date = f"{int(in_data_mark[i,self.len,2]):02d}/{int(in_data_mark[i,self.len,1]):02d}/{int(in_data_mark[i,self.len,0]):04d} {int(in_data_mark[i,self.len,4]):02d}:00"
        else: # ETTm1, ETTm2, Weather
            start_date = f"{int(in_data_mark[i,0,2]):02d}/{int(in_data_mark[i,0,1]):02d}/{int(in_data_mark[i,0,0]):04d} {int(in_data_mark[i,0,4]):02d}:{int(in_data_mark[i,0,5]):02d}"
            end_date = f"{int(in_data_mark[i,self.len,2]):02d}/{int(in_data_mark[i,self.len,1]):02d}/{int(in_data_mark[i,self.len,0]):04d} {int(in_data_mark[i,self.len,4]):02d}:{int(in_data_mark[i,self.len,5]):02d}"

        # Variable name
        variable_name = self.variable_names[self.data_path][j]
        
        # Prompt
        in_prompt = input_template.replace("value1, ..., valuen", values_str)
        in_prompt = in_prompt.replace("Trends", trends_str)
        in_prompt = in_prompt.replace("[t1]", start_date).replace("[t2]", end_date)
        in_prompt = in_prompt.replace("VariableName", variable_name)
        #print("in_prompt: ", in_prompt)

        tokenized_prompt = self.tokenizer.encode(in_prompt, return_tensors="pt").to(self.device)
        return tokenized_prompt

    def forward(self, tokenized_prompt):
        with torch.no_grad():
            prompt_embeddings = self.model(tokenized_prompt).last_hidden_state
        return prompt_embeddings

    def generate_embeddings(self, in_data, in_data_mark):
    
            input_templates = {
              
                '23p75__90p50': "The historical values of VariableName from [t1] to [t2] were value1, ..., valuen. The overall trend was Trends. Based on this data, learn the pattern of data and forecast the Temperature(C) values starting 7 days after [t2]. Try to catch the peaks",
            }

            input_template = input_templates.get(self.data_path, input_templates['23p75__90p50'])
            
            tokenized_prompts = []
            max_token_count = 0
            
            
            for i in range( len(in_data)):
                for j in range(0, in_data.shape[2]):
                    tokenized_prompt = self._prepare_prompt(input_template, in_data, in_data_mark, i, j).to(self.device)
                    max_token_count = max(max_token_count, tokenized_prompt.shape[1])
                    tokenized_prompts.append((i, tokenized_prompt.to(self.device), j))

            in_prompt_emb = torch.zeros((len(in_data), max_token_count, self.d_model, in_data.shape[2]), dtype=torch.float32, device=self.device)

            for i, tokenized_prompt, j in tokenized_prompts:
                prompt_embeddings = self.forward(tokenized_prompt)
                padding_length = max_token_count - tokenized_prompt.shape[1]
                if padding_length > 0:
                    last_token_embedding = prompt_embeddings[:, -1, :].unsqueeze(1)
                    padding = last_token_embedding.repeat(1, padding_length, 1)
                    prompt_embeddings_padded = torch.cat([prompt_embeddings, padding], dim=1)
                else:
                    prompt_embeddings_padded = prompt_embeddings
                        
                in_prompt_emb[i, :max_token_count, :, j] = prompt_embeddings_padded
                last_token_emb = in_prompt_emb[:, max_token_count-1:max_token_count, :, :]
                last_token_emb = last_token_emb.squeeze()

            return last_token_emb