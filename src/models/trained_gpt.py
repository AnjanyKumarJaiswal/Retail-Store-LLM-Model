from transformers import GPT2LMHeadModel, AutoTokenizer
import torch

class GenerateResponses:
    def model_responses(query, trained_model, model_name):
        
        model = GPT2LMHeadModel.from_pretrained(trained_model)
        model.eval()
        
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
        
        
        inputs = tokenizer(query, return_tensors='pt', padding=True, truncation=True)
        
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=60,
                num_return_sequences=1,
                do_sample=True,        # Enables sampling (adds randomness)
                temperature=0.7,        # Adjusts creativity of output
                top_k=50,
                top_p=0.9         
            )
        
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response


if __name__ == "__main__":
    query = "Can I modify my order after placing it?"
    response = GenerateResponses.model_responses(query)
    print(f"Model response: {response}")
