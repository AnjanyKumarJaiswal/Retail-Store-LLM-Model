from datasets import load_dataset
import pandas as pd 
from transformers import AutoTokenizer

class PrepareAndLoadDataset:
    
    def tokenization(model_name='gpt2-medium', files=None):
        
        datasets = load_dataset('json', data_files=files)
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
        tokenizer.pad_token = tokenizer.eos_token

        
        def tokenization_function(examples):
            customer_texts = examples['instruction']
            salesman_texts = examples['response']
            
            print(f"\n--- Processing New Batch ---\n")
            
            try:
                inputs = tokenizer(customer_texts, truncation=True, padding='max_length', max_length=25)
                labels = tokenizer(salesman_texts, truncation=True, padding='max_length', max_length=25)
                print(f"\nTokenized inputs and labels are done....moving to next part...\n")
            except Exception as e:
                print(f"Error tokenizing customer texts: {e}")
                return None
            
            
            inputs['labels'] = [
            (label if label != tokenizer.pad_token_id else -100) for label in labels['input_ids']
            ]
            
            
            return inputs
        
        
        
        tokenized_datasets = datasets.map(tokenization_function, batched=True)
        
        print("Data tokenized perfeclty......moving next....")
        
        return tokenized_datasets


if __name__ == "__main__":
    datafile = {
        "train":"../data/train_dataset.jsonl",
        "valid":"../data/valid_dataset.jsonl"
    }
    
    dataset = PrepareAndLoadDataset.tokenization(files=datafile)

    print(dataset)

