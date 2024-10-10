from transformers import GPT2LMHeadModel, Trainer, TrainingArguments
from preprocessing import PrepareAndLoadDataset

class FineTunedGPT2LLMModel:

    def fine_tune_model(data_files):
        model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
        model.config.pad_token_id = model.config.eos_token_id
        
        
        datasets = PrepareAndLoadDataset.tokenization(files=data_files)
        
        training_args = TrainingArguments(
            output_dir='../utils',
            eval_strategy='steps',
            eval_steps=200,
            save_steps=200,
            learning_rate=0.001,  
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            num_train_epochs=50,
            weight_decay=0.01,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            lr_scheduler_type="cosine",
            logging_dir='../logs',
            logging_steps=200,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=datasets['train'],
            eval_dataset=datasets['valid']
        )
        
        
        trainer.train()
        trainer.evaluate()
        

if __name__ == "__main__":
    data_files = {
        "train":"../data/train_dataset.jsonl",
        "valid":"../data/valid_dataset.jsonl"
    }
    FineTunedGPT2LLMModel.fine_tune_model(data_files=data_files)
