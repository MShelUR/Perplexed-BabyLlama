from transformers import (
    GPT2TokenizerFast,
    LlamaForCausalLM,
    LlamaConfig,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import  Subset
from random import sample

from pathlib import Path
import wandb


from babylm_dataset import BabylmDataset


#############
LR = 2.5e-4
BATCH_SIZE = 32
SEQ_LENGTH = 128

TEMPERATURE = 2.0
ALPHA = 0.5
#############


PATH = Path("./")

teacher_dir1 = PATH / 'models/Llama-360M'
teacher_dir2 = PATH / 'models/gpt-705M'


MODEL_NAME = f'Baby-Llama-58M'
MODEL_OUTPUT = Path('./models') /  MODEL_NAME
EVAL_SAMPLES = 8192


wandb_log = False
loss_mode = "weighted" # averaged (base), min, max, weighted


tokenizer_path = PATH / "models/gpt-clean-16000.json"
tokenizer = GPT2TokenizerFast(tokenizer_file= str(tokenizer_path))
tokenizer.bos_token = "<s>"
tokenizer.eos_token = "</s>"
tokenizer.pad_token = "<pad>"

# in the original code I had random_chunk = False
# random_chunk=True is expected to improve the model performance a bit
train_dataset = BabylmDataset(PATH / "data/babylm_10M_clean", SEQ_LENGTH, tokenizer=tokenizer, random_chunk=True)
full_eval_dataset = BabylmDataset(PATH / "data/babylm_dev_clean", SEQ_LENGTH, tokenizer=tokenizer, offset=0)

eval_indices = sample(range(len(full_eval_dataset)), EVAL_SAMPLES)
eval_dataset = Subset(full_eval_dataset, eval_indices)




tokenizer.model_max_length = SEQ_LENGTH

config = LlamaConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=512,
    num_hidden_layers=16,
    intermediate_size=1024,
    num_attention_heads=8,
    bos_token_id=tokenizer.convert_tokens_to_ids("<s>"),
    eos_token_id=tokenizer.convert_tokens_to_ids("</s>"),
    pad_token_id=tokenizer.convert_tokens_to_ids("<pad>"),
    max_position_embeddings=2*SEQ_LENGTH,
)

student = LlamaForCausalLM(config)
# student = LlamaForCausalLM.from_pretrained(student_dir)


teacher1 = LlamaForCausalLM.from_pretrained(teacher_dir1)
teacher2 = GPT2LMHeadModel.from_pretrained(teacher_dir2)
teachers = [teacher1, teacher2]


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)


print(f'model num parameters: student = {student.num_parameters()}')
print(f'model num parameters: teacher1 = {teacher1.num_parameters()}')
print(f'model num parameters: teacher2 = {teacher2.num_parameters()}')



#  Distillation Trainer
#  We modified the Trainer from this repo https://github.com/philschmid/knowledge-distillation-transformers-pytorch-sagemaker
# to work with an ensemble of teachers


class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature


class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_models=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teachers = teacher_models
        for teacher in self.teachers:
            # place each teacher on same device as student
            self._move_model_to_device(teacher, self.model.device)
            teacher.eval()

    def compute_loss(self, model, inputs, return_outputs=False):
        # compute student output
        outputs_student = model(**inputs)
        student_loss = outputs_student.loss

        # compute teacher output
        with torch.no_grad():
            all_teacher_logits = [] # [llama, gpt]
            for teacher in self.teachers:
                outputs_teacher = teacher(**inputs)
                all_teacher_logits.append(outputs_teacher.logits)
            avg_teacher_logits = torch.stack(all_teacher_logits).mean(dim=0)

        # assert size
        assert outputs_student.logits.size() == avg_teacher_logits.size()

        # Soften probabilities and compute distillation loss
        loss_function = nn.KLDivLoss(reduction="batchmean") #DKL(p,q)
        loss_logits_llama = (
            loss_function(
                F.log_softmax(outputs_student.logits / self.args.temperature, dim=-1), # p
                F.softmax(avg_teacher_logits[0] / self.args.temperature, dim=-1), # q
            )
            * (self.args.temperature ** 2)
        )
        loss_logits_gpt = (
            loss_function(
                F.log_softmax(outputs_student.logits / self.args.temperature, dim=-1), # p
                F.softmax(all_teacher_logits[1] / self.args.temperature, dim=-1), # q
            )
            * (self.args.temperature ** 2)
        )
        # Return weighted student loss
        #print(loss_logits_gpt)
        #print(loss_logits_llama)
        loss_logits = None
        if loss_mode == "weighted":
            gpt_inv_perplexity = 100/((loss_logits_gpt / (self.args.temperature ** 2)).item() ** 2)
            llama_inv_perplexity = 100/((loss_logits_llama / (self.args.temperature ** 2)).item() ** 2)
            total_w = gpt_inv_perplexity + llama_inv_perplexity
            gpt_weight = gpt_inv_perplexity/total_w
            llama_weight = llama_inv_perplexity/total_w

            #print(gpt_weight)
            #print(llama_weight)

            weighted_loss_gpt = torch.mul(loss_logits_gpt,gpt_weight)
            weighted_loss_llama = torch.mul(loss_logits_llama,llama_weight)

            loss_logits = torch.add(weighted_loss_gpt,weighted_loss_llama)
        elif loss_mode == "min":
            loss_logits = torch.minimum(loss_logits_gpt,loss_logits_llama)
        elif loss_mode == "max":
            loss_logits = torch.maximum(loss_logits_gpt,loss_logits_llama)


        loss = self.args.alpha * student_loss + (1.0 - self.args.alpha) * loss_logits
        return (loss, outputs_student) if return_outputs else loss


if wandb_log:
    wandb.login()
    wandb.init(project='babylm', name=MODEL_NAME)





training_args = DistillationTrainingArguments(
    output_dir=MODEL_OUTPUT,
    overwrite_output_dir=True,
    save_strategy = "epoch",
    evaluation_strategy = "epoch",
    num_train_epochs=6,
    gradient_accumulation_steps=1,
    per_device_train_batch_size=BATCH_SIZE,
    save_total_limit=1,  # Set to zero to avoid saving
    report_to="wandb",
    warmup_steps=200, 
    lr_scheduler_type="cosine",
    learning_rate=LR,
    logging_steps=20,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    weight_decay=0.1,
    alpha=ALPHA,
    temperature=TEMPERATURE,
)


trainer = DistillationTrainer(
        student,
        training_args,
        teacher_models=teachers,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,

    )



trainer.train()


trainer.save_model(MODEL_OUTPUT)
tokenizer.save_pretrained(MODEL_OUTPUT)











