# LMSYS - Chatbot Arena Human Preference Predictions

### Training

**Model of choice** 

We chose `gemma-2-9b-it` as our solution model, we picked this model because of ease of training with the size of the model and based on our experiment it outperforms llama 3/3.1 8b models.

We then make use of `Gemma2ForSequenceClassification` for win lose and draw 3 class classification，finetune in lora with bf16.

max_length ： 2048

**Lora config**
Below were the config we used to perform lora finetuning on the gemma model.
```python
  freeze_layers: 0
  lora_r: 64
  lora_alpha: 16
  lora_dropout: 0.05
  lora_bias: "none"
  lora_target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
```

**Steps**

1. Step 1：We make use of the orginal data 55k + 33k additional open source data after preprocessing，fold n_splits=20 and only trained with one fold.
2. Step 2：Based on the trained model in step 1， from `ultrafeedback` pseudo label on another 30k data and merged with the original data in step 1（100k+ total），then we train the from from scratch。

Step 1 took around 10 hours, step 2 ~15 hours on 4*A100 40G.
