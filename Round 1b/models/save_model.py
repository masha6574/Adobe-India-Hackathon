# In your save_model.py or a new script
from transformers import T5ForConditionalGeneration, T5Tokenizer

model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Save them to your models folder
tokenizer.save_pretrained("./models/t5-small-tokenizer")
model.save_pretrained("./models/t5-small-model")

print("T5-small model and tokenizer saved.")
