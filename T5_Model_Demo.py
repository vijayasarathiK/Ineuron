!pip install transformers --upgrade

from transformers import T5Tokenizer, T5ForConditionalGeneration
import time

model = T5ForConditionalGeneration.from_pretrained('t5-small')  #tiny-base
tokenizer = T5Tokenizer.from_pretrained('t5-small')  #tiny-base

start_time = time.time()
text = input("Enter: ")

def summarization_infer(text, max=50):
  preprocess_text = text.replace("\n", " ").strip()
  t5_prepared_Text = "summarize: "+preprocess_text
  tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt")

  summary_ids = model.generate(tokenized_text, min_length=30, max_length=max, top_k=100, top_p=0.8) #top-k top-p sampling strategy
  output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
  end_time = time.time()
  print (f'Time taken : {end_time-start_time}')
  return output

def translation_infer(text, max=50):
  preprocess_text = text.replace("\n", " ").strip()
  t5_prepared_Text = "translate English to German: "+preprocess_text
  tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt")

  translation_ids = model.generate(tokenized_text, min_length=10, max_length=50, early_stopping=True, num_beams=2)
  output = tokenizer.decode(translation_ids[0], skip_special_tokens=True)
  end_time = time.time()
  print (f'Time taken : {end_time-start_time}')
  return output

def grammatical_acceptibility_infer(text):
  preprocess_text = text.replace("\n", " ").strip()
  t5_prepared_Text = "cola sentence: "+preprocess_text
  tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt")

  grammar_ids = model.generate(tokenized_text, min_length=1, max_length=3)
  output = tokenizer.decode(grammar_ids[0], skip_special_tokens=True)
  end_time = time.time()
  print (f'Time taken : {end_time-start_time}')
  return output



from transformers import pipeline

summarization_pipeline = pipeline(task='summarization', model="t5-small") 
output = summarization_pipeline(text, min_length=30, max_length=60, top_k=100, top_p=0.8)
print(output)

from flask import Flask, request
from flask_ngrok import run_with_ngrok

app = Flask(__name__)


@app.route('/infer', methods=['POST'])
def infer():
  args = request.args['task']
  text = request.args['text']
  if args=='summarize':
    return summarization_infer(text)
  elif args=='translation':
    return translation_infer(text)
  else:
    return grammatical_acceptibility_infer(text)

if __name__=='__main__':
  app.run()

import pickle

filename = 'model.sav'
pickle.dump(summarization_pipeline, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))
Summarize = summarization_infer(text)
Translate = translation_infer(text)

print(Summarize)
print(Translate)
