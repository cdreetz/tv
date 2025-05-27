import json
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

class MarketMap:
   def __init__(self):
       self.data_folder = "8d7b3ce6f4596ddf83d6d955017a8210/"
       self.company_llm_summaries = self.data_folder + "company_llm_summaries.json"
       self.market_map_examples = self.data_folder + "market_map_examples.json"
       self.three_d_projection_emb = self.data_folder + "3d_projection_emb.csv"
       with open(self.company_llm_summaries, 'r') as f:
           content = json.load(f)
           self.companies = content
           self.companies_list = list(content.values())
   
   def get_summary(self):
       return self.companies_list[0]['company_summary']
   
   def get_founders(self, text):
       tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
       model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
       nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="max")
       
       results = nlp(text)
       founders = []
       
       for item in results:
           if item['entity_group'] == 'PER':
               name = item['word'].strip()
               if name and len(name) > 1:
                   founders.append(name)
       
       # Remove duplicates - prefer full names over partial names
       unique_founders = []
       founders_sorted = sorted(founders, key=len, reverse=True)
       
       for name in founders_sorted:
           is_subset = False
           for existing in unique_founders:
               if name in existing or existing in name:
                   is_subset = True
                   break
           if not is_subset:
               unique_founders.append(name)
       
       return unique_founders

if __name__ == "__main__":
   mm = MarketMap()
   summary = mm.get_summary()
   founders = mm.get_founders(summary)
   print(founders)
