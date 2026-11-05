"""Repeatable static interface checks; not a substitute for participant testing."""
import collections,json
from pathlib import Path
from bs4 import BeautifulSoup
root=Path(__file__).resolve().parents[1]
html=(root/'web/templates/index.html').read_text(encoding='utf-8')
soup=BeautifulSoup(html,'html.parser')
ids=[x.get('id') for x in soup.select('[id]')]
duplicates={k:v for k,v in collections.Counter(ids).items() if v>1}
labels={x.get('for') for x in soup.select('label[for]')}
missing=[]
for el in soup.select('input,select,textarea'):
 if el.name=='input' and el.get('type','text') in ('hidden','submit','button','reset'):continue
 label=el.get('aria-label') or el.get('aria-labelledby') or el.find_parent('label') or el.get('id') in labels
 if not label:missing.append({'tag':el.name,'id':el.get('id'),'type':el.get('type')})
buttons=soup.select('button')
empty=[]
for el in buttons:
 if not el.get_text(' ',strip=True) and not el.get('aria-label') and not el.get('aria-labelledby') and not el.get('title'):
  empty.append({'id':el.get('id')})
result={'kind':'static HTML heuristic review only; no browser state, assistive technology or human participants','duplicate_ids':duplicates,'controls_without_explicit_label':missing,'buttons_without_text_or_accessible_name':empty,'controls_checked':len(soup.select('input,select,textarea')),'buttons_checked':len(buttons),'limitations':['Scripts can dynamically add or change elements.','An accessible name does not prove usability or accessibility conformance.','Placeholders are not counted as explicit labels.']}
out=root/'artifacts/training_candidate_20261102/usability_static_audit.json';out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
