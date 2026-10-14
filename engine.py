"""Deterministic bilingual retrieval and evaluation. No external services."""
import math
import re
import unicodedata
from collections import Counter

def tokenize(text):
    text = unicodedata.normalize('NFKC', text).lower()
    text = re.sub('[\u064b-\u065f\u0670\u0640]', '', text)
    text = text.translate(str.maketrans({'أ':'ا','إ':'ا','آ':'ا','ى':'ي'}))
    return re.findall(r'[^\W_]+', text, re.UNICODE)

def chunk_documents(documents, size=90, overlap=20):
    if size <= overlap or overlap < 0:
        raise ValueError('Chunk size must exceed non-negative overlap')
    chunks, seen = [], set()
    for doc in documents:
        if not isinstance(doc.get('id'), str) or not doc['id'] or doc['id'] in seen:
            raise ValueError('Document IDs must be unique non-empty strings')
        if not isinstance(doc.get('text'), str):
            raise ValueError('Document text must be a string')
        seen.add(doc['id'])
        words = doc['text'].split()
        for start in range(0, len(words), size-overlap):
            excerpt = ' '.join(words[start:start+size])
            chunks.append({'id':f"{doc['id']}:{start}", 'source':doc['id'],
                           'title':str(doc.get('title', doc['id'])), 'text':excerpt})
            if start+size >= len(words):
                break
    return chunks

class EvidenceIndex:
    def __init__(self, documents, size=90, overlap=20):
        self.chunks = chunk_documents(documents, size, overlap)
        self.terms = [Counter(tokenize(c['text'])) for c in self.chunks]
        self.average = sum(map(lambda t:sum(t.values()),self.terms))/max(1,len(self.terms))
        self.frequency = Counter(t for counts in self.terms for t in counts)

    def search(self, query, k=3):
        if not isinstance(query,str) or len(query)>2000:
            raise ValueError('Query must be text, at most 2000 characters')
        if not 1 <= k <= 20:
            raise ValueError('k must be between 1 and 20')
        result=[]
        for chunk, counts in zip(self.chunks,self.terms):
            score=0.0
            for term in set(tokenize(query)):
                tf=counts[term]
                if tf:
                    idf=math.log(1+(len(self.terms)-self.frequency[term]+0.5)/(self.frequency[term]+0.5))
                    score += idf * tf * 2.5/(tf+1.5*(0.25+0.75*sum(counts.values())/max(self.average,1)))
            if score>0:
                result.append(dict(chunk,score=round(score,5)))
        return sorted(result,key=lambda r:(-r['score'],r['id']))[:k]

    def answer(self, query, k=3):
        hits=self.search(query,k)
        # Extractive evidence, never generated claims or instructions from documents.
        return {'status':'evidence_found' if hits else 'abstained',
                'answer':'\n\n'.join(f"[{h['id']}] {h['text']}" for h in hits) if hits else 'No matching evidence in this collection.',
                'citations':hits}

def evaluate(index, cases, k=3):
    if not cases:
        raise ValueError('Evaluation set cannot be empty')
    reciprocal, recalls, rows=[],[],[]
    for case in cases:
        relevant=set(case['relevant_sources'])
        if not relevant:
            raise ValueError('Each case needs at least one relevant source')
        # Repeated chunks from one source do not occupy multiple source ranks.
        ranked=list(dict.fromkeys(h['source'] for h in index.search(case['query'],k)))
        recall=len(relevant.intersection(ranked))/len(relevant)
        rr=next((1/(i+1) for i,s in enumerate(ranked) if s in relevant),0)
        recalls.append(recall); reciprocal.append(rr)
        rows.append({'query':case['query'],'sources':ranked,'recall':recall,'reciprocal_rank':rr})
    return {'k':k,'cases':len(cases),'recall_at_k':sum(recalls)/len(cases),
            'mrr':sum(reciprocal)/len(cases),'results':rows}
