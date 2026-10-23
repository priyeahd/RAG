#!/usr/bin/env python3
"""Step 3C: blind cross-event aggregation of Step 3B passage evidence.

No access to ground-truth poisoning labels; no inference of RAG causality.
Run from repository root:
  python CrosseventRAG/step3c_cross_event_rank.py
"""
import argparse
import csv
import math
from collections import defaultdict, Counter
from pathlib import Path

PUSH = 'pushes_wrong'


def read_csv(path, required):
    if not path.is_file():
        raise FileNotFoundError(f'Missing input: {path}')
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'{path.name}: missing columns {sorted(missing)}')
        return list(reader)


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results-dir', type=Path, default=Path(__file__).resolve().parent / 'results')
    ap.add_argument('--model-suffix', default='qwen3.5_9b')
    ap.add_argument('--min-events', type=int, default=10, help='Minimum distinct retrieved events required for eligible ranking')
    args = ap.parse_args()
    root = args.results_dir
    suffix = args.model_suffix
    ev = read_csv(root / f'step3b_v3_3_events_{suffix}.csv', ['event_id','verdict','suspect_documents','wrong_answer'])
    pa = read_csv(root / f'step3b_v3_3_passages_{suffix}.csv', ['event_id','document_id','context_position','question','wrong_answer','final_label','push_type','needs_review'])
    ed = read_csv(root / f'step3b_v3_3_documents_{suffix}.csv', ['event_id','document_id','passages','pushing_passages','instruction_passages'])

    ev_by_id = {r['event_id']: r for r in ev}
    if len(ev_by_id) != len(ev):
        raise ValueError('Duplicate event rows')
    pa_by_ev = defaultdict(list)
    pa_by_pair = defaultdict(list)
    for r in pa:
        if r['event_id'] not in ev_by_id:
            raise ValueError(f"Passage event missing from event CSV: {r['event_id']}")
        pa_by_ev[r['event_id']].append(r)
        pa_by_pair[(r['event_id'], r['document_id'])].append(r)
    pair_keys = {(r['event_id'],r['document_id']) for r in ed}
    if len(pair_keys) != len(ed) or pair_keys != set(pa_by_pair):
        raise ValueError('Step 3B document rows do not match passage event-document pairs')
    for r in ed:
        rows = pa_by_pair[(r['event_id'],r['document_id'])]
        pushes = [x for x in rows if x['final_label'] == PUSH]
        instructions = [x for x in pushes if x['push_type'] == 'instruction']
        if (int(r['passages']),int(r['pushing_passages']),int(r['instruction_passages'])) != (len(rows),len(pushes),len(instructions)):
            raise ValueError(f"Mismatch for {r['event_id']} / {r['document_id']}")
    for event_id, r in ev_by_id.items():
        rows = pa_by_ev[event_id]
        if len(rows) != 10:
            raise ValueError(f'{event_id}: expected 10 passages, got {len(rows)}')
        pushed_docs = {x['document_id'] for x in rows if x['final_label'] == PUSH}
        saved_docs = set(filter(None, r['suspect_documents'].split(';')))
        if pushed_docs != saved_docs:
            raise ValueError(f'{event_id}: event suspect docs disagree with passage labels')
        if r['verdict'] == 'wrong_answer_from_context' and not pushed_docs:
            raise ValueError(f'{event_id}: context verdict without supporting passages')
        if r['verdict'] == 'llm_own_mistake' and pushed_docs:
            raise ValueError(f'{event_id}: LLM verdict with supporting passage')

    records = defaultdict(lambda: {'retrieved_events':set(), 'flagged_events':set(),
         'instruction_events':set(), 'claim_events':set(), 'questions':set(),
         'answers':set(), 'passages':0,'pushing_passages':0,'instruction_passages':0,
         'claim_passages':0, 'needs_review':0})
    details = []
    for (event_id, doc), rows in sorted(pa_by_pair.items()):
        d = records[doc]
        d['retrieved_events'].add(event_id)
        d['passages'] += len(rows)
        pushes = [x for x in rows if x['final_label'] == PUSH]
        instr = [x for x in pushes if x['push_type']=='instruction']
        claims = [x for x in pushes if x['push_type']=='claim']
        reviews = [x for x in rows if x['needs_review'].strip().lower() in ('1','true','yes')]
        d['pushing_passages'] += len(pushes)
        d['instruction_passages'] += len(instr)
        d['claim_passages'] += len(claims)
        d['needs_review'] += len(reviews)
        if pushes:
            d['flagged_events'].add(event_id)
            if instr: d['instruction_events'].add(event_id)
            if claims: d['claim_events'].add(event_id)
            d['questions'].add(rows[0]['question'])
            d['answers'].add(rows[0]['wrong_answer'])
        details.append({'event_id':event_id,'document_id':doc,'retrieved_passages':len(rows),
            'pushing_passages':len(pushes),'instruction_passages':len(instr),
            'claim_passages':len(claims),'flagged':int(bool(pushes)),
            'wrong_answer':rows[0]['wrong_answer'],'needs_review_passages':len(reviews)})

    out = []
    for doc, d in records.items():
        n = len(d['retrieved_events'])
        k = len(d['flagged_events'])
        p = k/n if n else 0
        # Wilson lower 95% confidence bound: conservative, exposure-aware ranking.
        z = 1.96
        den = 1 + z*z/n
        lower = (p + z*z/(2*n) - z*math.sqrt((p*(1-p) + z*z/(4*n))/n))/den if n else 0
        out.append({'document_id':doc, 'retrieved_events':n, 'flagged_events':k,
           'flag_rate':round(p,6), 'wilson_lower_95':round(lower,6),
           'instruction_events':len(d['instruction_events']),
           'claim_events':len(d['claim_events']),
           'distinct_flagged_questions':len(d['questions']),
           'distinct_flagged_wrong_answers':len(d['answers']),
           'retrieved_passages':d['passages'],
           'pushing_passages':d['pushing_passages'],
           'instruction_passages':d['instruction_passages'],
           'claim_passages':d['claim_passages'],
           'needs_review_passages':d['needs_review'],
           'ranking_eligible':int(n >= args.min_events)})
    out.sort(key=lambda r: (r['ranking_eligible'],r['wilson_lower_95'],r['instruction_events'],r['flagged_events']),reverse=True)
    for idx, row in enumerate(out,1):
        row['rank'] = idx if row['ranking_eligible'] else ''

    rank_fields = ['rank','document_id','retrieved_events','flagged_events','flag_rate','wilson_lower_95',
      'instruction_events','claim_events','distinct_flagged_questions', 'distinct_flagged_wrong_answers',
      'retrieved_passages','pushing_passages','instruction_passages','claim_passages',
      'needs_review_passages','ranking_eligible']
    rank_path = root / f'step3c_document_ranking_{suffix}.csv'
    pair_path = root / f'step3c_event_document_evidence_{suffix}.csv'
    report_path = root / f'step3c_summary_{suffix}.txt'
    write_csv(rank_path, out, rank_fields)
    write_csv(pair_path, details, list(details[0]) if details else [])
    verdicts = Counter(x['verdict'] for x in ev)
    claims = sum(x['claim_passages'] for x in out)
    instr = sum(x['instruction_passages'] for x in out)
    lines = [
      'STEP 3C — BLIND CROSS-EVENT RANKING',
      f'Events: {len(ev)}',f'Passages: {len(pa)}',f'Event-document pairs: {len(ed)}',
      f'Documents: {len(out)}', f'Event verdicts: {dict(verdicts)}',
      f'Passage labels: instruction={instr}, claim={claims}',
      'Ranking: Wilson lower 95% bound on flagged-event rate, minimum retrieved events '+str(args.min_events),
      'WARNING: Rankings reflect association, not verified poisoning or causal responsibility.',
      'WARNING: Only candidate-error events supplied to Step 3B; for stronger comparisons use all 1,000 events.',
      'WARNING: Some answer mismatches are acceptable; manually audit before claims.',
      'TOP DOCUMENTS:',
    ]
    for r in out[:10]:
        lines.append(f"{str(r['rank']):>3} {r['document_id']}: flagged {r['flagged_events']}/{r['retrieved_events']} ({r['flag_rate']:.1%}); instructions={r['instruction_events']}; distinct wrong answers={r['distinct_flagged_wrong_answers']}")
    report_path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))
    print(f'\nCreated:\n{rank_path}\n{pair_path}\n{report_path}')

if __name__ == '__main__':
    main()
