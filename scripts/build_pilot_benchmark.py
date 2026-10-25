"""Materialize the curated pilot questions against downloaded source pages.

Run from the repository root after processing. Top-level code resolves source
hashes, checks annotated phrases and writes benchmark/pilot.jsonl. It does not
independently review the question labels; importing this script also runs it.
"""

from pathlib import Path

from filings_rag.evaluation import normalize, validate_queries
from filings_rag.storage import read_jsonl, write_jsonl

root = Path('data')
pages = read_jsonl(root / 'pages.jsonl')


def evidence(company, page, phrase):
    """Resolve a company/page quote to a source-hashed evidence record.

    Read the script’s loaded page list, check the normalized phrase and return
    document_id, page and quote. Raise ValueError if the phrase is absent, or
    StopIteration if no page matches. The I/H/M helpers fix the three pilot pages.
    """
    p = next(p for p in pages if p['company_id'] == company and p['page'] == page)
    text = normalize(p['text'])
    wanted = normalize(phrase)
    if wanted not in text:
        raise ValueError(f'Missing source phrase: {company} {page}: {phrase}')
    return {'document_id': p['document_id'], 'page': page, 'quote': phrase}


# Short quote helpers pin the visually inspected Infosys, HDFC Bank and Maruti pages.
I = lambda phrase: evidence('INFY', 77, phrase)
H = lambda phrase: evidence('HDFCBANK', 211, phrase)
M = lambda phrase: evidence('MARUTI', 38, phrase)

specs = [
    (
        'table',
        ['INFY'],
        'What was Infosys consolidated operating margin in FY2025?',
        '21.1%.',
        [[I('Operating margin (%) 22.6 22.3 21.1 20.7')]],
    ),
    (
        'temporal',
        ['INFY'],
        'How did Infosys consolidated operating margin change from FY2024 to FY2025?',
        '20.7% to 21.1%, an increase of 0.4 percentage points.',
        [[I('Operating margin (%) 22.6 22.3 21.1 20.7')]],
    ),
    (
        'table',
        ['INFY'],
        'What were Infosys standalone and consolidated basic EPS for FY2025?',
        'Standalone ₹61.58; consolidated ₹64.50.',
        [[I('Basic EPS (₹)(5) 61.58 65.62 64.50 63.39')]],
    ),
    (
        'exact',
        ['INFY'],
        'How many days of consolidated days sales outstanding did Infosys report for FY2025?',
        '69 days.',
        [[I('(DSO)(1) – – 69 71')]],
    ),
    (
        'semantic',
        ['INFY'],
        'Which standards informed the enterprise risk management framework described by Infosys?',
        'COSO and ISO 31000, tailored to its business requirements.',
        [[I('best practices based on COSO and ISO 31000')]],
    ),
    (
        'semantic',
        ['INFY'],
        'Does Infosys track days sales outstanding at a standalone level?',
        'No; the note says the company does not track DSO at a standalone level.',
        [[I('The Company does not track DSO at a standalone level.')]],
    ),
    (
        'exact',
        ['HDFCBANK'],
        'What net interest income did HDFC Bank disclose for FY2025?',
        '₹1,22,670.1 crore.',
        [[H('to ` 1,22,670.1 crore. Net Interest Margin (NIM) stood 3.48')]],
    ),
    (
        'exact',
        ['HDFCBANK'],
        'What was HDFC Bank net interest margin for FY2025?',
        '3.48%.',
        [[H('Net Interest Margin (NIM) stood 3.48 per cent.')]],
    ),
    (
        'temporal',
        ['HDFCBANK'],
        'What were HDFC Bank net NPA ratios for FY2025 and the preceding year?',
        '0.43% and 0.33%, respectively.',
        [[H('Net NPA ratio stood at 0.43 per cent as against 0.33 per cent in the previous year.')]],
    ),
    (
        'exact',
        ['HDFCBANK'],
        'What was HDFC Bank net profit for FY2025, in crore rupees?',
        '₹67,347.4 crore.',
        [[H('increased by 10.7 per cent to ` 67,347.4 crore from ` 60,812.3 crore.')]],
    ),
    (
        'semantic',
        ['HDFCBANK'],
        'What prior-year item did HDFC Bank identify when explaining the decline in provisions and contingencies?',
        'A ₹10,900.0 crore floating provision created in the previous year.',
        [
            [
                H(
                    'decrease is mainly on account of floating provision created in the previous year of ` 10,900.0 crore.'
                )
            ]
        ],
    ),
    (
        'exact',
        ['HDFCBANK'],
        'What basic earnings per share did HDFC Bank report for FY2025?',
        '₹88.29.',
        [[H('Basic Earnings Per Share (EPS) was ` 88.29 up from ` 85.83.')]],
    ),
    (
        'table',
        ['MARUTI'],
        'What net sales did Maruti Suzuki report in its FY2025 performance snapshot, retaining the reported unit?',
        '₹1,451,152 million.',
        [[M('Net Sales (` Mn) 1,451,152 1,349,378')]],
    ),
    (
        'table',
        ['MARUTI'],
        'How many vehicles did Maruti Suzuki sell in FY2025 and FY2024?',
        '2,234,266 and 2,135,323 vehicles, respectively.',
        [[M('Vehicles Sold 2,234,266 2,135,323')]],
    ),
    (
        'temporal',
        ['MARUTI'],
        'How did the number of Maruti Suzuki sales outlets change between FY2024 and FY2025?',
        'From 3,863 to 4,235, an increase of 372.',
        [[M('Sales Outlets 4,235 3,863')]],
    ),
    (
        'table',
        ['MARUTI'],
        'What was Maruti Suzuki FY2025 profit after tax in the performance snapshot, retaining its unit?',
        '₹139,552 million.',
        [[M('PAT (` Mn) 139,552 132,094')]],
    ),
    (
        'cross_company',
        ['INFY', 'HDFCBANK'],
        'Give FY2025 basic EPS for Infosys on a consolidated basis and for HDFC Bank in its financial performance discussion.',
        'Infosys consolidated: ₹64.50; HDFC Bank: ₹88.29. Preserve the respective reporting scopes.',
        [
            [I('Basic EPS (₹)(5) 61.58 65.62 64.50 63.39')],
            [H('Basic Earnings Per Share (EPS) was ` 88.29 up from ` 85.83.')],
        ],
    ),
    (
        'cross_company',
        ['MARUTI', 'HDFCBANK'],
        'List the FY2025 profit figures disclosed in Maruti Suzuki performance snapshot and HDFC Bank financial performance discussion, retaining each reported unit.',
        'Maruti Suzuki PAT: ₹139,552 million; HDFC Bank net profit: ₹67,347.4 crore. These retain the respective source scopes and units.',
        [
            [M('PAT (` Mn) 139,552 132,094')],
            [H('increased by 10.7 per cent to ` 67,347.4 crore from ` 60,812.3 crore.')],
        ],
    ),
    (
        'unanswerable',
        ['INFY'],
        'What was Infosys actual revenue for FY2028, the year ended March 31, 2028?',
        'Insufficient evidence: FY2028 actuals are outside this corpus.',
        [],
    ),
    (
        'unanswerable',
        ['MARUTI'],
        'What was Maruti Suzuki actual vehicle sales volume for FY2028?',
        'Insufficient evidence: FY2028 actuals are outside this corpus.',
        [],
    ),
]
queries = []
for index, (category, companies, question, reference, groups) in enumerate(specs, 1):
    queries.append(
        {
            'query_id': f'pilot-{index:03d}',
            'question': question,
            'category': category,
            'company_ids': companies,
            'period_end': '2025-03-31',
            'as_of': None,
            'answerable': bool(groups),
            'reference_answer': reference,
            'evidence_groups': groups,
            'split': 'dev',
            'review_status': 'verified',
            'reviewer': 'coding-agent: source text and rendered pages checked; no independent human review',
        }
    )
validate_queries(queries, root)
write_jsonl(Path('benchmark/pilot.jsonl'), queries)
print(f'Wrote {len(queries)} source-checked bootstrap queries')
