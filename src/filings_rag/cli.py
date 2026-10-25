"""Command-line entry point for the filings RAG pipeline.

The filings console command loads .env, parses arguments, dispatches work to the
relevant module and prints results. Importing this module does not run commands.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from .storage import read_jsonl


def emit(value):
    """Print a value as indented JSON while retaining readable Unicode characters."""
    print(json.dumps(value, indent=2, ensure_ascii=False))


def main():
    """Load .env, parse command-line arguments and dispatch the selected operation.

    Reads sys.argv through argparse. Commands cover download, process, index,
    search, ask, evaluate, stats, page, discover and vision-page. Print results
    or progress and convert handled operational errors into exit status 1.
    The installed filings entry point and python -m filings_rag.cli call this.
    """
    load_dotenv()
    parser = argparse.ArgumentParser(description='Indian filing ingestion, retrieval and evaluation')
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    commands = parser.add_subparsers(dest='command', required=True)
    download = commands.add_parser('download', help='Download exchange attachments from a source manifest')
    download.add_argument('--config', type=Path, default=Path('configs/pilot.json'))
    process = commands.add_parser('process', help='Extract page text and tables; generate chunks')
    process.add_argument('--max-pages', type=int, help='Smoke-test only; processes a prefix of each report')
    process.add_argument('--repairs', type=Path, help='Reviewed vision-page proposals in JSONL')
    vision = commands.add_parser(
        'vision-page', help='Send one page to a vision API; save unreviewed proposal'
    )
    vision.add_argument('company')
    vision.add_argument('number', type=int)
    vision.add_argument('--output', type=Path, default=Path('data/vision_proposals.jsonl'))
    commands.add_parser('index', help='Embed chunks through configured API and save dense index')
    commands.add_parser('stats', help='Show coverage and extraction flags')
    discover = commands.add_parser('discover', help='List PDF links from saved exchange HTML')
    discover.add_argument('html', type=Path)
    discover.add_argument('--base-url', required=True)
    page = commands.add_parser('page', help='Inspect extracted text and optionally render a source page')
    page.add_argument('company')
    page.add_argument('number', type=int)
    page.add_argument('--render', type=Path)
    for name in ('search', 'ask'):
        cmd = commands.add_parser(name)
        cmd.add_argument('question')
        cmd.add_argument('--mode', choices=['bm25', 'dense', 'hybrid'], default='bm25')
        cmd.add_argument('--company', action='append', help='Repeat for multiple company IDs')
        cmd.add_argument('--period-end', help='YYYY-MM-DD')
        cmd.add_argument('--as-of', help='YYYY-MM-DD; excludes unknown publication dates')
        cmd.add_argument('-k', type=int, default=10)
    evaluate = commands.add_parser('evaluate')
    evaluate.add_argument('--benchmark', type=Path, default=Path('benchmark/pilot.jsonl'))
    evaluate.add_argument('--output', type=Path, default=Path('experiments/pilot'))
    evaluate.add_argument('--modes', nargs='+', choices=['bm25', 'dense', 'hybrid'], default=['bm25'])
    evaluate.add_argument('--split', choices=['dev', 'test'], default='dev')
    args = parser.parse_args()
    root = args.data_dir
    try:
        if args.command == 'download':
            from .ingestion import download

            download(args.config, root)
        elif args.command == 'process':
            from .processing import process

            if args.max_pages is not None and args.max_pages < 1:
                raise ValueError('--max-pages must be positive')
            process(root, args.max_pages, args.repairs)
        elif args.command == 'vision-page':
            from .vision import extract_page

            emit(extract_page(root, args.company, args.number, args.output))
        elif args.command == 'index':
            from .retrieval import build_dense

            emit(build_dense(root))
        elif args.command == 'discover':
            from .ingestion import attachment_links

            emit(attachment_links(args.html.read_text(), args.base_url))
        elif args.command == 'stats':
            pages = read_jsonl(root / 'pages.jsonl')
            emit(
                {
                    'filings': len(read_jsonl(root / 'manifest.jsonl')),
                    'pages': len(pages),
                    'chunks': len(read_jsonl(root / 'chunks.jsonl')),
                    'pages_by_company': dict(Counter(p['company_id'] for p in pages)),
                    'flags': dict(Counter(flag for p in pages for flag in p['flags'])),
                    'partial_documents': sorted(
                        {p['company_id'] for p in pages if p['settings']['max_pages'] is not None}
                    ),
                    'dense_index_exists': (root / 'index' / 'metadata.json').exists(),
                }
            )
        elif args.command == 'page':
            matches = [
                p
                for p in read_jsonl(root / 'pages.jsonl')
                if p['company_id'] == args.company and p['page'] == args.number
            ]
            if len(matches) != 1:
                raise ValueError('Expected one page; check company, page number, and document versions')
            emit(matches[0])
            if args.render:
                import pdfplumber

                record = next(
                    r
                    for r in read_jsonl(root / 'manifest.jsonl')
                    if r['document_id'] == matches[0]['document_id']
                )
                args.render.parent.mkdir(parents=True, exist_ok=True)
                with pdfplumber.open(root / record['local_path']) as pdf:
                    pdf.pages[args.number - 1].to_image(resolution=120).save(args.render)
        elif args.command in {'search', 'ask'}:
            from datetime import date

            from .retrieval import Retriever

            for value in (args.period_end, args.as_of):
                if value:
                    date.fromisoformat(value)
            hits = Retriever(root).search(
                args.question, args.mode, args.k, args.company, args.period_end, args.as_of
            )
            if args.command == 'ask':
                from .generation import answer

                emit(answer(args.question, hits))
            else:
                emit(hits)
        elif args.command == 'evaluate':
            from .evaluation import evaluate

            emit(evaluate(root, args.benchmark, args.output, args.modes, args.split))
    except (ValueError, RuntimeError, OSError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
