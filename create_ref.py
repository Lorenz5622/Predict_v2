#!/usr/bin/env python3
import re
import sys
import json
import time
import html
from pathlib import Path
from typing import Optional, Tuple, List

import requests
import xml.etree.ElementTree as ET

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36 bib-fetcher/1.0"
HEADERS = {"User-Agent": UA}
ARXIV_NS = {'atom': 'http://www.w3.org/2005/Atom'}


def normalize_space(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip()


def parse_ref_line(line: str) -> Optional[dict]:
    line = line.strip()
    if not line:
        return None
    m = re.match(r'^\[(\d+)\]\s*(.+)$', line)
    if not m:
        return None
    num = m.group(1)
    body = m.group(2).strip()

    arxiv_id = None
    m_arxiv = re.search(r'arXiv\s*[:：]?\s*(\d{4}\.\d{4,5})(?:v\d+)?', body, re.I)
    if m_arxiv:
        arxiv_id = m_arxiv.group(1)

    title = None
    # 优先抓题名：通常位于 authors. title[J/C]. venue...
    parts = re.split(r'\.\s+', body, maxsplit=2)
    if len(parts) >= 2:
        title_candidate = parts[1]
        title_candidate = re.sub(r'\[[JCMR]\]$', '', title_candidate).strip()
        title_candidate = re.sub(r'\[[^\]]+\]', '', title_candidate).strip()
        title = normalize_space(title_candidate)

    if not title:
        m_title = re.search(r'\.\s*(.*?)\s*\[[A-Z/]+\]', body)
        if m_title:
            title = normalize_space(m_title.group(1))

    year = None
    y = re.search(r'(19|20)\d{2}', body)
    if y:
        year = y.group(0)

    return {
        'num': num,
        'raw': body,
        'title': title,
        'arxiv_id': arxiv_id,
        'year': year,
    }


def doi_to_bibtex(doi: str) -> Optional[str]:
    url = f'https://doi.org/{doi}'
    headers = dict(HEADERS)
    headers['Accept'] = 'application/x-bibtex'
    r = requests.get(url, headers=headers, timeout=20, allow_redirects=True)
    if r.ok and '@' in r.text:
        return r.text.strip()
    return None


def search_crossref(ref: dict) -> Optional[Tuple[str, str]]:
    query_title = ref.get('title') or ref['raw']
    params = {
        'query.title': query_title,
        'rows': 5,
        'select': 'DOI,title,published-print,published-online,created'
    }
    r = requests.get('https://api.crossref.org/works', params=params, headers=HEADERS, timeout=20)
    if not r.ok:
        return None
    items = r.json().get('message', {}).get('items', [])
    want = normalize_space((ref.get('title') or '').lower())
    for it in items:
        titles = it.get('title') or []
        title = normalize_space((titles[0] if titles else '').lower())
        if want and title and (want == title or want in title or title in want):
            doi = it.get('DOI')
            if doi:
                bib = doi_to_bibtex(doi)
                if bib:
                    return bib, f'crossref-doi:{doi}'
    # fallback first item with DOI
    for it in items:
        doi = it.get('DOI')
        if doi:
            bib = doi_to_bibtex(doi)
            if bib:
                return bib, f'crossref-doi:{doi}'
    return None


def search_arxiv_api(arxiv_id: str) -> Optional[str]:
    params = {'search_query': f'id:{arxiv_id}', 'start': 0, 'max_results': 1}
    r = requests.get('http://export.arxiv.org/api/query', params=params, headers=HEADERS, timeout=20)
    if not r.ok:
        return None
    root = ET.fromstring(r.text)
    entries = root.findall('atom:entry', ARXIV_NS)
    if not entries:
        return None
    e = entries[0]
    title = normalize_space(e.findtext('atom:title', default='', namespaces=ARXIV_NS))
    published = e.findtext('atom:published', default='', namespaces=ARXIV_NS)
    year = published[:4] if published else ''
    authors = [a.findtext('atom:name', default='', namespaces=ARXIV_NS) for a in e.findall('atom:author', ARXIV_NS)]
    author_field = ' and '.join(authors)
    key = f'arxiv:{arxiv_id}'
    bib = f'''@article{{{key},\n  title={{{title}}},\n  author={{{author_field}}},\n  journal={{arXiv preprint arXiv:{arxiv_id}}},\n  year={{{year}}},\n  url={{https://arxiv.org/abs/{arxiv_id}}}\n}}'''
    return bib


def search_dblp(ref: dict) -> Optional[Tuple[str, str]]:
    q = ref.get('title') or ref['raw']
    r = requests.get('https://dblp.org/search/publ/api', params={'q': q, 'h': 5, 'format': 'json'}, headers=HEADERS, timeout=20)
    if not r.ok:
        return None
    hits = r.json().get('result', {}).get('hits', {}).get('hit', [])
    if isinstance(hits, dict):
        hits = [hits]
    want = normalize_space((ref.get('title') or '').lower())
    for hit in hits:
        info = hit.get('info', {})
        title = normalize_space(str(info.get('title', '')).lower())
        biburl = info.get('bibtex')
        if biburl and title and (want == title or want in title or title in want):
            rr = requests.get(biburl, headers=HEADERS, timeout=20)
            if rr.ok and '@' in rr.text:
                return rr.text.strip(), f'dblp:{biburl}'
    for hit in hits:
        info = hit.get('info', {})
        biburl = info.get('bibtex')
        if biburl:
            rr = requests.get(biburl, headers=HEADERS, timeout=20)
            if rr.ok and '@' in rr.text:
                return rr.text.strip(), f'dblp:{biburl}'
    return None


def replace_bibtex_key(bib: str, new_key: str) -> str:
    return re.sub(r'(@\w+\s*\{)\s*([^,]+)', rf'\1{new_key}', bib, count=1)


def process_file(infile: Path, outfile: Path, reportfile: Path):
    text = infile.read_text(encoding='utf-8')
    refs = []
    for line in text.splitlines():
        item = parse_ref_line(line)
        if item:
            refs.append(item)

    out_entries: List[str] = []
    report = []

    for ref in refs:
        num = ref['num']
        key = f'[{num}]'
        bib = None
        source = None
        errors = []

        try:
            if ref.get('arxiv_id'):
                bib = search_arxiv_api(ref['arxiv_id'])
                source = f"arxiv:{ref['arxiv_id']}"
        except Exception as e:
            errors.append(f'arxiv failed: {e}')

        if not bib:
            try:
                got = search_crossref(ref)
                if got:
                    bib, source = got
            except Exception as e:
                errors.append(f'crossref failed: {e}')

        if not bib:
            try:
                got = search_dblp(ref)
                if got:
                    bib, source = got
            except Exception as e:
                errors.append(f'dblp failed: {e}')

        if bib:
            bib = replace_bibtex_key(bib, key)
            out_entries.append(bib)
            report.append({
                'num': num,
                'title': ref.get('title'),
                'status': 'ok',
                'source': source,
            })
        else:
            report.append({
                'num': num,
                'title': ref.get('title'),
                'status': 'failed',
                'raw': ref['raw'],
                'errors': errors,
            })
        time.sleep(0.5)

    outfile.write_text('\n\n'.join(out_entries) + ('\n' if out_entries else ''), encoding='utf-8')
    reportfile.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python fetch_bibtex_by_refs.py input.txt [output.bib]')
        sys.exit(1)
    infile = Path(sys.argv[1])
    outfile = Path(sys.argv[2]) if len(sys.argv) >= 3 else infile.with_suffix('.bib')
    reportfile = outfile.with_suffix('.report.json')
    process_file(infile, outfile, reportfile)
    print(f'Wrote BibTeX to: {outfile}')
    print(f'Wrote report to: {reportfile}')
