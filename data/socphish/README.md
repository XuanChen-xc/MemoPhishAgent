# SocPhish Public Release

This folder contains the public-use SocPhish release package derived from the raw crawl artifacts.

## Files

- `socphish_public.csv`: tabular release
- `socphish_public.jsonl`: JSON release
- `socphish_public_urls.txt`: one URL per line for direct execution with the MemoPhishAgent scripts
- `socphish_public_benign.txt`: benign URLs only, one per line
- `socphish_public_malicious.txt`: malicious URLs only, one per line
- `stats.json`: release statistics

## Schema

- `id`: stable record ID in the form `benign_site_123` or `malicious_site_456`
- `label`: `benign` or `malicious`
- `site_index`: original site index within the raw crawl folder
- `url`: sanitized public-release URL
- `domain`: normalized hostname extracted from `url`
- `is_shortener`: whether the hostname matches a known URL-shortener domain
- `had_query_params`: whether the original URL had query parameters
- `query_redacted`: whether the public URL had query parameters removed or sanitized

## Statistics

- Total records: `753`
- Benign records: `313`
- Malicious records: `440`
- Unique domains: `316`
- Records with query parameters: `63`
- Records with sanitized query parameters: `31`
- Records on shortener domains: `53`
- Duplicate sanitized URLs preserved in release: `6`
