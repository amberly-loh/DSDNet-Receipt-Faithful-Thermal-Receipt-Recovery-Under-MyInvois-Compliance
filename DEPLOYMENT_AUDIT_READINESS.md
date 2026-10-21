# Audit Deployment Readiness

## Supported deployment boundary

This application is a local/private receipt recovery, review and audit-export tool. It is not an authorised MyInvois submission client and does not create a compliant e-Invoice, digital signature, taxpayer validation or submission identifier.

## Production launch

1. Install `requirements.txt` in the project virtual environment.
2. Keep `RECEIPT_HOST=127.0.0.1` for a single-device installation. Bind to a private network interface only for an explicitly controlled demonstration network.
3. For public or organisational deployment, place the app behind an HTTPS reverse proxy, set `RECEIPT_HTTPS=1`, and configure Google OAuth with the final HTTPS redirect URI.
4. Run `python scripts/serve_production.py`. Do not use Flask's development server for deployment.
5. Back up `artifacts/app_data/receipts.sqlite3`, `.session_key`, and `artifacts/user_runs/` together. Restrict their operating-system permissions to the service account and authorised backup operators.
6. Verify `/api/health` reports `status: ok` and `audit_chain: verified` before accepting uploads.

## Evidence controls implemented

- Source and recovered files for new records have SHA-256 hashes stored in the processing audit.
- Receipt detail pages verify stored files and display verified, failed or legacy-unverified status.
- Application audit events form a SHA-256 chain; `/api/health` fails when the chain is altered.
- Original OCR, language suggestions and human-confirmed fields remain distinct.
- Receipt ownership, administrator authorisation, CSRF protection, session protection, upload limits, response security headers and spreadsheet formula neutralisation are tested.

## Operational requirements outside the code

- Define authorised users, administrators, retention owner, backup owner and incident-response contact.
- Use encrypted disks and encrypted backups; test restoration regularly.
- Document retention and legal-hold rules before deleting records.
- Review failed integrity checks as incidents. Do not repair or overwrite the affected evidence before preserving a forensic copy.
- Perform vulnerability scanning and an independent security review before Internet exposure.
- Complete held-out restoration/OCR evaluation before claiming research superiority or unattended field acceptance.
