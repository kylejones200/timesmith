# Security Policy

## Supported Versions

We actively support the following versions of TimeSmith with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security vulnerability in TimeSmith, please follow these steps:

### 1. **Do NOT** open a public GitHub issue

Security vulnerabilities should be reported privately to prevent exploitation.

### 2. Email the maintainer

Send an email to **kyletjones@gmail.com** with:
- A clear description of the vulnerability
- Steps to reproduce the issue
- Potential impact assessment
- Any suggested fixes (if you have them)

### 3. Response timeline

- **Initial response**: Within 48 hours
- **Status update**: Within 7 days
- **Resolution**: As quickly as possible, depending on severity

### 4. Disclosure policy

- We will acknowledge receipt of your report within 48 hours
- We will provide regular updates on the status of the vulnerability
- Once a fix is ready, we will:
  1. Release a security update
  2. Credit you (if desired) in the release notes
  3. Publish a security advisory on GitHub

### 5. What to report

Please report:
- Remote code execution vulnerabilities
- Authentication/authorization bypasses
- Data exposure or leakage
- Denial of service vulnerabilities
- Injection vulnerabilities (code, SQL, etc.)
- Cryptographic weaknesses
- Any other security-related issues

### 6. What NOT to report

Please do NOT report:
- Issues that require physical access to the system
- Issues that require social engineering
- Issues in optional dependencies (report to those projects instead)
- Issues that require very unlikely user interaction
- Spam or denial of service issues that don't have a security impact

## Security Best Practices

When using TimeSmith in production:

1. **Keep dependencies updated**: Regularly update TimeSmith and its dependencies
   ```bash
   pip install --upgrade timesmith
   ```

2. **Pin versions in production**: Use specific versions in production environments
   ```bash
   pip install timesmith==0.1.1
   ```

3. **Review optional dependencies**: Only install optional dependencies you need
   ```bash
   pip install timesmith[forecasters]  # Only if needed
   ```

4. **Validate inputs**: Always validate and sanitize user inputs before passing to TimeSmith

5. **Monitor for updates**: Subscribe to security advisories on GitHub

## Security Updates

Security updates will be:
- Released as patch versions (e.g., 0.1.1 → 0.1.2)
- Documented in CHANGELOG.md under the "Security" section
- Published as GitHub security advisories
- Announced via release notes

## Acknowledgments

We appreciate the security research community's efforts to keep TimeSmith safe. Security researchers who responsibly disclose vulnerabilities will be credited (if desired) in:
- Release notes
- SECURITY.md acknowledgments
- GitHub security advisories

Thank you for helping keep TimeSmith secure!
