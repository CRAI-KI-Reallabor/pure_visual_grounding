import json
import re

def parse_generated_report(report_str):
    """
    Robustly parses a stringified JSON report (from LLM) into a Python dict.

    Handles:
    - Output as dict (returns directly)
    - JSON as a string (even with formatting artifacts or code blocks)
    - Markdown (```json ... ```)
    - Leading/trailing/embedded non-JSON text
    - Unicode BOM, stray characters, etc.

    Returns:
        dict if parsed, else str (raw output)
    """

    if isinstance(report_str, dict):
        return report_str


    if not isinstance(report_str, str):
        print("[Warning] parse_generated_report: input not string or dict, returning as is.")
        return report_str


    report_str = report_str.encode("utf-8", "ignore").decode("utf-8-sig")
    report_str = ''.join([c for c in report_str if c.isprintable() or c in '\n\r\t'])


    try:
        return json.loads(report_str)
    except Exception:
        pass

    # Try extracting from markdown code block (```json ... ```)
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", report_str, re.IGNORECASE)
    if match:
        block = match.group(1)
        try:
            return json.loads(block)
        except Exception:
            pass

    # Try extracting from any code block (``` ... ```)
    match = re.search(r"```\s*([\s\S]*?)```", report_str)
    if match:
        block = match.group(1)
        try:
            return json.loads(block)
        except Exception:
            pass


    start = report_str.find('{')
    end = report_str.rfind('}')
    if start != -1 and end != -1 and start < end:
        json_candidate = report_str[start:end+1]
        try:
            return json.loads(json_candidate)
        except Exception:
            pass

    # If all parsing attempts fail, return the original string and log warning
    print("[Warning] Could not parse generated report as JSON. Returning raw string.")
    return report_str