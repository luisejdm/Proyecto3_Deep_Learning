"""Response parser — maps raw model output to (kind, payload) tuples."""

import re

def parse_response(text: str):
    first_line = next((line.strip() for line in text.strip().splitlines() if line.strip()), "")

    final_match = re.match(r"FINAL:\s*(.+)", first_line, re.IGNORECASE)
    if final_match:
        return "final", final_match.group(1).strip()

    action_match = re.match(r"ACTION:\s*(\w+)\(([^)]*)\)", first_line, re.IGNORECASE)
    if action_match:
        tool_name = action_match.group(1).strip()
        raw_args = [a.strip() for a in action_match.group(2).split(",") if a.strip()]
        args = []
        for a in raw_args:
            if "=" in a:
                a = a.split("=", 1)[1].strip()
            a = a.strip("\"'")
            try:
                args.append(float(a))
            except ValueError:
                args.append(a)
        return "action", (tool_name, args)

    return "unknown", first_line
