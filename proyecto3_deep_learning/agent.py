from __future__ import annotations

from dataclasses import dataclass

from model import ModelRunner
from parsing import parse_response
from tools import ToolRegistry
import time


@dataclass
class Agent:
    runner: ModelRunner
    tools: ToolRegistry
    system_prompt: str

    def run(
        self,
        task: str,
        max_steps: int = 5,
        temperature: float = 0.3,
        verbose: bool = False,
    ) -> tuple[str | None, list[dict]]:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]
        trace: list[dict] = []

        for step in range(1, max_steps + 1):
            time.sleep(1.5)
            response = self.runner.generate(messages, temperature=temperature)
            if verbose:
                print(f"[step {step}] {response}")
            messages.append({"role": "assistant", "content": response})

            kind, payload = parse_response(response)

            if kind == "final":
                tool_succeeded = any(
                    entry.get("type") == "action" for entry in trace
                )
                if not tool_succeeded:
                    trace.append({
                        "step": step,
                        "type": "blocked_final",
                        "content": payload,
                    })
                    messages.append({
                        "role": "user",
                        "content": (
                            "You cannot give a FINAL answer without first calling a tool. "
                            "Call the appropriate tool first before answering."
                        ),
                    })
                    continue
                trace.append({"step": step, "type": "final", "content": payload})
                return payload, trace

            if kind == "action":
                name, args = payload
                try:
                    result = self.tools.execute(name, *args)
                    result_text = (
                        f"${float(result):.2f}"
                        if isinstance(result, (int, float))
                        else str(result)
                    )
                    observation = f"Tool result: {result_text}"
                    trace.append({
                        "step": step,
                        "type": "action",
                        "tool": name,
                        "args": args,
                        "result": result_text,
                    })
                except KeyError as error:
                    observation = f"Error: {error}"
                    trace.append({
                        "step": step,
                        "type": "error",
                        "tool": name,
                        "args": args,
                        "error": str(error),
                    })
                except Exception as error:
                    observation = f"Error: {error}"
                    trace.append({
                        "step": step,
                        "type": "error",
                        "tool": name,
                        "args": args,
                        "error": str(error),
                    })
                messages.append({"role": "user", "content": observation})
            else:
                trace.append({"step": step, "type": "unknown", "raw": payload})
                messages.append({
                    "role": "user",
                    "content": (
                        "Invalid format. Respond with exactly one line starting with "
                        "ACTION: tool_name(ARGS) or FINAL: <answer>."
                    ),
                })

        return None, trace