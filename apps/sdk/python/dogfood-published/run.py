"""Published-package E2E for morse-ai (PyPI).

Mirrors the TS dogfood scripts: anthropic / openai / langgraph, each a
single real LLM call inside a morse_ai.run(), sent to Morse dev.

Usage: python run.py [anthropic|openai|langgraph|all]
Env:   MORSE_API_KEY, MORSE_ENDPOINT, ANTHROPIC_API_KEY, OPENAI_API_KEY
"""

import os
import sys
import time

import morse_ai

ENDPOINT = os.environ.get(
    "MORSE_ENDPOINT", "https://dev-api.morsehq.dev/api/v1/traces/batch"
)


def require(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        sys.exit(f"Missing env var: {name}")
    return v


def scenario_anthropic() -> None:
    require("ANTHROPIC_API_KEY")
    import anthropic

    from morse_ai.adapters import anthropic as mv_anthropic

    client = mv_anthropic.wrap(anthropic.Anthropic())
    with morse_ai.run("pub-e2e-py-anthropic") as r:
        res = client.messages.create(
            model=os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5"),
            max_tokens=128,
            messages=[
                {
                    "role": "user",
                    "content": "Name three reasons to adopt observability. Be terse.",
                }
            ],
        )
        text = "".join(b.text for b in res.content if getattr(b, "type", "") == "text")
        print(f"  reply: {text[:100]!r}")
        r.set_outcome(True, "completed")


def scenario_openai() -> None:
    require("OPENAI_API_KEY")
    import openai

    from morse_ai.adapters import openai as mv_openai

    client = mv_openai.wrap(openai.OpenAI())
    with morse_ai.run("pub-e2e-py-openai") as r:
        res = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {
                    "role": "user",
                    "content": "Name three reasons to adopt observability. Be terse.",
                }
            ],
        )
        print(f"  reply: {str(res.choices[0].message.content)[:100]!r}")
        r.set_outcome(True, "completed")


def scenario_langgraph() -> None:
    require("OPENAI_API_KEY")
    import operator
    from typing import Annotated

    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, START, StateGraph
    from typing_extensions import TypedDict

    class State(TypedDict):
        messages: Annotated[list, operator.add]

    llm = ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        max_tokens=80,
    )

    def call_model(state: State) -> dict:
        return {"messages": [llm.invoke(state["messages"])]}

    graph = StateGraph(State)
    graph.add_node("model", call_model)
    graph.add_edge(START, "model")
    graph.add_edge("model", END)
    app = graph.compile()

    with morse_ai.run("pub-e2e-py-langgraph") as r:
        result = app.invoke(
            {"messages": [HumanMessage("Reply with the single word: pong.")]}
        )
        print(f"  reply: {str(result['messages'][-1].content)[:100]!r}")
        r.set_outcome(True, "completed")


SCENARIOS = {
    "anthropic": scenario_anthropic,
    "openai": scenario_openai,
    "langgraph": scenario_langgraph,
}


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = list(SCENARIOS) if which == "all" else [which]

    morse_ai.init(api_key=require("MORSE_API_KEY"), endpoint=ENDPOINT)
    print(f"[pub-e2e-py] morse_ai={morse_ai.__version__} endpoint={ENDPOINT}")

    failures = []
    for name in names:
        print(f"\n--- {name} ---")
        start = time.time()
        try:
            SCENARIOS[name]()
            print(f"[pub-e2e-py OK {name}] elapsed={time.time() - start:.1f}s")
        except Exception as exc:  # noqa: BLE001 - report, keep going
            failures.append(name)
            print(f"[pub-e2e-py FAIL {name}] {type(exc).__name__}: {exc}")

    morse_ai.flush()
    morse_ai.shutdown()
    if failures:
        print(f"\nFAILED: {', '.join(failures)}")
        return 1
    print("\nAll scenarios passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
