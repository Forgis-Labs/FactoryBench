"""Agentic baseline for FactoryBench.

A GPT-5.1-driven ReAct agent equipped with four general-purpose tools:
manual RAG, signal statistics, forecaster, and a Python sandbox.

Measured against zero-shot GPT-5.1 on the same items, tool access is a
wash overall (-2.9 pp, CI [-7.4, +1.6]); it helps where the task is
computational (L1 +9.8 pp) and hurts where the agent delegates its
extrapolation to a forecaster weaker than itself (L3 -10.5 pp). See
app:agent-baseline. This docstring used to call the agent "a stronger
upper bound than the zero-shot panel"; that claim was retracted in
e2f7c5b and the code should not keep asserting it.

The knowledge graph is intentionally *not* exposed as a tool: it is the
same catalogue L4 gold answers are drawn from, so exposing it would be
ground-truth leakage.
"""
