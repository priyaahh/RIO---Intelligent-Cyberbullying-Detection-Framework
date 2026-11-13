"""Quick smoke test for expanded mock predictor vocabulary."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.orchestrator import OrchestratorAgent

orch = OrchestratorAgent()

tests = [
    ("You are so stupid and ugly",           "Insult"),
    ("Nobody likes you, go away",             "Harassment"),
    ("You disgrace your whole family",        "Insult"),
    ("I hate everything about you",           "Harassment"),
    ("Your work is completely useless",       "Insult"),
    ("I will kill you if you do that again",  "Threat"),
    ("You are a worthless piece of shit",     "Insult"),
    ("Go back to your country beaner",        "Hate Speech"),
    ("She bullied me every single day",       "Harassment"),
    ("You retarded fool, die already",        "Threat"),
    ("I am going to stab you tonight",        "Threat"),
    ("Fuck off you dumbass loser",            "Insult"),
    ("Have a great day!",                     "Not Cyberbullying"),
    ("I disagree but I respect your opinion", "Not Cyberbullying"),
    ("The weather today is sunny",            "Not Cyberbullying"),
]

correct = 0
print(f"{'RESULT':<22} {'CONF':>5}  {'EXPECTED':<22} {'OK?'}  TEXT")
print("-" * 90)
for text, expected in tests:
    r = orch.process(text)
    ok = "OK" if r.primary_label.lower()[:5] in expected.lower() else "MISS"
    if ok == "OK":
        correct += 1
    print(f"{r.primary_label:<22} {r.confidence:>5.2f}  {expected:<22} {ok}  {text[:40]}")

print(f"\nScore: {correct}/{len(tests)} ({correct/len(tests)*100:.0f}%)")

