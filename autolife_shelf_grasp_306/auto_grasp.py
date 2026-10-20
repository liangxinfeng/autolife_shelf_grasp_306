"""Candidate ranking and single-use automatic execution authorization."""
import math


class CandidateUnavailable(ValueError):
    """Only this object's depth or collision-checked path is unavailable."""


def rank_candidates(items):
    def key(item):
        score = float(item.get('score') or 0.)
        if not math.isfinite(score): score = 0.
        box = item.get('box') or [0, 0, 0, 0]
        area = max(0., float(box[2])-float(box[0])) * max(0., float(box[3])-float(box[1]))
        if not math.isfinite(area): area = 0.
        return (-score, -area, str(item['id']))
    # Ranking is only a heuristic; every candidate still needs fresh perception
    # and the complete existing depth/IK/collision checks before execution.
    seen = set(); result = []
    for item in sorted(items, key=key):
        if item['id'] not in seen:
            result.append(item['id']); seen.add(item['id'])
    return result


class AutoExecutionGate:
    """Checkbox changes, selection changes and stop revoke pending consent."""
    def __init__(self):
        self.serial = 0; self.intent = None

    def revoke(self):
        self.serial += 1; self.intent = None

    def arm(self, request, context, enabled):
        self.revoke()
        if enabled: self.intent = (self.serial, request, context)

    def consume(self, intent, request, context, enabled):
        if not enabled or intent is None or intent != self.intent:
            return False
        if intent[1:] != (request, context): return False
        self.revoke()
        return True
