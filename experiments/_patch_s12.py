import sys
p = "experiments/s12_guard.py"
s = open(p, encoding="utf-8").read()
old = '''    if pred["reason"] != "plain_extension":
        return {"verdict": "allow", "cls": "house_" + pred["reason"]}
'''
new = '''    if pred["reason"] != "plain_extension":
        return {"verdict": "allow", "cls": "house_" + pred["reason"]}
    if not pred["added"]:
        return {"verdict": "allow", "cls": "noop_edge_already_active_no_resource_effect"}      # consumes nothing (observed dC = 0): nothing to conserve
'''
if s.count(old) != 1:
    sys.exit("pattern mismatch")
open(p, "w", encoding="utf-8").write(s.replace(old, new))
print("patched")
