"""Add the TA training section + PNG to gen.py (idempotent)."""
import sys
p = sys.argv[1]; s = open(p).read()
if "_ta_section" in s: print("already patched"); sys.exit(0)
s = s.replace("{_fast_section()}\n", "{_fast_section()}\n{_ta_section()}\n", 1)
s = s.replace("def _fast_section():", '''def _ta_section():
    try:
        import ta_dash; return ta_dash.section()
    except Exception as ex:
        return f"<h2>TA training</h2><div class='card warn'>TA section error: {html.escape(repr(ex))}</div>"

def _fast_section():''', 1)
s = s.replace('''    json.dump({"runid": A["runid"]''', '''    try:
        import ta_dash; ta_dash.png(os.path.join(a.out, "ta_training.png"))
    except Exception as ex:
        print("ta png error", repr(ex))
    json.dump({"runid": A["runid"]''', 1)
assert s.count("_ta_section") == 2 and "ta_training.png" in s
open(p, "w").write(s); print("patched")
