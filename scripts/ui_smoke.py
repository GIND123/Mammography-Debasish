"""Render the DiceMed result page for a CC/MLO pair and save a screenshot (UI smoke test).

    python scripts/ui_smoke.py <bundle.pt> <cc> <mlo> <out.png>
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["DICEMED_BUNDLE"] = sys.argv[1]

from PIL import ImageGrab  # noqa: E402

import Interface  # noqa: E402

app = Interface.MammographyTool()
while "fast" not in app.predictors and app.predictor_error is None:
    app.update()
    time.sleep(0.05)
assert app.predictor_error is None, app.predictor_error
r = app.predictor.predict(sys.argv[2], sys.argv[3], explain=True)
app.show_result_page(r, {"CC": sys.argv[2], "MLO": sys.argv[3]})
for _ in range(20):
    app.update()
    time.sleep(0.05)
x, y = app.winfo_rootx(), app.winfo_rooty()
ImageGrab.grab(bbox=(x, y, x + app.winfo_width(), y + app.winfo_height())).save(sys.argv[4])
print(r.summary())
app.destroy()
