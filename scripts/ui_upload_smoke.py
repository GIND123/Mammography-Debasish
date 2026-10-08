"""Screenshot the upload page with both images loaded (UI smoke test)."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["DICEMED_BUNDLE"] = sys.argv[1]
from tkinter import filedialog  # noqa: E402

from PIL import ImageGrab  # noqa: E402

import Interface  # noqa: E402

app = Interface.MammographyTool()
app.show_upload_page()
files = iter([sys.argv[2], sys.argv[3]])
filedialog.askopenfilename = lambda **kw: next(files)  # stand-in for the file dialog
app.upload("CC")
app.upload("MLO")
for _ in range(20):
    app.update()
    time.sleep(0.05)
x, y = app.winfo_rootx(), app.winfo_rooty()
ImageGrab.grab(bbox=(x, y, x + app.winfo_width(), y + app.winfo_height())).save(sys.argv[4])
print("predict button:", app.predict_btn.cget("state"))
app.destroy()
