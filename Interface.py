import customtkinter as ctk
from tkinter import filedialog
import tkinter as tk
from PIL import Image, ImageTk   # Import ImageTk

class MammographyTool(ctk.CTk):
    IMG_W, IMG_H = 250, 300
    LEFT_X, Y = 130, 180
    RIGHT_X = 530
    BG_COLOR = "#1E3B8B"

    def __init__(self):
        super().__init__()
        self.title("Mammography Tool")
        self.geometry("900x600")
        self.resizable(False, False)

        self.bg_image_path = "Background.png"
        self.colour_image_path = "bgcolour.png"
        self.show_home()


    def show_home(self):
        for widget in self.winfo_children():
            widget.destroy()

        bg_img = Image.open(self.bg_image_path).resize((900, 600))
        self.bg_photo = ctk.CTkImage(light_image=bg_img, size=(900, 600))
        self.bg_label = ctk.CTkLabel(self, image=self.bg_photo, text="")
        self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        self.next_btn = ctk.CTkButton(
            self,
            text="Next",
            fg_color="#16a34a",
            text_color="white",
            height=50,
            width=160,
            command=self.show_upload_page
        )
        self.next_btn.place(relx=0.5, rely=0.92, anchor="center")

    def show_upload_page(self):
        for widget in self.winfo_children():
            widget.destroy()

        bg_img = Image.open(self.colour_image_path).resize((900, 600))
        self.bg_colour_photo = ctk.CTkImage(light_image=bg_img, size=(900, 600))
        self.bg_label = ctk.CTkLabel(self, image=self.bg_colour_photo, text="")
        self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        self.home_btn = ctk.CTkButton(self, text="Home", width=80, command=self.show_home)
        self.home_btn.place(x=10, y=10)

        self.cc_path = None
        self.mlo_path = None
        self.cc_img_tk = None
        self.mlo_img_tk = None



        # CC upload button and image centered below
        cc_x = 200
        cc_btn_y = 100
        self.upload_cc_btn = ctk.CTkButton(
            self, text="Upload CC", width=160, height=40, fg_color="#2563eb", text_color="white",
            command=self.upload_cc
        )
        self.upload_cc_btn.place(x=cc_x, y=cc_btn_y, anchor="n")
        self.cc_img_label = tk.Label(self, bg=self.BG_COLOR)
        self.cc_img_label.place(x=cc_x+80, y=cc_btn_y+50, anchor="n")  # +80 to center image under button

        # MLO upload button and image centered below
        mlo_x = 540
        mlo_btn_y = 100
        self.upload_mlo_btn = ctk.CTkButton(
            self, text="Upload MLO", width=160, height=40, fg_color="#2563eb", text_color="white",
            command=self.upload_mlo
        )
        self.upload_mlo_btn.place(x=mlo_x, y=mlo_btn_y, anchor="n")
        self.mlo_img_label = tk.Label(self, bg=self.BG_COLOR)
        self.mlo_img_label.place(x=mlo_x+80, y=mlo_btn_y+50, anchor="n")  # +80 to center image under button

        # Predict button
        self.predict_btn = ctk.CTkButton(
            self, text="Predict", width=200, height=50, fg_color="#16a34a", text_color="white",
            command=self.predict_and_show_result, state="disabled"
        )
        self.predict_btn.place(relx=0.5, rely=0.92, anchor="center")

    def upload_cc(self):
        file = filedialog.askopenfilename(
            title="Select CC Image",
            filetypes=[("Images", "*.png *.jpg *.jpeg")]
        )
        if file:
            self.cc_path = file
            img = Image.open(file).resize((self.IMG_W, self.IMG_H))
            self.cc_img_tk = ImageTk.PhotoImage(img)
            self.cc_img_label.configure(image=self.cc_img_tk)
            self.cc_img_label.image = self.cc_img_tk
            self.check_predict_ready()

    def upload_mlo(self):
        file = filedialog.askopenfilename(
            title="Select MLO Image",
            filetypes=[("Images", "*.png *.jpg *.jpeg")]
        )
        if file:
            self.mlo_path = file
            img = Image.open(file).resize((self.IMG_W, self.IMG_H))
            self.mlo_img_tk = ImageTk.PhotoImage(img)
            self.mlo_img_label.configure(image=self.mlo_img_tk)
            self.mlo_img_label.image = self.mlo_img_tk
            self.check_predict_ready()

    def check_predict_ready(self):
        if self.cc_path and self.mlo_path:
            self.predict_btn.configure(state="normal")
        else:
            self.predict_btn.configure(state="disabled")

    def predict_and_show_result(self):
        # Directly show result page, no extra window
        self.show_result_page(self.cc_path, self.mlo_path)

    def show_result_page(self, cc_path, mlo_path):
        for widget in self.winfo_children():
            widget.destroy()

        import torch
        import matplotlib.pyplot as plt
        from PIL import Image
        import numpy as np
        import io
        import sys
        import importlib.util
        import os

        # Dynamically import Model.py as a module
        model_path = os.path.join(os.path.dirname(__file__), "Model.py")
        spec = importlib.util.spec_from_file_location("model_module", model_path)
        model_module = importlib.util.module_from_spec(spec)
        sys.modules["model_module"] = model_module
        spec.loader.exec_module(model_module)

        # Prepare transform and model
        transform = model_module.transform
        model = model_module.load_model()
        DEVICE = model_module.DEVICE
        IDX_TO_CLASS = model_module.IDX_TO_CLASS

        # Prepare images
        cc_img = Image.open(cc_path).convert('RGB')
        mlo_img = Image.open(mlo_path).convert('RGB')
        cc_img_t = transform(cc_img)
        mlo_img_t = transform(mlo_img)
        cc_b = cc_img_t.unsqueeze(0).to(DEVICE).requires_grad_(True)
        mlo_b = mlo_img_t.unsqueeze(0).to(DEVICE).requires_grad_(True)

        # Register hooks
        cc_target = model.cc_net.features[-1]
        mlo_target = model.mlo_net.features[-1]
        cc_hook = model_module.FeatureGradHook(cc_target)
        mlo_hook = model_module.FeatureGradHook(mlo_target)

        # Predict
        model.zero_grad(set_to_none=True)
        logits = model(cc_b, mlo_b)
        pred_idx = int(torch.argmax(logits, dim=1).item())
        pred_cls = IDX_TO_CLASS[pred_idx]
        score = logits[0, pred_idx]
        model.zero_grad(set_to_none=True)
        score.backward(retain_graph=False)

        # Grad-CAM
        cam_cc  = model_module.compute_cam(cc_hook.activations,  cc_hook.gradients,  up_size=(model_module.IMG_SIZE, model_module.IMG_SIZE))
        cam_mlo = model_module.compute_cam(mlo_hook.activations, mlo_hook.gradients, up_size=(model_module.IMG_SIZE, model_module.IMG_SIZE))
        img_cc,  overlay_cc  = model_module.overlay_cam(cc_img_t,  cam_cc)
        img_mlo, overlay_mlo = model_module.overlay_cam(mlo_img_t, cam_mlo)

        # Plot to buffer (no plt.show to avoid extra window)
        fig, axs = plt.subplots(2, 2, figsize=(9, 7))
        axs[0,0].imshow(img_cc);        axs[0,0].set_title("CC - Original");   axs[0,0].axis("off")
        axs[0,1].imshow(overlay_cc);    axs[0,1].set_title("CC - Grad-CAM");   axs[0,1].axis("off")
        axs[1,0].imshow(img_mlo);       axs[1,0].set_title("MLO - Original");  axs[1,0].axis("off")
        axs[1,1].imshow(overlay_mlo);   axs[1,1].set_title("MLO - Grad-CAM");  axs[1,1].axis("off")
        plt.suptitle(f"Predicted ACR Classification: {pred_cls}", fontsize=14)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)

        # Show result page
        bg_img = Image.open(self.colour_image_path).resize((900, 600))
        self.bg_colour_photo = ctk.CTkImage(light_image=bg_img, size=(900, 600))
        self.bg_label = ctk.CTkLabel(self, image=self.bg_colour_photo, text="")
        self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        self.home_btn = ctk.CTkButton(self, text="Home", width=80, command=self.show_home)
        self.home_btn.place(x=10, y=10)

        # Display matplotlib result
        result_img = Image.open(buf)
        result_img = result_img.resize((700, 500))
        self.result_img_tk = ImageTk.PhotoImage(result_img)
        self.result_label = tk.Label(self, image=self.result_img_tk, bd=0)
        self.result_label.place(relx=0.5, rely=0.5, anchor="center")

    def home_upload_and_show_tool(self):
        files = filedialog.askopenfilenames(
            title="Select Mammogram Images",
            filetypes=[("Images", "*.png *.jpg *.jpeg")]
        )
        if not files:
            return

        cc_path, mlo_path = None, None
        for file in files:
            upper = file.upper()
            if 'CC' in upper and not cc_path:
                cc_path = file
            elif 'MLO' in upper and not mlo_path:
                mlo_path = file

        # Fallback: assign generically if neither CC nor MLO found
        if not cc_path and not mlo_path:
            if len(files) == 1:
                mlo_path = files[0]
            elif len(files) >= 2:
                mlo_path, cc_path = files[0], files[1]
        elif not mlo_path and cc_path and len(files) > 1:
            mlo_path = [f for f in files if f != cc_path][0]
        elif not cc_path and mlo_path and len(files) > 1:
            cc_path = [f for f in files if f != mlo_path][0]

        self.show_tool(cc_path, mlo_path)

    def show_tool(self, cc_path=None, mlo_path=None):
        for widget in self.winfo_children():
            widget.destroy()

        # Place blue background image
        bg_img = Image.open(self.colour_image_path).resize((900, 600))
        self.bg_colour_photo = ctk.CTkImage(light_image=bg_img, size=(900, 600))
        self.bg_label = ctk.CTkLabel(self, image=self.bg_colour_photo, text="")
        self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        self.home_btn = ctk.CTkButton(self, text="Home", width=80, command=self.show_home)
        self.home_btn.place(x=10, y=10)
        self.exit_btn = ctk.CTkButton(
            self, text="Exit", width=80,
            fg_color="#dc2626", hover_color="#b91c1c", text_color="white",
            command=self.destroy
        )
        self.exit_btn.place(relx=1.0, rely=1.0, anchor="se", x=-10, y=-10)
        self.check_btn = ctk.CTkButton(
            self, text="Check Another Scan",
            fg_color="#16a34a", text_color="white", height=50, width=200,
            command=self.home_upload_and_show_tool
        )
        self.check_btn.place(relx=0.5, rely=0.92, anchor="center")

        # Remove leftover tk.Labels
        for widget in self.winfo_children():
            if isinstance(widget, tk.Label):
                widget.destroy()

        # --- LEFT (MLO) ---
        if mlo_path:
            left_frame = tk.Frame(self, width=self.IMG_W, height=self.IMG_H+40, bg=self.BG_COLOR, highlightthickness=0, bd=0)
            left_frame.place(x=self.LEFT_X, y=self.Y)
            mlo_img = Image.open(mlo_path).resize((self.IMG_W, self.IMG_H))
            mlo_img_tk = ImageTk.PhotoImage(mlo_img)  # Use PIL ImageTk
            img_label = tk.Label(left_frame, image=mlo_img_tk, bd=0)
            img_label.image = mlo_img_tk
            img_label.pack(pady=(0,0))
            # Label below, centered
            label = tk.Label(
                left_frame, text="MLO", font=("Arial", 18, "bold"),
                fg="white", bg=self.BG_COLOR, bd=0, highlightthickness=0
            )
            label.pack(pady=(0,0))
        # --- RIGHT (CC) ---
        if cc_path:
            right_frame = tk.Frame(self, width=self.IMG_W, height=self.IMG_H+40, bg=self.BG_COLOR, highlightthickness=0, bd=0)
            right_frame.place(x=self.RIGHT_X, y=self.Y)
            cc_img = Image.open(cc_path).resize((self.IMG_W, self.IMG_H))
            cc_img_tk = ImageTk.PhotoImage(cc_img)  # Use PIL ImageTk
            img_label = tk.Label(right_frame, image=cc_img_tk, bd=0)
            img_label.image = cc_img_tk
            img_label.pack(pady=(0,0))
            label = tk.Label(
                right_frame, text="CC", font=("Arial", 18, "bold"),
                fg="white", bg=self.BG_COLOR, bd=0, highlightthickness=0
            )
            label.pack(pady=(0,0))

if __name__ == "__main__":
    app = MammographyTool()
    app.mainloop()
