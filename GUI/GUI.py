import tkinter as tk
from tkinter import filedialog, messagebox
import torch
import pandas as pd
import numpy as np
from main_Transformer import Conformer  
from main_1DCNN import Simple1DCNN
import torch.nn.functional as F
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import signal

def load_conformer_model():
    model = Conformer()
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    return model

def load_simple1dcnn_model():
    model = Simple1DCNN()
    model.load_state_dict(torch.load('best_model_cnn.pth'))
    model.eval()
    return model

current_model = None

def select_model(model_name):
    global current_model
    if model_name == 'conformer':
        current_model = load_conformer_model()
        messagebox.showinfo("モデルの選択", "Transformersを選択しました")
    elif model_name == 'simple1dcnn':
        current_model = load_simple1dcnn_model()
        messagebox.showinfo("モデルの選択", "1D-CNNを選択しました")
    else:
        messagebox.showerror("Error", "モデルを選択してください")


def preprocess_data(file_path):
    data = pd.read_csv(file_path, header=None, dtype=float).values
    data = np.asarray(data, dtype=np.float32)
    data = torch.tensor(data).unsqueeze(1)
    return data

def predict(data):
    global current_model
    if current_model is None:
        messagebox.showerror("Error", "モデルを選択してください")
        return None
    with torch.no_grad():
        outputs = current_model(data)
        probabilities = F.softmax(outputs, dim=1)
    return probabilities.numpy()

def upload_data():
    global current_model
    if current_model is None:
        messagebox.showerror("Error", "モデルを選択してください")
        return
    filename = filedialog.askopenfilename()  
    data = preprocess_data(filename)
    probabilities = predict(data)
    probabilities_percent = probabilities * 100  

    result_str = (f"BOO：{probabilities_percent[0][0]:.2f}%\n"
                  f"DU：{probabilities_percent[0][1]:.2f}%\n"
                  f"MIX：{probabilities_percent[0][2]:.2f}%\n"
                  f"Normal：{probabilities_percent[0][3]:.2f}%")
    label_result.config(text=result_str)

    plot_curve_from_csv(filename)

    fig_quadrant = plot_quadrant_chart(probabilities_percent[0])

def plot_curve_from_csv(file_path):
    y_data = pd.read_csv(file_path, header=None).values.flatten()
    x_data = np.arange(len(y_data))  

    fig, ax1 = plt.subplots(figsize=(6.5, 6.5))

    ax1.plot(x_data, y_data, label='Original Data')
    ax1.set_title("Original CSV Data")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Flow Rate")
    ax1.legend()

    canvas = FigureCanvasTkAgg(fig, master=frame_charts)  
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=0, column=0)  
    canvas.draw()

def load_and_display_image():
    filename = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
    img = resize_image(filename, 600, 600)
    tk_img = ImageTk.PhotoImage(img)
    label_img.config(image=tk_img)
    label_img.image = tk_img

def resize_image(img_path, max_width, max_height):
    img = Image.open(img_path)
    if img.height > img.width:
        if img.height > max_height:
            width = max_height * img.width // img.height
            img = img.resize((width, max_height))
    else:
        if img.width > max_width:
            height = max_width * img.height // img.width
            img = img.resize((max_width, height))
    return img

def adjust_probabilities(probabilities):
    max_index = np.argmax(probabilities)
    adjusted_probabilities = [0, 0, 0, 0]
    adjusted_probabilities[max_index] = probabilities[max_index]
    return adjusted_probabilities

def plot_quadrant_chart(probabilities):
    adjusted_probabilities = adjust_probabilities(probabilities)

    fig3 = plt.Figure(figsize=(6.5, 6.5), dpi=100)
    ax3 = fig3.add_subplot(111)
    ax3.set_xlim([-1.0, 1.0])
    ax3.set_ylim([-1.0, 1.0])
    ax3.set_aspect('equal', adjustable='datalim')

    ax3.text(-0.5, 0.5, 'BOO', fontsize=12, ha='center')
    ax3.text(0.5, 0.5, 'Normal', fontsize=12, ha='center')
    ax3.text(0.5, -0.5, 'DU', fontsize=12, ha='center')
    ax3.text(-0.5, -0.5, 'MIX', fontsize=12, ha='center')

    ax3.spines['left'].set_position('center')
    ax3.spines['bottom'].set_position('center')
    ax3.spines['right'].set_color('none')
    ax3.spines['top'].set_color('none')
    ax3.xaxis.set_ticks_position('bottom')
    ax3.yaxis.set_ticks_position('left')

    max_index = np.argmax(adjusted_probabilities)
    x, y = 0, 0

    if max_index == 0:  
        x, y = -0.5, 0.5
        x += (adjusted_probabilities[1] + adjusted_probabilities[2] + adjusted_probabilities[3])/3
        y -= (adjusted_probabilities[1] + adjusted_probabilities[2] + adjusted_probabilities[3])/3
    elif max_index == 1:  
        x, y = 0.5, -0.5
        x -= (adjusted_probabilities[0] + adjusted_probabilities[2] + adjusted_probabilities[3])/3
        y += (adjusted_probabilities[0] + adjusted_probabilities[2] + adjusted_probabilities[3])/3
    elif max_index == 2:  
        x, y = -0.5, -0.5
        x += (adjusted_probabilities[0] + adjusted_probabilities[1] + adjusted_probabilities[3])/3
        y += (adjusted_probabilities[0] + adjusted_probabilities[1] + adjusted_probabilities[3])/3
    elif max_index == 3: 
        x, y = 0.5, 0.5
        x -= (adjusted_probabilities[0] + adjusted_probabilities[1] + adjusted_probabilities[2])/3
        y -= (adjusted_probabilities[0] + adjusted_probabilities[1] + adjusted_probabilities[2])/3

    ax3.plot(x, y, 'ro')
    canvas_quadrant = FigureCanvasTkAgg(fig3, master=frame_charts)
    canvas_quadrant_widget = canvas_quadrant.get_tk_widget()
    canvas_quadrant_widget.grid(row=0, column=1)  
    canvas_quadrant.draw()
    return fig3

root = tk.Tk()
root.title("LUTS自動診断アプリ")  
root.geometry("1300x1300")  

button_select_conformer = tk.Button(root, text="Transformer モデル", command=lambda: select_model('conformer'), width=25, height=2, font=("Arial", 13))
button_select_conformer.pack()
button_select_simple1dcnn = tk.Button(root, text="1D-CNN モデル", command=lambda: select_model('simple1dcnn'), width=25, height=2, font=("Arial", 13))
button_select_simple1dcnn.pack()
button_img = tk.Button(root, text="UFM画像ファイル", command=load_and_display_image, width=30, height=2, font=("Arial", 13))
button_img.pack()
button_upload = tk.Button(root, text="CSVファイル", command=upload_data, width=30, height=2, font=("Arial", 13))
button_upload.pack()

label_result = tk.Label(root, text="", font=("Arial", 16))
label_result.pack()
label_img = tk.Label(root)
label_img.pack()
frame_charts = tk.Frame(root)
frame_charts.pack(fill=tk.BOTH, expand=True)

root.mainloop()
