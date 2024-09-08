#!/usr/bin/env python
# coding: utf-8

from stockMarketProject.StockPredictor import predict, load_model
from pyvirtualdisplay import Display
import tkinter as tk
from tkinter import ttk

# Start the virtual display
display = Display(visible=0, size=(800, 600))
display.start()

# Load the model
model = load_model()  # Load your model
predictors = ["Close", "Volume", "Open", "High", "Low"]

# Create the Tkinter root window
root = tk.Tk()
root.geometry("800x800")
root.title("Stock Market Predictor Model")

# Create a frame for the input fields
frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Create and place labels and entry fields for each predictor
entries = {}
for i, predictor in enumerate(predictors):
    label = ttk.Label(frame, text=predictor)
    label.grid(row=i, column=0, sticky=tk.W)
    entry = ttk.Entry(frame)
    entry.grid(row=i, column=1, sticky=(tk.W, tk.E))
    entries[predictor] = entry

# Function to handle prediction
def predict_stock():
    entry_data = {predictor: float(entries[predictor].get()) for predictor in predictors}
    prediction = predict(model, entry_data)
    result_label.config(text=f"Predicted Value: {prediction}")

# Create and place the predict button
predict_button = ttk.Button(frame, text="Predict", command=predict_stock)
predict_button.grid(row=len(predictors), column=0, columnspan=2)

# Create and place the result label
result_label = ttk.Label(frame, text="")
result_label.grid(row=len(predictors) + 1, column=0, columnspan=2)

# Run the Tkinter event loop
root.mainloop()

# Stop the virtual display when the application is closed
display.stop()