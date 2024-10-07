#FOR EXTERNAL CAMERA
import tkinter as tk
from tkinter import ttk
from tkinter import PhotoImage
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import ImageTk, Image
import pygame
import sqlite3
import os
from ultralytics import YOLO
from datetime import datetime
import threading 
import csv

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import traceback
import yagmail

#DATABASE
DB_PATH = os.path.join(BASE_DIR, "detected.db")
IMAGE_PATH = os.path.join(BASE_DIR, "images")

# Open SQLite connection
with sqlite3.connect(DB_PATH) as conn:
    cursor = conn.cursor()

#Email Config
EMAIL_ADDRESS = "swine.detections@gmail.com" 
EMAIL_PASSWORD = "Bscs4a!!"   
RECIPIENT_EMAIL = "aldrexmark@gmail.com" 

def show_notification(class_name):
    notification_window = tk.Toplevel(root)
    notification_window.title("Notification")

    # Add text to the notification window
    notification_label = tk.Label(notification_window, text=f"Detected: {class_name}", font=("Arial", 20))
    notification_label.pack(padx=20, pady=20)

    # Play a sound notification
    pygame.mixer.music.load("C:/Users/Aldrex/Documents/Thesis/bell_sound.mp3") 
    pygame.mixer.music.play()
    
    # Send an email if "negative_pig" is detected
    if class_name == "person":
        send_email(class_name)
        
def send_email(class_name):
    subject = f"Alert: {class_name} Detected!"
    body = f"The system has detected {class_name}. Please check the camera feed and reports detections."

    # Send email using yagmail
    yag = yagmail.SMTP(EMAIL_ADDRESS, EMAIL_PASSWORD)
    yag.send(RECIPIENT_EMAIL, subject, body)
    yag.close()
    print("Email sent successfully!")
    
#LAYMANS TERM ============================================= #

def translate_to_laymans_terms(class_name, confidence):
    # if class_name == 'negative_pig':
    #     return f"Detected an unhealthy pig, with a ({confidence*100:.1f}%) confidence level."
    # return f"Detected a {class_name}, with ({confidence*100:.1f}%) confidence level."
     
    if class_name == 'person':
        return f"Found a person, with a ({confidence*100:.1f}%) sureness."
    return f"Found a {class_name}, with ({confidence*100:.1f}%) sureness."

#CAMERA =================================================== #     
def open_camera():
    global camera_window
    root.withdraw()  # Hide the main window

    # Create the camera window
    camera_window = tk.Toplevel(root)
    camera_window.title("Camera")
    camera_window.state('zoomed')

    # Create a label for displaying the camera feed
    camera_label = tk.Label(camera_window)
    camera_label.pack(side="top", fill="both", expand=True)

    # Create a frame for the buttons
    button_frame = tk.Frame(camera_window)
    button_frame.pack(side="top", pady=10)

    # Create three buttons under the camera feed
    button1 = tk.Button(button_frame, text="Back to Home", font=("Arial", 12), command=button1_action, width=45, height=2, bg="#1B4A89", fg="white", borderwidth=2, relief="flat")
    button1.pack(side="left", padx=10)

    button2 = tk.Button(button_frame, text="Reports", font=("Arial", 12), command=button2_action, width=45, height=2, bg="#1B4A89", fg="white", borderwidth=2, relief="flat")
    button2.pack(side="left", padx=10)

    button3 = tk.Button(button_frame, text="Exit", font=("Arial", 12), command=exit_program, width=45, height=2, bg="#1B4A89", fg="white", borderwidth=2, relief="flat")
    button3.pack(side="left", padx=10)

    # OpenCV DNN
    net = cv2.dnn.readNet(r"C:\Users\Aldrex\Documents\Thesis\dnn_model\yolov4-tiny.weights",
                          r"C:\Users\Aldrex\Documents\Thesis\dnn_model\yolov4-tiny.cfg")
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(320, 320), scale=1/255)

    # Load Class Lists
    classes = []
    with open(r"C:\Users\Aldrex\Documents\Thesis\dnn_model\classes.txt") as file_object:
        for class_name in file_object.readlines():
            class_name = class_name.strip()
            classes.append(class_name)

    # Camera Initialization
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    button_person = False
    def click_button(event, x, y, flags, params):
        nonlocal button_person
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, y)
            polygon = np.array([[(20, 20), (220, 20), (220, 70), (20, 70)]])
            is_inside = cv2.pointPolygonTest(polygon, (x, y), False)
            if is_inside > 0:
                print("Inside the Button")
                if button_person is False:
                    button_person = True
                else:
                    button_person = False

                print("Now Button Person is:", button_person)
                camera_window.destroy()  # Close the camera window
                root.deiconify()  # Show the main window

    # Modified camera loop to update the label
    pygame.init()
    while True:
        ret, frame = cap.read()

        # Object Detection
        (class_ids, scores, bboxes) = model.detect(frame)
        
        for class_id, score, bbox in zip(class_ids, scores, bboxes):
            (x, y, w, h) = bbox
            class_name = classes[class_id]
            confidence = float(score)
            explanation = translate_to_laymans_terms(class_name, confidence)
            cv2.putText(frame, str(class_name), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (200, 0, 50), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 0, 50), 3)

            # Capture image data within the bounding box
            roi = frame[y:y + h, x:x + w]

            # Convert the image data to binary format
            _, buffer = cv2.imencode('.jpg', roi)
            image_data = buffer.tobytes()

            # Insert data into the database
            cursor.execute("INSERT INTO detections (class_name, confidence, x, y, width, height, image_data, explanation) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                           (class_name, confidence, x, y, w, h, image_data, explanation))
            conn.commit()

            if class_name == "cell phone": 
                show_notification(class_name)

            
        # Button
        cv2.rectangle(frame, (20, 20), (220, 70), (0, 0, 200), -1)
        polygon = np.array([[(20, 20), (220, 20), (220, 70), (20, 70)]])
        cv2.fillPoly(frame, polygon, (0, 0, 200))
        cv2.putText(frame, "Camera", (30, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 255, 255))

        # Convert the frame to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the frame to ImageTk format
        frame_tk = ImageTk.PhotoImage(Image.fromarray(frame_rgb))

        # Update the label with the new frame
        camera_label.configure(image=frame_tk)
        camera_label.photo = frame_tk

        # Update the Tkinter window
        camera_window.update()

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def button1_action():
    camera_window.destroy()  # Close the camera window
    root.state('zoomed')
    root.deiconify()  # Show the main window

def button2_action():
    show_report()
    
def show_report():
    # Create the reports window
    reports_window = tk.Toplevel(root)
    reports_window.title("Reports")
    reports_window.geometry("800x600")
    reports_window.configure(bg="#F0F0F0")  # Set background color

    # Create a frame for the reports
    reports_frame = tk.Frame(reports_window, bg="#F0F0F0")
    reports_frame.pack(side="top", fill="both", expand=True)

    # Create a scrollbar for the reports
    scrollbar = tk.Scrollbar(reports_frame)
    scrollbar.pack(side="right", fill="y")

    # Create a text widget to display reports
    reports_text = tk.Text(reports_frame, yscrollcommand=scrollbar.set, wrap="none", font=("Arial", 12), bg="#F0F0F0", fg="#1B4A89")
    reports_text.pack(side="left", fill="both", expand=True)

    # Create a button to save the report to CSV
    save_button = tk.Button(reports_window, text="Save to CSV", command=lambda: save_to_csv(reports_text), font=("Arial", 12),
                            width=15, height=2, bg="#1B4A89", fg="white", borderwidth=2, relief="flat")
    save_button.pack(side="bottom", pady=10)

    # Configure the scrollbar
    scrollbar.config(command=reports_text.yview)

    # Retrieve the last 20 detected items from the database
    cursor.execute("SELECT id, class_name, confidence, explanation FROM detections ORDER BY id DESC LIMIT 20")
    rows = cursor.fetchall()

    # Display the information in the text widget
    for row in rows:
        reports_text.insert(tk.END, f"ID: {row[0]}, Detections: {row[1]}, Possibility: {row[2]}, Explanation: {row[3]}\n")

def save_to_csv(reports_text):
    # Fetch the data directly from the database again
    cursor.execute("SELECT id, class_name, confidence, explanation FROM detections ORDER BY id DESC LIMIT 20")
    rows = cursor.fetchall()

    if rows:
        # Get the current date and time for the file name
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Define the save path for the CSV file with date and time
        save_directory = os.path.join(os.path.expanduser('~'), 'Documents')
        save_path = os.path.join(save_directory, f'report_{current_datetime}.csv')

        # Save the report to a CSV file
        with open(save_path, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            # Write the header
            writer.writerow(["ID", "Class Name", "Confidence", "Explanation"])
            # Write the rows
            for row in rows:
                writer.writerow(row)

        messagebox.showinfo("CSV Saved", f"The report has been saved to:\n{save_path}")
    else:
        messagebox.showinfo("No Data", "There is no data to save.")

    
#UPLOAD VIDEO AND DETECTION =============================================================
class VideoProcessingWindow(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("Video Processing")
        self.state('zoomed')
        
        # Initialize progress bar and detection popup window
        self.progress_bar = None
        self.detection_popup = None
        
        # Initialize cap as None
        self.cap = None

        # Load the background image
        bg_path = r"C:\Users\Aldrex\Documents\Thesis\pictures\bg.png"
        bg_image = Image.open(bg_path)
        self.resized_bg = bg_image.resize((self.winfo_screenwidth(), self.winfo_screenheight()), Image.LANCZOS)

        # Convert the Pillow image to PhotoImage
        self.tk_bg = ImageTk.PhotoImage(self.resized_bg)

        # Create a label for the background image
        bg_label = tk.Label(self, image=self.tk_bg)
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        
        #Placeholder Image
        # Create a label for the placeholder image
        self.placeholder_path_video = r"C:\Users\Aldrex\Documents\Thesis\pictures\placeholder_image.png"
        placeholder_image_video = Image.open(self.placeholder_path_video)
        resized_placeholder_image_video = placeholder_image_video.resize((550, 400), Image.LANCZOS)
        self.tk_placeholder_image_video = ImageTk.PhotoImage(resized_placeholder_image_video)
        self.placeholder_label_video = tk.Label(self, image=self.tk_placeholder_image_video)
        self.placeholder_label_video.place(x=200, y=250)
        
        # Video player
        self.video_label = tk.Label(self)
        self.video_label.place(x=200, y=250)
        
        # Create a label for displaying text
        text_label = tk.Label(self, text="Detect Behaviours on an Video", font=("Arial", 40), bg="#3b5e8c", fg="white", padx=10, pady=10)
        text_label.place(x=160, y=28)
        
        image_window_text = tk.Label(self, text = "Upload a video and the system will actively monitor swine behaviors in real-time, addressing potential concerns promptly and ensuring enhanced swine welfare. ",
                              font=("Arial", 22), bg='white', fg='black', wraplength=1500, justify=tk.LEFT)
        image_window_text.place(x=55, y=130)
        
        # Add a picture
        picture_image_window_path = r"C:\Users\Aldrex\Documents\Thesis\pictures\icon.png"
        original_picture_image_window = Image.open(picture_image_window_path)
        resized_picture_image_window = original_picture_image_window.resize((80, 80), Image.LANCZOS)

        # Convert the Pillow image to PhotoImage
        tk_picture_image_window = ImageTk.PhotoImage(resized_picture_image_window)

        # Create a label for the picture
        picture_label_image_window = tk.Label(self, image=tk_picture_image_window)
        picture_label_image_window.image = tk_picture_image_window
        picture_label_image_window.place(x=50, y=25)

         # Home Button
        back_to_home_button = tk.Button(self, text="Back to Home", command=self.back_to_home, font=("Arial", 16),
                                        width=20, height=1, bg="#545454", fg="white", borderwidth=2, relief="flat")
        back_to_home_button.place(x=200, y=675) 
        
        # Exit Button
        exit_program_button = tk.Button(self, text="Exit", command=self.exit_program, font=("Arial", 16),
                                        width=20, height=1, bg="#FF3131", fg="white", borderwidth=2, relief="flat")
        exit_program_button.place(x=500, y=675)
        
        # Upload Button
        upload_image_button = tk.Button(self, text="Upload Video", command=self.upload_video, font=("Arial", 16),
                                  width=30, height=2, bg="#1B4A89", fg="white", borderwidth=2, relief="flat")
        upload_image_button.place(x=900, y=325)
        
        # Process Button
        process_image_button = tk.Button(self, text="Process Video", command=self.process_video, font=("Arial", 16),
                                   width=30, height=2, bg="#1B4A89", fg="white", borderwidth=2, relief="flat")
        process_image_button.place(x=900, y=425)  
        
        # Clear Button
        clear_image_button = tk.Button(self, text="Clear Video", command=self.clear_video, font=("Arial", 16),
                                       width=30, height=2, bg="#1B4A89", fg="white", borderwidth=2, relief="flat")
        clear_image_button.place(x=900, y=625) 
        
        #Initialization
        self.video_path = None
    
    #Upload Video Function
    def upload_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
        if file_path:
            # Store the file path in the instance variable for later use
            self.video_path = file_path
            print("Selected Video File:", self.video_path)
            #Message Box
            messagebox.showinfo("Success", f"Video file '{self.video_path}' has been successfully selected.")
            
            # Display the first 10 seconds of the video
            self.display_video_preview()
            
            # Display video preview
    def display_video_preview(self):
        if self.video_path:
            cap = cv2.VideoCapture(self.video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            duration = 10  # Display the first 10 seconds
            num_frames = int(duration * fps)

            frames = []
            for _ in range(num_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)

            cap.release()

            # Resize frames to match placeholder image size
            resized_frames = [cv2.resize(frame, (550, 400)) for frame in frames]

            # Convert frames to RGB format
            rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in resized_frames]

            # Convert the frames to PhotoImage
            tk_frames = [ImageTk.PhotoImage(Image.fromarray(frame)) for frame in rgb_frames]

            # Update the video label
            self.video_label.config(image=tk_frames[0])
            self.video_label.image = tk_frames[0]

            # Play the frames
            self.play_video_frames(tk_frames, 0)

    # Play video frames
    def play_video_frames(self, frames, index):
        if index < len(frames):
            self.video_label.config(image=frames[index])
            self.video_label.image = frames[index]
            self.after(1000, lambda: self.play_video_frames(frames, index + 1))
        else:
            # After playing all frames, reset to placeholder image
            self.video_label.config(image=self.tk_placeholder_image_video)
            self.video_label.image = self.tk_placeholder_image_video
            
    def process_video(self):
        if self.video_path:
            cap = cv2.VideoCapture(self.video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Create progress bar
            self.create_progress_bar(total_frames)

            ret, frame = cap.read()
            frame_count = 0

            if frame is not None:
                H, W, _ = frame.shape
                # Save directory
                save_directory = os.path.join(os.path.expanduser('~'), 'Videos', 'Processed Videos')
                # Ensure the directory exists, create it if not
                if not os.path.exists(save_directory):
                    os.makedirs(save_directory)

                # Get the current date and time for the file name
                current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Define the save path for the processed video with date and time
                save_path = os.path.join(save_directory, f'processed_video_{current_datetime}.mp4')

                out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

                # Load the YOLO model
                model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'last.pt')
                model = YOLO(r"C:\Users\Aldrex\Documents\Thesis\dnn_model\best.pt")

                threshold = 0.45

                def process_frames():
                    nonlocal ret, frame, frame_count

                    while ret:
                        results = model(frame)[0]

                        for result in results.boxes.data.tolist():
                            x1, y1, x2, y2, score, class_id = result

                            color = (0, 255, 0)  # Default color (green) for normal_pig

                            if score > threshold:
                                if results.names[int(class_id)] == "negative_pig":
                                    color = (0, 0, 255)

                                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                                cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

                        frame_count += 1
                        self.update_progress(frame_count)  # Update progress bar

                        out.write(frame)
                        ret, frame = cap.read()

                    cap.release()
                    out.release()
                    cv2.destroyAllWindows()

                    # Notify the user that the video processing is complete
                    messagebox.showinfo("Video Processing Complete", f"Video has been processed and saved to:\n{save_path}")

                    print("Video Processing Complete")

                    # After the progress bar is complete, call process_and_play_video
                    self.process_and_play_video(current_datetime)

                # Start a separate thread for video processing
                processing_thread = threading.Thread(target=process_frames)
                processing_thread.start()
            else:
                print("Please upload a video first.")

    def process_and_play_video(self, current_datetime):
        # Display the processed video
        cap = cv2.VideoCapture(os.path.join(os.path.expanduser('~'), 'Videos', 'Processed Videos', f'processed_video_{current_datetime}.mp4'))

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            cv2.imshow("Processed Video", frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):  # Press 'q' to exit
                break

        cap.release()
        cv2.destroyAllWindows()
                
    # Clear Video
    def clear_video(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.video_label.configure(image='')

        # Hide the progress bar and label
        if self.progress_bar:
            self.progress_bar.place_forget()
        if self.progress_label:
            self.progress_label.place_forget()
    
    #Progress Bar    
    def create_progress_bar(self, max_value):
        style = ttk.Style()
        style.configure("TProgressbar", thickness=50)  
        
        self.progress_bar = ttk.Progressbar(self, length=375, mode='determinate', maximum=max_value)
        self.progress_bar.place(x=900, y=555) 
        
        # Create a label for displaying progress text
        self.progress_label = tk.Label(self, text="Processing...", font=("Arial", 12), bg="white", fg="black")
        self.progress_label.place(x=900, y=525) 

        
    def update_progress(self, value):
        if self.progress_bar:
            self.progress_bar['value'] = value
            self.progress_label.config(text=f"Processing {value}/{self.progress_bar['maximum']} frames")
            self.update()
        
    # def show_detection_popup(self, results):
        # if results:
            # detection_text = "\n".join([f"{box[4]:.2%} confidence: {results.names[int(box[5])].upper()}" for box in results.boxes.data.tolist()])
            # messagebox.showinfo("Detection", f"Detections:\n{detection_text}")

    #Other Defs
    def button_process_video_action():
        # Hide the root window
        root.withdraw()

        # Create the video processing window and maximize it
        video_processing_window = VideoProcessingWindow(root)

        # Start the Tkinter main event loop
        video_processing_window.mainloop()

        # Destroy the root window when the video processing window is closed
        root.destroy()
    
    #Back to Home
    def back_to_home(self):
        # Implement the action to go back to the main window
        self.destroy()
        root.state('zoomed')
        root.deiconify()
    
    #Exit 
    def exit_program(self):
        # Implement the action to exit the program
        self.destroy()
        root.destroy()
        cursor.close()
        conn.close()
        
# ================================================================================================
#UPLOAD PICTURE AND DETECTION =============================================================

class ImageProcessingWindow(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("Image Processing")
        self.state('zoomed')

        # Load the background image
        bg_path = r"C:\Users\Aldrex\Documents\Thesis\pictures\bg.png"
        bg_image = Image.open(bg_path)
        self.resized_bg = bg_image.resize((self.winfo_screenwidth(), self.winfo_screenheight()), Image.LANCZOS)
        
        # Convert the Pillow image to PhotoImage
        self.tk_bg = ImageTk.PhotoImage(self.resized_bg)

        # Create a label for the background image
        bg_label = tk.Label(self, image=self.tk_bg)
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        
        #Placeholder Image
        # Create a label for the placeholder image
        self.placeholder_path = r"C:\Users\Aldrex\Documents\Thesis\pictures\placeholder_image.png"
        placeholder_image = Image.open(self.placeholder_path)
        resized_placeholder_image = placeholder_image.resize((550, 400), Image.LANCZOS)
        self.tk_placeholder_image = ImageTk.PhotoImage(resized_placeholder_image)
        self.placeholder_label = tk.Label(self, image=self.tk_placeholder_image)
        self.placeholder_label.place(x=200, y=250)
        
        # Create a label for displaying text
        text_label = tk.Label(self, text="Detect Behaviours on an Image", font=("Arial", 40), bg="#3b5e8c", fg="white", padx=10, pady=10)
        text_label.place(x=160, y=28)
        
        image_window_text = tk.Label(self, text = "Upload an image and the system will automatically identify swine behaviors, distinguishing between normal and potential issues and providing insights for improved welfare.",
                              font=("Arial", 22), bg='white', fg='black', wraplength=1500, justify=tk.LEFT)
        image_window_text.place(x=55, y=130)
        
        
        # Add a picture
        picture_image_window_path = r"C:\Users\Aldrex\Documents\Thesis\pictures\icon.png"
        original_picture_image_window = Image.open(picture_image_window_path)
        resized_picture_image_window = original_picture_image_window.resize((80, 80), Image.LANCZOS)

        # Convert the Pillow image to PhotoImage
        tk_picture_image_window = ImageTk.PhotoImage(resized_picture_image_window)

        # Create a label for the picture
        picture_label_image_window = tk.Label(self, image=tk_picture_image_window)
        picture_label_image_window.image = tk_picture_image_window
        picture_label_image_window.place(x=50, y=25)

        
        # Home Button
        back_to_home_button = tk.Button(self, text="Back to Home", command=self.back_to_home, font=("Arial", 16),
                                        width=20, height=1, bg="#545454", fg="white", borderwidth=2, relief="flat")
        back_to_home_button.place(x=200, y=675) 
        
        # Exit Button
        exit_program_button = tk.Button(self, text="Exit", command=self.exit_program, font=("Arial", 16),
                                        width=20, height=1, bg="#FF3131", fg="white", borderwidth=2, relief="flat")
        exit_program_button.place(x=500, y=675)
        
        # Upload Button
        upload_image_button = tk.Button(self, text="Upload Image", command=self.upload_image, font=("Arial", 16),
                                  width=30, height=2, bg="#1B4A89", fg="white", borderwidth=2, relief="flat")
        upload_image_button.place(x=900, y=325)
        
        # Process Button
        process_image_button = tk.Button(self, text="Process Image", command=self.process_image, font=("Arial", 16),
                                   width=30, height=2, bg="#1B4A89", fg="white", borderwidth=2, relief="flat")
        process_image_button.place(x=900, y=425)  
        
        # Clear Button
        clear_image_button = tk.Button(self, text="Clear Image", command=self.clear_image, font=("Arial", 16),
                                       width=30, height=2, bg="#1B4A89", fg="white", borderwidth=2, relief="flat")
        clear_image_button.place(x=900, y=525) 
        
        
        # Instance variable to store the video path
        self.video_path = None
        
    #Upload Image
    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
        if file_path:
            # Store the file path in the instance variable for later use
            self.image_path = file_path
            print("Selected Image File:", self.image_path)

            # Display the uploaded image
            self.display_uploaded_image()
            
            # Show a popup box
            messagebox.showinfo("Image Uploaded", "Image has been successfully uploaded!")
            
    
    #Disaply Upload
    def display_uploaded_image(self):
        if hasattr(self, 'uploaded_image_label') and isinstance(self.uploaded_image_label, tk.Label):
            # Destroy the label widget
            self.uploaded_image_label.destroy()

         # Hide the placeholder image
        self.placeholder_label.place_forget()

        # Open and resize the uploaded image
        uploaded_image = Image.open(self.image_path)

        # Get the dimensions of the uploaded image
        uploaded_width, uploaded_height = uploaded_image.size

        # Resize the placeholder image to match the dimensions of the uploaded image
        resized_placeholder_image = uploaded_image.resize((550, 400), Image.LANCZOS)

        # Convert the Pillow image to PhotoImage
        self.tk_uploaded_image = ImageTk.PhotoImage(resized_placeholder_image)

        # Create a label for the uploaded image or update the existing label
        if not hasattr(self, 'uploaded_image_label'):
            self.uploaded_image_label = tk.Label(self, image=self.tk_uploaded_image)
            self.uploaded_image_label.place(x=200, y=250)  # Adjust the coordinates as needed
        else:
            self.uploaded_image_label.config(image=self.tk_uploaded_image)
            self.uploaded_image_label.image = self.tk_uploaded_image  # Keep a reference to the image
        
        
    #Process Image
    def process_image(self):
        if not hasattr(self, 'image_path') or not self.image_path:
            # Display a popup message if no image is uploaded
            messagebox.showwarning("No Image", "Please upload an image first.")
            return
        
        # Load the image using OpenCV
        frame = cv2.imread(self.image_path)

        # Check if the frame is not None
        if frame is not None:
            # Load the YOLO model
            model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'last.pt')
            model = YOLO(r"C:\Users\Aldrex\Documents\Thesis\dnn_model\best.pt")

            threshold = 0.2

            # Object Detection
            results = model(frame)[0]

            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result

                color = (0, 255, 0)  # Default color (green) for normal_pig

                if score > threshold:
                    if results.names[int(class_id)] == "negative_pig":
                        color = (0, 0, 255)  # Change color to red for negative_pig

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)
                    cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3, cv2.LINE_AA)


            # Display the processed image
            cv2.imshow("Processed Image", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Automatically save the processed image
            self.save_processed_image(frame)

            # Show a message box after the processed image window is closed
            messagebox.showinfo("Image Saved", f"Processed image saved at Pictures/Processed Images")

    def save_processed_image(self, cv2_image):
        # Define the directory where you want to save the processed images
        save_directory = os.path.join(os.path.expanduser('~'), 'Pictures', 'Processed Images')

        # Ensure the directory exists, create it if not
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Get the current date and time for the file name
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Define the save path for the processed image with date and time
        save_path = os.path.join(save_directory, f'processed_image_{current_datetime}.jpg')

        # Save the processed image
        cv2.imwrite(save_path, cv2_image)

        # Return the path where the image is saved
        return save_path

    
    #Clear Upload
    def clear_image(self):
        if hasattr(self, 'uploaded_image_label') and isinstance(self.uploaded_image_label, tk.Label):
            # Destroy the label widget
            self.uploaded_image_label.destroy()
            # Set uploaded_image_label to None to indicate it needs to be recreated
            self.uploaded_image_label = None

        # Clear the image_path variable
        self.image_path = None

        # Set the scale factor to 0.5 for resizing
        scale_factor = 0.5

        # Calculate the new width and height based on the scale factor
        new_width = int(self.original_width * scale_factor) if hasattr(self, 'original_width') else 0
        new_height = int(self.original_height * scale_factor) if hasattr(self, 'original_height') else 0

        # Update the dimensions to ensure positive values
        new_width = max(new_width, 1)
        new_height = max(new_height, 1)

        # Resize the original image if dimensions are available
        if new_width > 0 and new_height > 0:
            resized_uploaded_image = self.resized_bg.resize((new_width, new_height), Image.LANCZOS)
            # Convert the Pillow image to PhotoImage
            self.tk_uploaded_image = ImageTk.PhotoImage(resized_uploaded_image)

            # Recreate the label for the uploaded image
            self.uploaded_image_label = tk.Label(self, image=self.tk_uploaded_image)
            self.uploaded_image_label.place(x=50, y=300)  # Adjust the coordinates as needed

    
    #Other Defs
    def button_process_image_action(self):
        # Hide the root window
        root.withdraw()

        # Create the image processing window and maximize it
        image_processing_window = ImageProcessingWindow(root)

        # Destroy the root window when the image processing window is closed
        image_processing_window.protocol("WM_DELETE_WINDOW", lambda: self.on_close(image_processing_window))

        # Start the Tkinter main event loop
        image_processing_window.mainloop()

    def on_close(self, window):
        # Destroy the image processing window
        window.destroy()

        # Destroy the root window
        root.destroy()
        
    def back_to_home(self):
        # Implement the action to go back to the main window
        self.destroy()
        root.state('zoomed')
        root.deiconify()

    def exit_program(self):
        # Implement the action to exit the program
        self.destroy()
        root.destroy()
        cursor.close()
        conn.close()

#==============================================================================================

def exit_program():
    root.destroy()  # Close the home page window
    cursor.close()
    conn.close()

#ABOUT WINDOW
def open_about():
    about_window = AboutWindow(root)

    # Set the desired size for the AboutWindow (width x height)
    about_window.geometry("800x600")

    # Center the about window on the screen
    about_window.update_idletasks()
    width = about_window.winfo_width()
    height = about_window.winfo_height()
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    about_window.geometry("+{}+{}".format(x, y))

    # Start the event loop for the about window
    about_window.mainloop()

# Create the main window
root = tk.Tk()

# Set window title
root.title("Unhealthy Swine Detector")

# Maximize the window
root.state('zoomed')

# Set the background image
background_image = ImageTk.PhotoImage(Image.open(r"C:\Users\Aldrex\Documents\Thesis\pictures\bg.png"))
background_label = tk.Label(root, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# Load and resize the image using Pillow
image_path = r"C:\Users\Aldrex\Documents\Thesis\pictures\pic_1.png"
original_image = Image.open(image_path)
resized_image = original_image.resize((550, 700), Image.LANCZOS)

# Convert the Pillow image to PhotoImage
tk_image = ImageTk.PhotoImage(resized_image)

# Create a label for the resized image
image_label = tk.Label(root, image=tk_image)
image_label.place(x=70, y=50)

# Load and resize the image using Pillow
image_path_1 = r"C:\Users\Aldrex\Documents\Thesis\pictures\icon.png"
original_image_1 = Image.open(image_path_1)
resized_image_1 = original_image_1.resize((110, 110), Image.LANCZOS)

# Convert the Pillow image to PhotoImage
tk_image_1 = ImageTk.PhotoImage(resized_image_1)

# Create a label for the resized image
image_label_1 = tk.Label(root, image=tk_image_1)
image_label_1.place(x=1250, y=180)

# Get the screen width and height
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Create a label
label = tk.Label(root, text="Unhealthy Swine", font=("Arial", 60, "bold"))
label.place(x=760, y=75)  

label = tk.Label(root, text="Detector", font=("Arial", 75, "bold"))
label.place(x=820, y=175)  

#Camera Button
# Load and prepare the image
camera_logo_image_path = r"C:\Users\Aldrex\Documents\Thesis\pictures\camera_icon.png"
camera_logo_image = Image.open(camera_logo_image_path)
camera_logo_image = camera_logo_image.resize((100, 100), Image.LANCZOS)  # Resize the image to fit the button
photo = ImageTk.PhotoImage(camera_logo_image)

# Create the button with image on top and text on bottom
button_open_camera = tk.Button(root, image=photo, command=open_camera, text="\n\nOpen Camera", compound="top", font=("Arial", 18), width=320, height=300, bg="#002659", fg="white", borderwidth=2, relief="flat")
button_open_camera.photo = photo  # Keep a reference to the image to ensure it shows
button_open_camera.place(x=730, y=340)

# Create About button
button_settings = tk.Button(root, text="About", command=open_about, font=("Arial", 16), width=27, height=2, bg="#545454", fg="white", borderwidth=1, relief="flat")
button_settings.place(x=730, y=665)   

# Create exit button
button_exit = tk.Button(root, text="Exit", command=exit_program, font=("Arial", 16), width=27, height=2, bg="#FF3131", fg="white", borderwidth=1, relief="flat")
button_exit.place(x=1080, y=665)

# Create a button to open the VideoProcessingWindow
button_process_video = tk.Button(root, text="Process a Video", command=lambda: VideoProcessingWindow(root),font=("Arial", 16), width=27, height=5, bg="#002659", fg="white",borderwidth=2, relief="flat")
button_process_video.place(x=1080, y=340)

# Create a button to open the ImageProcessingWindow
button_process_video = tk.Button(root, text="Process an Image", command=lambda: ImageProcessingWindow(root),font=("Arial", 16), width=27, height=5, bg="#002659", fg="white",borderwidth=2, relief="flat")
button_process_video.place(x=1080, y=508)

# Calculate the weights for rows and columns
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
root.grid_rowconfigure(2, weight=1)
root.grid_columnconfigure(0, weight=1)

# ABOUT WINDOW
class AboutWindow(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("About")

        # Load and resize a background image using Pillow
        about_bg_path = r"C:\Users\Aldrex\Documents\Thesis\bg.png"
        about_bg_image = Image.open(about_bg_path)
        self.resized_about_bg = about_bg_image.resize((800, 600), Image.LANCZOS)

        # Convert the Pillow image to PhotoImage
        self.tk_about_bg = ImageTk.PhotoImage(self.resized_about_bg)

        # Create a canvas for the background image
        self.canvas = tk.Canvas(self, width=800, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Place the background image on the canvas
        self.bg_image_item = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_about_bg)

        # Add a picture
        picture_path = r"C:\Users\Aldrex\Documents\Thesis\pictures\icon.png"
        original_picture = Image.open(picture_path)
        resized_picture = original_picture.resize((100, 100), Image.LANCZOS)

        # Convert the Pillow image to PhotoImage
        tk_picture = ImageTk.PhotoImage(resized_picture)

        # Create a label for the picture
        picture_label = tk.Label(self, image=tk_picture)
        picture_label.image = tk_picture 
        picture_label.place(x=660, y=25)
        
        # Add a picture
        picture_path_1 = r"C:\Users\Aldrex\Documents\Thesis\pictures\pic_2.png"
        original_picture_1 = Image.open(picture_path_1)
        resized_picture_1 = original_picture_1.resize((470, 170), Image.LANCZOS)

        # Convert the Pillow image to PhotoImage
        tk_picture_1 = ImageTk.PhotoImage(resized_picture_1)

        # Create a label for the picture
        picture_label_1 = tk.Label(self, image=tk_picture_1)
        picture_label_1.image = tk_picture_1
        picture_label_1.place(x=160, y=150)

        # Add text content
        about_label = tk.Label(self, text="Unhealthy Swine Detector", font=("Arial", 40), bg='white', fg='black')
        about_label.place(x=40, y=30)

        about_description = tk.Label(self, text="All About Us", font=("Arial", 25), bg='white', fg='black', wraplength=600, justify=tk.CENTER)
        about_description.place(x=290, y=100)
        
        about_text = tk.Label(self, text = "The computer science students from West Visayas State University, Karen Arroyo, Ellan Flores, Mary Ruth Lusuegro, Rey Matthew Parreno, and Aldrex Mark Tingatinga, have collaborated on a project that aims to develop an application using computer vision technology to analyze swine behavior. The project's design displays a forward-thinking approach, suggesting the possibility of expanding the application's capabilities to analyze the behavior of other animals.",
                              font=("Arial", 11), bg='white', fg='black', wraplength=750, justify=tk.LEFT)
        about_text.place(x=30, y=350)
        
        about_text = tk.Label(self, text = "This project combines technology and agriculture to enhance animal welfare and farm management practices, serving as a valuable tool for monitoring and detecting negative behavior in swine, empowering the team to make informed decisions about their livestock, and recognizing the potential for their work to impact various aspects of animal agriculture.",
                              font=("Arial", 11), bg='white', fg='black', wraplength=750, justify=tk.LEFT)
        about_text.place(x=30, y=450)

        # Create a button to close the "About Us" window
        close_button = tk.Button(self, text="Close", command=self.destroy, font=("Arial", 16), width=15, height=1, bg="#1B4A89", fg="white", borderwidth=2, relief="flat")
        close_button.place(x=300, y=550)

        # Bind the window resize event
        self.bind("<Configure>", self.on_resize)

    def on_resize(self, event):
        # Update canvas size and redraw background only when necessary
        if event.width != self.canvas.winfo_width() or event.height != self.canvas.winfo_height():
            self.canvas.config(width=event.width, height=event.height)
            self.redraw_background()

    def redraw_background(self):
        # Resize the background image and update canvas
        self.resized_about_bg = self.resized_about_bg.resize((self.canvas.winfo_width(), self.canvas.winfo_height()), Image.LANCZOS)
        self.tk_about_bg = ImageTk.PhotoImage(self.resized_about_bg)
        self.canvas.itemconfig(self.bg_image_item, image=self.tk_about_bg)
        
# Calculate the center position for the window
x = int((screen_width - root.winfo_reqwidth()) / 2)
y = int((screen_height - root.winfo_reqheight()) / 2)

# Set the window position
root.geometry("+{}+{}".format(x, y))

# Start the main event loop
root.mainloop()
