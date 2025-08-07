import os
import time
import numpy as np
import cv2
import face_recognition
import pickle
from datetime import datetime
import csv
import matplotlib.pyplot as plt

class Colour:
    hm = {
        "cr": "\033[0m",  # Reset
        "r": "\033[31m",  # Red
        "g": "\033[32m",  # Green
        "y": "\033[93m",  # Yellow
        "b": "\033[34m",  # Blue
        "h": "\033[7m",   # Highlight
        "bo": "\033[1m",  # Bold (Bright)
        "i": "\033[3m",   # Italic
        "u": "\033[4m",   # Underline
        "w": "\033[37m",  # White
        "yb": "\033[43m", # Yellow Background
        "bb": "\033[40m", # Black Background
        "bw": "\033[47m", # White Background
        "cs": "\033[2J"   # Clear Screen
    }

    @staticmethod
    def change_color(s):
        for key in s.split(","):
            print(Colour.hm.get(key, ""), end="")



class Attendance_System:

    def __init__(self):
        self.main()

    def ensure_directory(self,path):
        try:
            if not os.path.exists(path):
                os.makedirs(path)
        except Exception as e:
            print("Error:"+e)

    def take_attendance(self):
        try:
            Colour.change_color("bo,u")
            batch = input("\n📁 Enter batch name: ")
            Colour.change_color("cr")
            Colour.change_color("bo") 
            subject = input("👉Enter subject name: ")
            Colour.change_color("cr")
            heading=True

            months = {
                1: "January", 2: "February", 3: "March", 4: "April",
                5: "May", 6: "June", 7: "July", 8: "August",
                9: "September", 10: "October", 11: "November", 12: "December"
            }

            current_month_number = datetime.today().month
            batch_path = f"attendance/{batch}/{months[current_month_number]}"
            self.ensure_directory(batch_path)
            file_name = f"{batch_path}/{subject}_{datetime.today().date()}.csv"
            
            known_face_encodings = self.load_encodings()
            known_face_names = ["Divy", "Elon", "Manasvi", "Meet", "SRK"]
            student_list = known_face_names.copy()
            
            blink_status = {name: {"eyes_closed_frames": 0, "blink_detected": False, "attendance_marked": False} for name in known_face_names}
            
            cap = cv2.VideoCapture(0)
            while True:
                ret, frame = cap.read()
                if not ret:
                    Colour.change_color("r,bo")
                    print("Failed to grab frame.")
                    Colour.change_color("cr")
                    time.sleep(1)
                    os.system('cls' if os.name == 'nt' else 'clear') 
                    break
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                face_landmarks_list = face_recognition.face_landmarks(rgb_frame)
                
                for (top, right, bottom, left), face_encoding, landmarks in zip(face_locations, face_encodings, face_landmarks_list):
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"
                    
                    if True in matches:
                        distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        best_match_index = np.argmin(distances)
                        if matches[best_match_index]:
                            name = known_face_names[best_match_index]
                    
                    if name not in blink_status:
                        blink_status[name] = {"eyes_closed_frames": 0, "blink_detected": False, "attendance_marked": False}

                    def eye_aspect_ratio(eye):
                        A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))  
                        B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))  
                        C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))  
                        return (A + B) / (2.0 * C)

                    left_eye = landmarks["left_eye"]
                    right_eye = landmarks["right_eye"]
                    left_ear = eye_aspect_ratio(left_eye)
                    right_ear = eye_aspect_ratio(right_eye)
                    avg_ear = (left_ear + right_ear) / 2.0
                    
                    BLINK_THRESHOLD = 0.20
                    BLINK_FRAMES = 3
                    
                    if avg_ear < BLINK_THRESHOLD:
                        blink_status[name]["eyes_closed_frames"] += 1
                    else:
                        if blink_status[name]["eyes_closed_frames"] >= BLINK_FRAMES:
                            blink_status[name]["blink_detected"] = True
                        blink_status[name]["eyes_closed_frames"] = 0
                    
                    if blink_status[name]["blink_detected"] and not blink_status[name]["attendance_marked"] and name in student_list:
                        Colour.change_color("g,bo")
                        print(f"{name} blinked! Marking attendance.")
                        with open(file_name, mode='w', newline='') as file:
                            writer = csv.writer(file)
                            if heading:
                                writer.writerow(["Name", "Status", "Time"])
                                heading=False
                            writer.writerow([name, "Present", datetime.now().strftime("%H:%M:%S")])
                        student_list.remove(name)
                        blink_status[name]["blink_detected"] = False  
                        blink_status[name]["attendance_marked"] = True  
                    
                    color = (0, 255, 0) if blink_status[name]["attendance_marked"] else (0, 255, 255)
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    if not blink_status[name]["attendance_marked"]:
                        cv2.putText(frame, "Blink to mark attendance", (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    else:
                        cv2.putText(frame, "Attendance Marked", (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                cv2.imshow("Attendance System", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            with open(file_name, mode='a', newline='') as file:
                writer = csv.writer(file)
                for student in student_list:
                    writer.writerow([student, "Absent", "--"])
                writer.writerow([])
                writer.writerow(["Total","Present","Absent"])
                writer.writerow([len(known_face_names),len(known_face_names)-len(student_list),len(student_list)])
                heading=True
            
            cap.release()
            cv2.destroyAllWindows()
            print(f"Attendance saved to {file_name}")
            Colour.change_color("cr")
            time.sleep(2)
            os.system('cls' if os.name == 'nt' else 'clear')    


        except Exception as e:
            Colour.change_color("r,bo")
            print("Invalid entry")
            Colour.change_color("cr")
            time.sleep(1)
            os.system('cls' if os.name == 'nt' else 'clear') 
        
    def load_encodings(self):
        try:
            with open("face_encodings.pkl", "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            Colour.change_color("r,bo")
            print("No known face encodings found. Starting fresh.")
            Colour.change_color("cr")
            return []
        
    def view_attendance_analytics(self):
        try:
            Colour.change_color("bo,u")
            batch = input("\n📁 Enter batch name: ")
            Colour.change_color("cr")

            months = {
                    1: "January", 2: "February", 3: "March", 4: "April",
                    5: "May", 6: "June", 7: "July", 8: "August",
                    9: "September", 10: "October", 11: "November", 12: "December"
                }
            Colour.change_color("bo,u")  # Bold & Underline
            month_number = int(input("\n📅 Enter month number (1-12): "))
            Colour.change_color("cr")  # Reset
                
            batch_path = f"attendance/{batch}/{months[month_number]}"
            if not os.path.exists(batch_path):
                Colour.change_color("r,bo")  # Red & Bold
                print("\n❌ Batch folder for the given month does not exist.")
                Colour.change_color("cr")
                time.sleep(1)
                os.system('cls' if os.name == 'nt' else 'clear')
                return
            
            Colour.change_color("y,bo")  # Yellow & Bold
            print("\n📊 Attendance Options:")
            print("[1] Single Subject")
            print("[2] All Subjects")
            Colour.change_color("cr")
            Colour.change_color("bo") 
            choice=int(input("Enter choice: "))
            Colour.change_color("cr")
            if choice==1:
                Colour.change_color("bo") 
                subject = input("👉Enter subject name: ")
                Colour.change_color("cr")

                total_students = 0
                total_present = 0
                total_absent = 0
                
                for file in os.listdir(batch_path):
                    if file.startswith(subject):
                        file_path = os.path.join(batch_path, file)
                        with open(file_path, "r") as f:
                            lines = f.readlines()
                            if len(lines) > 1:
                                last_line = lines[-1].strip().split(",")
                                total_students = int(last_line[0])
                                total_present += int(last_line[1])
                                total_absent += int(last_line[2])
                
                
                if total_students == 0:
                    Colour.change_color("r,bo")
                    print("\n❌ No attendance records found for the given month/subject.")
                    Colour.change_color("cr")
                    time.sleep(1)
                    os.system('cls' if os.name == 'nt' else 'clear')    
                    return
                
                labels = ["Present", "Absent"]
                sizes = [total_present, total_absent]
                colors = ["green", "red"]
                
                plt.figure(figsize=(6, 6))
                plt.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=140)
                plt.title(f"Attendance Distribution for {subject} ({months[month_number]})\nTotal Students: {total_students}")
                plt.gcf().canvas.manager.set_window_title(f"{batch} {months[month_number]} {subject} Subject")
                plt.show()
                os.system('cls' if os.name == 'nt' else 'clear')
                

            elif choice==2:
                subject_attendance = {}
            
                for file in os.listdir(batch_path):
                    subject = file.split("_")[0]  # Extract subject name from file name
                    file_path = os.path.join(batch_path, file)
                    with open(file_path, "r") as f:
                        lines = f.readlines()
                        if len(lines) > 1:
                            last_line = lines[-1].strip().split(",")
                            total_present = int(last_line[1])
                            if subject in subject_attendance:
                                subject_attendance[subject] += total_present
                            else:
                                subject_attendance[subject] = total_present
                
                if not subject_attendance:
                    print("No attendance records found for the given month.")
                    return
                
                subjects = list(subject_attendance.keys())
                attendance_counts = list(subject_attendance.values())
                
                plt.figure(figsize=(8, 8))
                plt.pie(attendance_counts, labels=subjects, autopct="%1.1f%%", startangle=140)
                plt.title(f"Attendance Distribution for {months[month_number]}")
                plt.gcf().canvas.manager.set_window_title(f"{batch} {months[month_number]} All Subject")
                plt.show()
                os.system('cls' if os.name == 'nt' else 'clear')

            else:
                print("invalid choice")
        except Exception as e:
            Colour.change_color("r,bo")
            print("❌Invalid entry")
            Colour.change_color("cr")
            time.sleep(2)
            os.system('cls' if os.name == 'nt' else 'clear')
        
    def main(self):
        try:
            while True:
                Colour.change_color("y,bo")  
                print("\n╔═════════════════════════════════════╗")
                Colour.change_color("r,bo,bw")  
                print("║  FACE RECOGNITION ATTENDANCE SYSTEM ║")
                Colour.change_color("cr")  
                Colour.change_color("y,bo")  
                print("╚═════════════════════════════════════╝")
                Colour.change_color("cr")  

                Colour.change_color("g,bo")  
                print("\n[1] Take Attendance")
                print("[2] View Attendance Analytics")
                Colour.change_color("y,bo")  
                print("[3] Exit")
                Colour.change_color("cr")  

                Colour.change_color("bo")  
                choice = input("\n👉 Enter your choice: ")
                Colour.change_color("cr")  
                
                os.system('cls' if os.name == 'nt' else 'clear')
                if choice == "1":
                    Colour.change_color("g,bo")
                    print("\n✔ Taking Attendance...")
                    Colour.change_color("cr")
                    self.take_attendance()

                elif choice == "2":
                    Colour.change_color("b,bo")
                    print("\n📊 Viewing Attendance Analytics...")
                    Colour.change_color("cr")
                    self.view_attendance_analytics()

                elif choice == "3":
                    Colour.change_color("r,bo")
                    print("\n👋 Good Bye. Have a great day!")
                    Colour.change_color("cr")
                    break

                else:
                    Colour.change_color("r,bo")
                    print("\n❌ Invalid choice. Please try again.")
                    Colour.change_color("cr")
                    time.sleep(2)
                    os.system('cls' if os.name == 'nt' else 'clear')

        except Exception as e:
            Colour.change_color("r,bo")
            print("\n⚠️ Invalid entry. Please enter a valid choice!")
            Colour.change_color("cr")
            time.sleep(2)
            os.system('cls' if os.name == 'nt' else 'clear')
        

obj=Attendance_System()