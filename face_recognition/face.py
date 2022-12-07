import face_recognition
import cv2
import numpy as np

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
reza_image = face_recognition.load_image_file("image_dataset/reza.jpeg")
reza_face_encoding = face_recognition.face_encodings(reza_image)[0]

# Load 2nd sample picture and learn how to recognize it.
sumon_image = face_recognition.load_image_file("image_dataset/sumon.jpeg")
sumon_face_encoding = face_recognition.face_encodings(sumon_image)[0]

# Load 3rd sample picture and learn how to recognize it.
tanbir_image = face_recognition.load_image_file("image_dataset/tanbir.jpeg")
tanbir_face_encoding = face_recognition.face_encodings(tanbir_image)[0]

# Load 4th sample picture and learn how to recognize it.
tushar_image = face_recognition.load_image_file("image_dataset/tushar.jpeg")
tushar_face_encoding = face_recognition.face_encodings(tushar_image)[0]

# Load 5th sample picture and learn how to recognize it.
liton_image = face_recognition.load_image_file("image_dataset/liton.jpeg")
liton_face_encoding = face_recognition.face_encodings(liton_image)[0]

# Load 6th sample picture and learn how to recognize it.
galib_image = face_recognition.load_image_file("image_dataset/galib.jpeg")
galib_face_encoding = face_recognition.face_encodings(galib_image)[0]

# Load 7th sample picture and learn how to recognize it.
ali_image = face_recognition.load_image_file("image_dataset/ali.jpeg")
ali_face_encoding = face_recognition.face_encodings(ali_image)[0]

# Load 8th sample picture and learn how to recognize it.
nusrat_image = face_recognition.load_image_file("image_dataset/nusrat.jpeg")
nusrat_face_encoding = face_recognition.face_encodings(nusrat_image)[0]

# Load 9th sample picture and learn how to recognize it.
kamal_image = face_recognition.load_image_file("image_dataset/kamal.jpeg")
kamal_face_encoding = face_recognition.face_encodings(kamal_image)[0]

# Load 10th sample picture and learn how to recognize it.
shahida_image = face_recognition.load_image_file("image_dataset/shahida.jpeg")
shahida_face_encoding = face_recognition.face_encodings(shahida_image)[0]

# Load 11th sample picture and learn how to recognize it.
rakib_image = face_recognition.load_image_file("image_dataset/rakib.jpeg")
rakib_face_encoding = face_recognition.face_encodings(rakib_image)[0]

# Load 12th sample picture and learn how to recognize it.
runa_image = face_recognition.load_image_file("image_dataset/runa.jpeg")
runa_face_encoding = face_recognition.face_encodings(runa_image)[0]

# Load 13th sample picture and learn how to recognize it.
majid_image = face_recognition.load_image_file("image_dataset/majid.jpeg")
majid_face_encoding = face_recognition.face_encodings(majid_image)[0]

# Load 14th sample picture and learn how to recognize it.
mita_image = face_recognition.load_image_file("image_dataset/mita.jpeg")
mita_face_encoding = face_recognition.face_encodings(mita_image)[0]

# Load 15th sample picture and learn how to recognize it.
sanjida_image = face_recognition.load_image_file("image_dataset/sanjida.jpeg")
sanjida_face_encoding = face_recognition.face_encodings(sanjida_image)[0]



# Create arrays of known face encodings and their names
known_face_encodings = [
    reza_face_encoding,
    sumon_face_encoding,
    tanbir_face_encoding,
    tushar_face_encoding,
    liton_face_encoding,
    galib_face_encoding,
    ali_face_encoding,
    nusrat_face_encoding,
    kamal_face_encoding,
    shahida_face_encoding,
    rakib_face_encoding,
    runa_face_encoding,
    majid_face_encoding,
    mita_face_encoding,
    sanjida_face_encoding,
    
]
known_face_names = [
    "Reza",
    "Sumon",
    "Tanbir",
    "Tushar",
    "Liton",
     "Galib",
     "Ali",
     "Nusrat",
     "Kamal",
      "Shahida",
      "Rakib",
      "Runa",
     "Majid",
     "Mita",
      "Sanjida"
    
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

       
            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
