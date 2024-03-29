from asyncio.windows_events import NULL
import streamlit as st 
import mediapipe as mp 
import cv2
import numpy as np 
import time
import tempfile
from PIL import Image
from tensorflow import keras
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode


mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
DEMO_IMAGE = 'demo.jpeg'
DEMO_VIDEO = 'demo.avi'

# load model
emotion_dict = {0:'angry', 1 :'happy', 2: 'neutral', 3:'sad', 4: 'surprise'}
# load json and create model
json_file = open('emotion_model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)

# load weights into new model
classifier.load_weights("emotion_model1.h5")

#load face
try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class Faceemotion(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        #image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout)
            label_position = (x, y)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img


st.title('Face Mesh Application')

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 350px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 350px
        margin-left: -350px 
    }
    </style>
    """,
    unsafe_allow_html = True
)

st.sidebar.title("Facemesh sidebar")
st.sidebar.subheader("Parameters")

@st.cache()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    
    if width is None and height is None: 
        return image 
    
    if width is None: 
        r = width/float(w)
        dim = (int(w*r), height)
        
    else: 
        r = width/float(w)
        dim = (width, int(h*r))
        
    # Resizing the Image
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

app_mode = st.sidebar.selectbox('Choose the App Mode', 
                                   ['About', 'Run on Image', 'Run on Video', 'Webcam Face Detection']
                                   )

if app_mode == 'About': 
    st.markdown(
        "Made by nickname8888.\n This Application is made using the MediaPipe framework which is used for building multimodal applied machine learning pipelines.\n The web app is built using Streamlit which is an an open-source python framework for building web apps for Machine Learning and Data Science."
    )

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width: 350px
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width: 350px
            margin-left: -350px 
        }
        </style>
        """,
        unsafe_allow_html = True
    )

elif app_mode == 'Run on Image': 
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    
    st.sidebar.markdown('---')
    
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width: 350px
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width: 350px
            margin-left: -350px 
        }
        </style>
        """,
        unsafe_allow_html = True
    )
    
    st.markdown("**Detect Faces**")
    
    max_faces = st.sidebar.number_input("Maximum Faces: ", value=2, min_value=1)
    st.markdown("---")
    
    detection_confidence = st.sidebar.slider("Minimum Detection Confiednce", min_value=0.0, value=0.5, max_value=1.0)
    st.markdown("---")
    
    img_file = st.sidebar.file_uploader("Upload an Image", type=["png","jpg","jpeg"])
    if img_file is not None: 
        image = np.array(Image.open(img_file))
    else: 
        demo_image = DEMO_IMAGE
        image = np.array(Image.open(DEMO_IMAGE))
    
    st.sidebar.text('Original Image')
    st.sidebar.image(image)
    
    face_count = 0
    
    ## Dashboard
    with mp_face_mesh.FaceMesh(
        static_image_mode = True,
        max_num_faces = max_faces,
        min_detection_confidence = detection_confidence
    ) as face_mesh: 
        
        results = face_mesh.process(image)
        out_image = image.copy()
        
        ## Face landmark drawing
        
        for face_landmark in results.multi_face_landmarks: 
            face_count += 1 
            
            mp_drawing.draw_landmarks(
                image = out_image,
                landmark_list = face_landmark,
                connections = mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec = drawing_spec 
            )
            
        st.subheader('Output Image')
        st.image(out_image, use_column_width=True)
        

elif app_mode == 'Run on Video': 
    st.set_option("deprecation.showfileUploaderEncoding", False)
    
    use_webcam = st.sidebar.button("Use Webcam")
    record = st.sidebar.checkbox("Record Video")
    
    if record: 
        st.checkbox("Recording", value=True)
    
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width: 350px
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width: 350px
            margin-left: -350px 
        }
        </style>
        """,
        unsafe_allow_html = True
    )
        
    max_faces = st.sidebar.number_input("Maximum Faces: ", value=5, min_value=1)
    st.sidebar.markdown("---")
    
    tracking_confidence = st.sidebar.slider("Minimum Tracking Confidence", min_value=0.0, value=0.5, max_value=1.0)
    st.sidebar.markdown("---")
    
    detection_confidence = st.sidebar.slider("Minimum Detection Confidence", min_value=0.0, value=0.5, max_value=1.0)
    st.sidebar.markdown("---")
    
    st.markdown("## Output")
    
    stframe = st.empty()
    video_file = st.sidebar.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])
    tffile = tempfile.NamedTemporaryFile(delete=False)
    
    # Getting input video file from here.
    if not video_file: 
        if use_webcam: 
            video = cv2.VideoCapture(0)
        else:
            video = cv2.VideoCapture(DEMO_VIDEO)
            tffile.name = DEMO_VIDEO
    else: 
        tffile.write(video_file.read())
        video = cv2.VideoCapture(tffile.name)
    
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(video.get(cv2.CAP_PROP_FPS))
    
    # Recording part! 
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('Output.mp4', codec, fps_input, (width, height))
    
    st.sidebar.text("Input Video")
    st.sidebar.video(tffile.name)
    
    fps = 0
    i = 0
    
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    kpi1, kpi2, kpi3 = st.columns(3)
    
    with kpi1:
        st.markdown("**Frame Rate**")
        kpi1_text = st.markdown("0")
    
    with kpi2:
        st.markdown("**Detected Faces**")
        kpi2_text = st.markdown("0")
    
    with kpi3:
        st.markdown("**Image Width**")
        kpi3_text = st.markdown("0")
    
    st.markdown("<hr>/", unsafe_allow_html=True)
    
    face_count = 0
    
    with mp_face_mesh.FaceMesh(
        min_tracking_confidence = tracking_confidence,
        max_num_faces = max_faces,
        min_detection_confidence = detection_confidence
    ) as face_mesh: 
        
        prevTime = 0
        
        while video.isOpened(): 
            i += 1
            ret, frame = video.read()
            
            if not ret : 
                continue 
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame)
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            face_count = 0
            if results.multi_face_landmarks: 
                for face_landmark in results.multi_face_landmarks: 
                    face_count += 1 
                    
                    mp_drawing.draw_landmarks(
                        image = frame,
                        landmark_list = face_landmark,
                        connections = mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec = drawing_spec,
                        connection_drawing_spec = drawing_spec
                    )
        
                # FPS counter
                currTime = time.time()
                fps = 1/(currTime - prevTime)
                prevTime = currTime
                    
                if record: 
                    out.write(frame)
                    
                frame = cv2.resize(frame, (0,0), fx=0.8, fy=0.8)
                frame = image_resize(image=frame, width=640)
                stframe.image(frame, channels='BGR', use_column_width=True)

elif app_mode == 'Webcam Face Detection': 
    st.header("Webcam Live Feed")
    st.write("Click on start to use webcam and detect your face emotion")
    webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                    video_processor_factory=Faceemotion)

    
    