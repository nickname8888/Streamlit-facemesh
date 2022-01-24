from asyncio.windows_events import NULL
import streamlit as st 
import mediapipe as mp 
import cv2
import numpy as np 
import time
import tempfile
from PIL import Image

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
DEMO_IMAGE = 'demo.jpeg'
DEMO_VIDEO = 'demo.mp4'

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
                                   ['About', 'Run on Image', 'Run on Video']
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

# st.video('https://www.youtube.com/watch?v=zkw9aI1-feM')

elif app_mode == 'Run on Image': 
    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)
    
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
    
    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)
    
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