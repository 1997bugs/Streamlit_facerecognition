import streamlit as st
import cv2
import s3fs
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
#from mtcnn.mtcnn import MTCNN
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
#from deepface import DeepFace
from facenet_pytorch import MTCNN
from PIL import Image
from torchvision.utils import save_image
from numpy import asarray
from itertools import cycle
from scipy.spatial.distance import cosine
from keras.utils.layer_utils import get_source_inputs
#from keras.engine import  Model
from keras.layers import Input
from torchvision import transforms
#from keras_vggface.vggface import VGGFace
#from keras_vggface.utils import preprocess_input
import os
from mtcnn.mtcnn import MTCNN as mtcnn_det

os.environ['KMP_DUPLICATE_LIB_OK']='True'
fs = s3fs.S3FileSystem(anon=False)

st.title("Face Recongnition ECC Project")

st.write(""" Shriya-Karthik-Ujwala """)

def extract_face(file):
    pixels = asarray(file)
    plt.axis("off")
    plt.imshow(pixels)
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    x1, y1, width, height = results[0]["box"]
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize((224, 224))
    face_array = asarray(image)
    return face_array
# Load - +- images
# Detect faces - display all recognized faces and save them 

def main():

    add_selectbox = st.sidebar.selectbox(
        "Results you want to look at",
        ("About Project ", "Login Page - Load Images", "Face Detection", "Verify Face")
    )
    images =[]
    if add_selectbox == "About Project ":
        st.write("Part of ECC Course - Blah Blah some introduction")
        st.text('Part of ECC Course - Blah Blah some introduction')
        im = cv2.imread("group.JPG")
        im=cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        st.image(im)
	

    if add_selectbox == "Login Page - Load Images":
        st.subheader("Load Images Here")
        uploaded_imgs = st.file_uploader("Upload Images", type=['jpg', 'png', 'jpeg'],accept_multiple_files = True)
        images = uploaded_imgs
        if len(uploaded_imgs) > 0:
            caption = []
            for x in uploaded_imgs:
            #caption = list(range(0,len(uploaded_imgs),1)) # your caption here
                caption.append(x.name)
            cols = cycle(st.columns(4)) # st.columns here since it is out of beta at the time I'm writing this
            for idx, filteredImage in enumerate(uploaded_imgs):
                next(cols).image(filteredImage, width=150, caption=caption[idx])

                # Create face detector
            mtcnn = MTCNN(margin=20, keep_all=True, post_process=False) #device='cuda:0'
            face_arr = []
            faces_all = []
            # Detect faces in batch
            for im in uploaded_imgs:
                img = np.array(Image.open(im))
                frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                faces = mtcnn(frame)
                #face_new = mtcnn_det.detect_faces(Image.open(im))
                ##for fxs in face_new:
                #    st.image(fxs)
                faces_all.append(faces)
                if faces is not None:
                    print(faces.shape)
                    for x in faces:
                        face_arr.append(x)

            col1,col2 = st.columns(2)
            col1.metric("Images Uploaded", len(uploaded_imgs))
            col2.metric("Faces Detected",len(face_arr))
            #st.metric("Images Uploaded", len(uploaded_imgs))
            #plt.figure(figsize=(12, 4))
            #plt.title('Detected faces per frame')
            #plt.plot([len(f) for f in faces])
            #plt.show()
            i = 0
            for fc in face_arr:
                print(fc.shape)
                #plt.imshow(fc.permute(1,2,0))
                #st.image(fc.permute(1, 2, 0))
                i_c = fc[0]
                print("printing shape here",i_c.shape)
                filename = './faces/face'+str(i)+'.jpg'
                save_image(i_c, filename)
                read_im = cv2.imread(filename)
                read_im = cv2.cvtColor(read_im, cv2.COLOR_BGR2RGB)
                #st.image(read_im)
                pil_image = transforms.ToPILImage()(fc.squeeze_(0))
                im = transforms.ToPILImage()(fc).convert("RGB")
                print(pil_image.size,im.size)
                col1,col2 = st.columns(2)
                with col1:
                    st.image(pil_image)
                with col2:
                    st.image(read_im)
                i += 1
            

            
            

    if add_selectbox == "Face Detection":
            #uploaded_file = st.file_uploader("Choose File", type=["jpg","png"])
        uploaded_imgs = images
        st.metric("Images Uploaded", len(uploaded_imgs))
        if uploaded_imgs is not None:
                # Create face detector
            mtcnn = MTCNN(margin=20, keep_all=True, post_process=False) #device='cuda:0'
            # Detect faces in batch
            faces = mtcnn(uploaded_imgs)
            fig, axes = plt.subplots(len(faces), 2, figsize=(6, 15))
            for i, frame_faces in enumerate(faces):
                for j, face in enumerate(frame_faces):
                    axes[i, j].imshow(face.permute(1, 2, 0).int().numpy())
                    axes[i, j].axis('off')
            fig.show()

    if add_selectbox == "Verify Face":
        st.subheader("Load Images Here")
        image_files = st.file_uploader("Upload Images", type=['jpg', 'png', 'jpeg'])
        st.subheader("Baseline Here")
        face = st.file_uploader("Upload Face Check", type=['jpg', 'png', 'jpeg'])
        column1, column2 = st.columns(2)
        
        with column1:
            image1 = st.file_uploader("Choose File", type=["jpg","png"])
            
        with column2:
            image2 = st.file_uploader("Select File", type=["jpg","png"])
        if (image1 is not None) & (image2  is not None):
            col1, col2 = st.beta_columns(2)
            image1 =  Image.open(image1)
            image2 =  Image.open(image2)
            with col1:
                st.image(image1)
            with col2:
                st.image(image2)

            filenames = [image1,image2]

            faces = [extract_face(f) for f in filenames]
            samples = asarray(faces, "float32")
            samples = preprocess_input(samples, version=2)
            model = VGGFace(model= "resnet50" , include_top=False, input_shape=(224, 224, 3),
                pooling= "avg" )
                # perform prediction
            embeddings = model.predict(samples)
            thresh = 0.4
            score = cosine(embeddings[0], embeddings[1])
            if score <= thresh:
                st.success( " >face is a match (%.3f <= %.3f) " % (score, thresh))
            else:
                st.error(" >face is NOT a match (%.3f > %.3f)" % (score, thresh))

            """
            print(faces_all)
            fig, axes = plt.subplots(len(faces_all), 3, figsize=(6, 15))
            for i, frame_faces in enumerate(faces_all):
                for j, face in enumerate(frame_faces):
                    print(face.shape)
                    axes[i, j].imshow(face.permute(1, 2, 0))
                    axes[i, j].axis('off')
            fig.show()
            """ 
            """ 
            detector = MTCNN()
            faces = detector.detect_faces(data)
            for face in faces:
                x, y, width, height = face['box']
                rect = Rectangle((x, y), width, height, fill=False, color='maroon')
                ax.add_patch(rect)
                for _, value in face['keypoints'].items():
                    dot = Circle(value, radius=2, color='maroon')
                    ax.add_patch(dot) 
            st.pyplot(fig) """


if __name__ == '__main__':
		main()
