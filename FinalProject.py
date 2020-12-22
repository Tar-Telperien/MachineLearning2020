import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import os
import random
import time
from multiprocessing import Process, Queue, current_process
import matplotlib.pyplot as plt

def read_labels_and_images():
    start_time = time.time()
    filepath = "manyBoards2/"
    # Read the label file, mapping image filenames to (x, y) pairs for UL corner
    with open(filepath + "labels_dict.txt") as label_file:
        label_string = label_file.readline()
        label_map = eval(label_string)
    print("Read label map with", len(label_map), "entries")

    # Read the directory and make a list of files
    filenames = []
    for filename in os.listdir(filepath):
        #if random.random() < -0.95: continue # to select just some files for full training
        if random.random() < 0.95: continue # to use one image in twenty
        if filename.find("png") == -1: continue # process only images
        filenames.append(filename)
    print("Read", len(filenames), "images.")

    # Extract the features from the images
    print("Extracting features")

    t, tl, p, pl = [], [], [], []
    checkpoint, check_interval, num_done = 0, 5, 0 # Just for showing progress
    for filename in filenames:
        if 100*num_done > (checkpoint + check_interval) * len(filenames):
            checkpoint += check_interval
            print((int)(100 * num_done / len(filenames)), "% done")
        num_done += 1
        img = cv2.imread(filepath + filename, cv2.IMREAD_GRAYSCALE)/255.0
        img = cv2.resize(img, (224, 224))
        img = tf.reshape(img, [224, 224, 1])
        for contrast in [1, 1.2, 1.5, 1.7]: #may need to remove this if colour images cause a problem; see tf_fingers_hub.py for comment
            img2 = tf.image.adjust_contrast(img, contrast_factor=contrast)
            #print("Adjusted contrast")
            for bright in [0.0, 0.1, 0.2, 0.3]:
                img3 = tf.image.adjust_brightness(img2, delta=bright)
                #print("Adjusted brightness")
                if np.random.random() < 0.8: # 80% of images
                    t.append(img.numpy())
                    tl.append(label_map[filename])
                else:
                    p.append(img.numpy())
                    pl.append(label_map[filename])
                print("Appended one image")
    return (t, tl, p, pl)

def build_finger_model():
    # Build the model
    model = tf.keras.Sequential([
        hub.KerasLayer("https://tfhub.dev/google/imagenet/resnet_v1_50/feature_vector/4",
                       trainable=False),
        tf.keras.layers.Dense(100),
        tf.keras.layers.Dense(2, activation='linear')
    ])
    model.build([None, 224, 224, 3])  # Batch input shape.
    
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='mse',
                  metrics=['accuracy'])
    print("Done building the network topology.")

    # Read training and prediction data
    train, train_labels, predict, predict_labels = read_labels_and_images()
    #checkpoint = tf.keras.callbacks.ModelCheckpoint("models/fingersteps/hubcp_{epoch}")

    # Train the network
    print("Starting to train the network, on", len(train), "samples.")
    model.fit(np.asarray(train, dtype=np.float32), np.asarray(train_labels), \
              epochs=3, batch_size=8, verbose=1)#, callbacks=[checkpoint])
    model.save("sudoku_centre")
    print("Done training network.")

    # Predict
    p = model.predict_classes(np.asarray(predict))
    num_correct, num_total = 0, 0
    results = [[0 for i in range(3)] for j in range(3)]
    for pr, pl in zip(p, predict_labels):
        #print("Predict", pr, "\tActual", pl, "***" if (pr != pl) else ".")
        if pr == pl: num_correct += 1
        num_total += 1
        results[pl][pr] += 1
    print("Accuracy on prediction data", (100 * num_correct)/num_total)
    print(np.array(results))
        
    return model


# Reads model saved in filename on disk, runs it using webcam input
def load_and_run_model(filename):
    model = tf.keras.models.load_model(filename)
    cap = cv2.VideoCapture(0)
    keep_going = True
    while keep_going:
        ret, frame = cap.read()
        h, w = frame.shape[:2]
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        crop = frame[:,(w-h)//2:(w+h//2)]
        crop = cv2.resize(crop, (224, 224))
        tf_img = tf.reshape(crop, [224, 224, 3])

        p = model.predict_classes(np.asarray([tf_img], dtype=np.float32), batch_size=1)[0]
        #p = model.predict(np.asarray([tf_img], dtype=np.float32))[0] # See the raw data
        
        #color = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR),(256, 256))
        color = cv2.resize(crop, (256, 256))
        cv2.putText(color, str(p+1), (20, 30) , cv2.FONT_HERSHEY_SIMPLEX,\
                        1.0,(255,0, 0),2, lineType=cv2.LINE_AA)
        cv2.imshow("Number of Fingers", color)
        if cv2.waitKey(3000) & 0xFF == ord(' '):
            keep_going = False

    cap.release()
    cv2.destroyAllWindows()


# Loads a model saved in filename from disk,
# Visualizes the features in the first hidden layer
# Remember that in matplotlib, black is large, white is small (including negative)
def load_and_visualize_model(filename):
    np.set_printoptions(precision=3, suppress=True)
    model = tf.keras.models.load_model(filename)
    lv = None
    for v in model.variables:
        if len(v.numpy().shape) < 2: continue
        if v.name.find("block1") == -1: continue
        if v.name == "resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/weights:0":
            lv = v
            break
        print(v.name)
        print(v.numpy().shape)

    print(lv)
    v = tf.squeeze(lv).numpy()
    print(v.shape)
    num_features = v.shape[2] * v.shape[3]
    fig, ax = plt.subplots(nrows=8, ncols=8)#num_features//4)
    ax = ax.flatten()
    for i in range(num_features):
        if i >= 64: continue
        img = v[:,:,i//64, i%64]
        ax[i].imshow(img, cmap='Greys')
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        print(img)
    plt.tight_layout()
    plt.show()
    plt.savefig(filename + ".png")

    
# Run this function to capture pictures and save them to files on disk.
# Also save the labels dictionary. 
def capture_and_save_images():
    #Video capture from webcam
    num_each_digit = 1000
    labels = {}
    count = 0
    filepath = "fingers/"
    for label in [1, 2, 3]:
        cap = cv2.VideoCapture(1)
        for i in range(num_each_digit):
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            crop = gray[:,(w-h)//2:(w+h//2)]
            crop = cv2.resize(crop, (64, 64))
            name = "finger_" + str(label) + "_" + str(i) + ".png"
            filename = filepath + name            
            cv2.imwrite(filename, crop)
            labels[name] = label
            
            cv2.imshow("Visualizing the Cropped Image", cv2.pyrUp(cv2.pyrUp(crop)))
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            
            count += 1
            # Save every 100 images
            if count % 100 == 0:
                with open(filepath + "labels", "w") as label_output_file:
                    label_output_file.write(str(labels))

        # Save after each label
        with open(filepath + "labels", "w") as label_output_file:
            label_output_file.write(str(labels))
        cap.release()
        cv2.waitKey()

    cap.release()
    cv2.destroyAllWindows()


#capture_and_save_images()
build_finger_model()
#load_and_run_model("models/fingers_hub_1")
#load_and_visualize_model("models/fingers_hub_1")

#HERE IS WHERE YOU DO FUNCTION_Y STUFF: PUT IN WHAT YOU WANT RUN
