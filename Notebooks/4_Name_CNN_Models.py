import os 
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import pickle


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score, balanced_accuracy_score, classification_report


import tensorflow as tf
print("Tensorflow:",tf.__version__)
print("GPU list:",tf.config.list_physical_devices('GPU'))
from tensorflow.keras import datasets, layers, models
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from keras.optimizers import Adam

from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K


import keras
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input




def confusion_matrix_plot(y_true, y_pred, file_save="",labels=[], size=8, fz=15):
    
    cf_matrix = confusion_matrix(y_true, y_pred, normalize=("true"))
    cf_matrix_v = confusion_matrix(y_true, y_pred)
    
    if (len(labels)==0):
        labels = np.unique(y_true)
        print(labels)
    
    # Define a color map for the heatmap
    cmap = sns.color_palette("Blues", as_cmap=True)
    
    plt.figure(figsize = (size,size))
    heatmap = sns.heatmap(cf_matrix, annot=False, cmap=cmap, fmt='d', xticklabels=labels, yticklabels=labels, square=True, cbar=True)
    # heatmap = sns.heatmap(data, annot=False, fmt='d', cmap=cmap)
    
    # Calculate and display percentages
    for i in range(len(cf_matrix)):
        for j in range(len(cf_matrix[i])):
            text = f"{cf_matrix[i][j]*100: 0.2f}% ({cf_matrix_v[i][j]})"
            
            # Calculate the background color at the center of the cell
            cell_bg_color = np.mean(cmap(cf_matrix[i][j]))
            print(i,j, cell_bg_color)
            # Choose a text color based on the background color
            text_color = 'black' if cell_bg_color > 0.7 else 'white'
                    
            heatmap.text(j + 0.5, i + 0.5, text, ha='center', va='center', color=text_color)
    
    # ax.set_xticklabels(ax.get_xticklabels(),rotation = 40)
    # ax.set_yticklabels(ax.get_xticklabels(),rotation = 40)
    

    plt.title("Confusion Matrix",fontsize=fz)
    plt.xlabel("Predicted",fontsize=fz-2)
    plt.ylabel("Real",fontsize=fz-2)
    
    plt.tight_layout()
    
    if(file_save!=""):
        plt.savefig(file_save)
    
    plt.show(block=False)
    
    return cf_matrix


def load_images_vgg16(db_path):

    ## List images from dataset ##
    image_file_list = os.listdir(db_path)

    ## Lists to save information ##
    img_list = []
    label_list = []

    for file in image_file_list:

        #### Load-pre-process-save images ####
        ## Create image file ##
        image_file = os.path.join(db_path, file)
        ## Load image ##
        img = cv2.imread(image_file)
        ## Tranform to gray sacale ##
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC) 
        ## Reshape the image to (W, H, C) ##
        # H, W = img.shape
        # img = img.reshape(W, H, 1)
        ## Normalize pixels to [0,1] ##
        # img = np.float64(img)/255.0
        ## Invert image pixels ##
        img = (255.0 - img)

        ## save image in list ##
        img_list.append(img)
        #######################################

        #### Assign labels ####
        if("PD_" in file):
            label_list.append(1)
        else:
            label_list.append(0)
        #######################

    ## Conver images to array object ##
    img_arr = np.asarray(img_list)
    img_arr = preprocess_input(img_arr)

    ## Conver label to array object ##
    label_arr = np.asarray(label_list)

    print(img_arr.shape, type(img_arr))
    print(label_arr.shape, type(label_arr))



    return img_arr, label_arr


def split_data_Train_Test(X, y):

    #### Codify labels to onehot ####
    num_classes = len(np.unique(y))
    

    #### Split Data ####
    ## Train and test ##
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
    
    ## Convert labels to categorical ##
    y_train_c = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test_c = tf.keras.utils.to_categorical(y_test, num_classes)


    values, counts = np.unique(y_train, return_counts=True)
    print(f"Train data {X_train.shape}, {y_train_c.shape}, {counts}")

    values, counts = np.unique(y_test, return_counts=True)
    print(f"Test data {X_test.shape}, {y_test_c.shape}, {counts}")

    idx = np.random.randint(0, len(y_train_c))
    print_image(X_train[idx], title=f"lab:{y_train_c[idx]}")

    return X_train, y_train_c, X_test, y_test_c


def print_image(img, title="Image"):
    plt.figure(figsize=(3,3))
    plt.imshow(img, cmap="gray")
    plt.title(title)
    # plt.axis("off")
    plt.xticks([])
    plt.yticks([])
    plt.show(block=False)


def create_dir(path):

    dirs = path.split("/")
    n_path = ""
    for i in range(len(dirs)):
        
        n_path = os.path.join(n_path, dirs[i])
        # print(n_path)
        a = os.path.isdir(n_path)   
        # print(a) 
        if(not(a)):
            os.mkdir(n_path) 


# plot accuracy and loss
def plotgraph(epochs, train, val, ax, plot, title):
    # Plot training & validation accuracy values
    ax[plot].plot(epochs, train, 'b')
    ax[plot].plot(epochs, val, 'r')
    ax[plot].set_title(title)
    #ax[plot].ylabel(title)
    ax[plot].set(xlabel='Epoch', ylabel=title)
    #ax[plot].xlabel('Epoch')
    ax[plot].legend(['Train', 'Val'])






#### Load data ####
b_list = ["dense", "block5", "block4", "block3", "block2", "block1"]

pp = ""
task = "name"
result_path = os.path.join(pp, "Results")
create_dir(result_path)
result_file = os.path.join(result_path, f"{task}_fine_tunning.xlsx")

## Save info ##
block_list = []
acc_list = []
b_acc_list = []
sens_list = []
spec_list = []
f1_s_list = []
trainable_count_list = []
non_trainable_count_list = []
total_parameters_list = []

for layer_name_fine in b_list:

    # task = "spiral"
    dp = 0.25
    lr = 1e-6
    wd = 1e-4

    # epch = 2000
    # pt = 100
    # md = 0.00001
    # bz = 6

    epch = 2
    pt = 1
    md = 0.00001
    bz = 6

    model_name = f"Prueba_VGG16_{layer_name_fine}"
    
    

    db_path = os.path.join(pp,f"hw_drawings", task)
    X, y = load_images_vgg16(db_path)
    # X_train, y_train_c, X_val, y_val_c, X_test, y_test_c = split_data(X, y)
    X_train, y_train_c, X_test, y_test_c = split_data_Train_Test(X, y)


    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(256,256,3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu', name="dense")(x)
    predictions = Dense(2, activation='softmax', name="prediction")(x)
    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    # first: train only the top layers (which were randomly initialized)
    
    trainable_flag = False
    #### Set up  VGG16 layers as trainable or not ####
    for layer in model.layers:
        if(layer_name_fine in layer.name):
            trainable_flag = True
    
        print(layer.name, trainable_flag)
        layer.trainable = trainable_flag

    for i, layer in enumerate(model.layers):
        print(i, layer.name)


    # img_path = 'elephant.jpg'
    # img = keras.utils.load_img(img_path, target_size=(224, 224))
    # x = keras.utils.img_to_array(img)
    # x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)

    # features = model.predict(x)

    model.summary()


    trainable_count = np.sum([np.prod(v.shape) for v in model.trainable_weights])
    non_trainable_count = np.sum([np.prod(v.shape) for v in model.non_trainable_weights])


    print("Model Training: ", model_name, "\n\n")

    model_path = os.path.join(pp,"Models",task,model_name)
    create_dir(model_path)

    ## Define model file ##
    model_file = os.path.join(model_path, f"{model_name}_model.keras")
    ## Define log file
    log_dir = os.path.join(model_path, f"fit/{model_name}_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr, weight_decay=wd), metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=md, patience=pt, verbose=1, mode='min')
    mcp_save = ModelCheckpoint(model_file, save_best_only=True, monitor='val_loss', verbose=1, mode='min')



    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(X_train,
                        y_train_c,
                        epochs=epch,
                        batch_size=bz,
                        verbose=1,
                        validation_data=(X_test, y_test_c),
                        callbacks=[early_stopping, mcp_save, tensorboard_callback])








    history_file = os.path.join(model_path, f"history_{model_name}.picl")
    with open(history_file, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    file_pi.close()


    with open(history_file, 'rb') as file_pi:
        history = pickle.load(file_pi)
    file_pi.close()

    pd.DataFrame.from_dict(history)




    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(1,len(acc)+1)


    #### crear subplot ####
    fig, ax = plt.subplots(1, 2, figsize=(10,4))
    #### graficar curva acc ###
    plotgraph(epochs, acc, val_acc, ax, 0, "accuracy")
    #### graficar curva lost ###
    plotgraph(epochs, loss, val_loss, ax, 1, "val_lost")

    fig_file = os.path.join(model_path, f"history_{model_name}.pdf")
    plt.savefig(fig_file)
    plt.show(block=False)









    model_file = os.path.join(model_path, f"{model_name}_model.keras")
    print(model_file)

    model = models.load_model(model_file)
    model.summary()


    model.evaluate(X_train, y_train_c)

    # model.evaluate(X_val, y_val_c)

    model.evaluate(X_test, y_test_c)
    y_pred_c = model.predict(X_test)
    y_pred = y_pred_c.argmax(axis=1)
    y_true = y_test_c.argmax(axis=1)
    print(y_true.shape, y_pred.shape)
    

    ##  Evaluate metrics ##
    acc = accuracy_score(y_true, y_pred)*100
    b_acc = balanced_accuracy_score(y_true, y_pred)*100
    sens = recall_score(y_true, y_pred)*100
    ## Specificity ##
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn/(tn+fp)*100
    f1_s = f1_score(y_true, y_pred)


    #### Save info ####
    block_list.append(layer_name_fine)
    acc_list.append(acc)
    b_acc_list.append(b_acc)
    sens_list.append(sens)
    spec_list.append(spec)
    f1_s_list.append(f1_s)
    trainable_count_list.append(trainable_count)
    non_trainable_count_list.append(non_trainable_count)
    total_parameters_list.append(trainable_count+non_trainable_count)


    #### Mostrar la microclasificación ####
    label_dictionary = {
        0: "HC",
        1: "PD"

    }
    report = classification_report(y_true, y_pred, digits=4, output_dict=False, target_names=list(label_dictionary.values()))
    table_report = classification_report(y_true, y_pred, digits=4, output_dict=True, target_names=list(label_dictionary.values()))
    table_report = pd.DataFrame.from_dict(table_report).T
    print(report)


    labels = list(label_dictionary.values())
    cm_file_save = os.path.join(model_path, f"CM_{model_name}.pdf")
    confusion_matrix_plot(y_true, y_pred, cm_file_save,labels=labels, size=4, fz=15)

result_df = pd.DataFrame([])
result_df["Fine_Tune_Block"] = block_list
result_df["total_parameters"] = total_parameters_list
result_df["trainable_count"] = trainable_count_list
result_df["non_trainable_count"] = non_trainable_count_list
result_df["Accuracy"] = acc_list
result_df["Balanced_Accuracy"] = b_acc_list
result_df["Specificity"] = spec_list
result_df["Sensitivity"] = sens_list
result_df["F1_Score"] = f1_s_list


print(result_file)
# result_df.sort_values(by="Fine_Tune_Block", ascending=False, inplace=True)
# result_df.reset_index(drop=True, inplace=True)
result_df.to_excel(result_file, index=False)
result_df

