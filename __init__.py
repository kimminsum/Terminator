import tensorflow as tf
from PIL import Image
import numpy as np
import csv
import os


class Terminator:
    def __init__(self, epoch, img_size, img_number, label, ShowModelSummary, train_trigger, test_trigger):
        self.PATH = os.getcwd()
        ### Relate Accuracy ###
        self.train_times: int = epoch # epoch
        self.img_size: int = img_size # image size -> this parameter maybe affluence to accuracy
        self.img_number: int = img_number # train image numbers
        ### Selective Option ###
        self.label: dict = label # object's name
        self.ShowModelSummary: bool = ShowModelSummary # choose to show summary
        self.train_trigger: bool = train_trigger
        self.test_trigger: bool = test_trigger
        ### .ETC ###
        self.resized_img_list: list = []
        self.object_name_list: list = []
        self.main()
    """
    Train Tensorflows'Model
    """
    def train_model(self, target_list: list, name_list: list, trigger):
        if trigger:
            self.model = tf.keras.Sequential([
                                            tf.keras.layers.InputLayer(input_shape=[self.img_size**2,]), # input layer : [img_size, img_size]
                                            tf.keras.layers.Dense(128, activation="relu"), # hidden layer : 128 (relu)
                                            tf.keras.layers.Dense(10, activation="softmax") # output layer : 10 (softmax)
                                            ])
            self.model.compile  (
                                optimizer = "adam",
                                loss = "sparse_categorical_crossentropy",
                                metrics = ["accuracy"]
                                )
            if self.ShowModelSummary: self.model.summary() # show model's summary
            self.model.fit  (
                                target_list, name_list, # Input Layer, Label List
                                epochs = self.train_times,
                                batch_size = 10,
                                validation_split = 0.25
                                )
    """
    Convert image to .csv || label, image list && Load image data from .csv
    """
    def convert_img_to_csv(self, trigger):
        if trigger:
            img_list : list = []
            resized_img_list : list = []
            # convert img to .csv by type
            for i in range(self.img_number):
                try:
                    img_list.append(Image.open(f"img/{i}.png").convert("L"))
                except:
                    img_list.append(Image.open(f"img/{i}.jpg").convert("L"))                
                else:
                    img_list.append(Image.open(f"img/{i}.jpeg").convert("L"))

            for i in range(self.img_number):
                resized_img_list.append(np.array(img_list[i].resize((self.img_size, self.img_size))).reshape(1, -1)[0]/255.)

            # save to .csv
            with open(f"{self.PATH}/database/image.csv", "w", newline="") as f:
                file = csv.writer(f)
                file.writerow(["label", "image Data"])
                for i in range(self.img_number):
                    file.writerow(resized_img_list[i])
            f.close()


    def read_img_from_csv(self, trigger):
        if trigger:
            rows : list = []
            with open(f"{self.PATH}/database/image.csv", "r") as f:
                file = csv.reader(f)
                next(file)
                for row in file:
                    rows.append(row)

            for i in range(len(rows)):
                self.object_name_list.append(rows[i][0])
                self.resized_img_list.append(list(map(float, rows[i][:])))
            f.close()
            print(len(self.resized_img_list), "read")
            print(self.resized_img_list[0])
            # print(self.object_name_list, self.resized_img_list)
    """
    Model Save && Load
    """
    def save_model(self, model, trigger):
        if trigger:
            model.save(f"{self.PATH}/model/image.h5")


    def load_model(self, trigger):
        if trigger:
            self.model = tf.keras.models.load_model(f"{self.PATH}/model/image.h5")

    """
    Test the model    
    """
    def test_model(self, test_image, trigger):
        if trigger:
            img = np.array(test_image.resize((self.img_size, self.img_size))).reshape(1, -1)[0]/255.
            print(self.label[self.model.predict(np.array([img])).argmax()])

    """
    Terminal
    """
    def main(self):
        self.convert_img_to_csv(self.train_trigger)
        self.read_img_from_csv(self.train_trigger)

        X = np.array(self.resized_img_list)

        y_list: list = []
        for i in range(20):
            y_list.append(0)
        for j in range(30):
            y_list.append(1)
        Y = np.array(y_list)

        self.train_model(X, Y, self.train_trigger)
        try: 
            self.save_model(self.model, self.test_trigger)
            self.load_model(self.test_trigger)
        except:
            self.load_model(self.test_trigger)
            self.save_model(self.model, self.test_trigger)
        self.test_model(Image.open("img/2.jpg").convert("L"), self.test_trigger) # test the model


if __name__== "__main__":
    Terminator(500, # Epoch
               50, # Image Size
               50, # Image Number
               {0: "가위", 1: "보"}, # Label
               True, # ShowModelSummary
               False, # Train Trigger
               True) # Test Trigger