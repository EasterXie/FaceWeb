ign
        self.classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.num_faces = 0

    def transform(self, fram