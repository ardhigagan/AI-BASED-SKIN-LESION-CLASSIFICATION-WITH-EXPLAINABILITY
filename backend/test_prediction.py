
from app.ml.prediction_service import predict_image

image_path = r"C:\Users\KIIT0001\Desktop\skin-lesion-classifier\data\processed\Testing_data\199.jpg"

result = predict_image(image_path)

print("\nPrediction Result:")
print(result)
