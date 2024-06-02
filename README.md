# Dice_Image_Classification
CNN Model to classify dice image based on dice number

docker build -t nazwa .
docker run -it nazwa
docker run -v $(pwd)/output:/app/output my_image_name

docker run -p 7860:7860  -it gradio_interface