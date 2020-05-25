
import os
from google.cloud import vision
import io
import json


def detect_text(path):
    """Detects text in the file."""
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations

    # Save output
    bas_path = './data/output'
    out_path = f'{bas_path}{os.path.basename(path)}.json'
    if not (os.path.isdir(bas_path)):
        os.mkdir(bas_path)
    with open(out_path, 'w') as jf:
        json.dump(texts, jf)

    print('Texts:')

    for text in texts:
        print('\n"{}"'.format(text.description))

        vertices = (['({},{})'.format(vertex.x, vertex.y)
                    for vertex in text.bounding_poly.vertices])

        print('bounds: {}'.format(','.join(vertices)))

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))


def main():
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "./credentials/visionapi.json"
    file_path = './data/MP1.jpeg'
    detect_text(file_path)


if __name__ == "__main__":
    main()
