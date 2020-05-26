
import os
import io
import pickle
from google.cloud import vision


class CovidNumbers():
    region_type = 'district'
    region_name = None
    stats = {
        'total': {
            'confirmed': None,
            'recovered': None,
            'deceased': None,
            'migrated': None,
            'active': None
            },
        'daily': {
            'confirmed': None,
            'recovered': None,
            'deceased': None,
            'migrated': None,
            'active': None
            }
    }


# class Bulletin():
#     STATE_NAME = 'UP'
#     NUM_DISTRICTS = 75

def detect_text(path):
    """Detects text in the file."""
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations

    # Save output
    bas_path = './data/output/'
    out_path = f'{bas_path}{os.path.splitext(os.path.basename(path))[0]}_response.pickle'
    if not (os.path.isdir(bas_path)):
        os.mkdir(bas_path)
    with open(out_path, 'wb') as jf:
        pickle.dump(response, jf)

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
    path = './data/MP1.jpeg'

    # Do OCR
    detect_text(path)

    # TODO
    # Plot text with bounding box and display
    # Extract line by line
    # Convert Hindi District names to English using fuzzy lookup
    # Tabulapy?
    

if __name__ == "__main__":
    main()
