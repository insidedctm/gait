from GaitRecogniser import GaitRecogniser
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input_path')
args = parser.parse_args()


recogniser = GaitRecogniser()
recogniser.train()

prediction = recogniser.predict_from_file(args.input_path)
print(prediction)