from argparse import ArgumentParser


class TrainOptions:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
		self.parser.add_argument('--config', type=str, default='001', help='Path to the config file.')
		self.parser.add_argument('--inversion_model_path', type=str, default='./pretrained_models/e4e_ffhq_encode.pt', help='pretrained inversion model')

	def parse(self):
		opts = self.parser.parse_args()
		return opts