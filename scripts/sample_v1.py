import os
from absl import flags
from absl import app
from ml_collections import config_flags

from ..sample_multi_v1 import evaluate

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", "configs/sample_unidiffuser_v1.py", "Configuration.", lock_config=False)
flags.DEFINE_string("nnet_path", "models/uvit_v1.pth", "The nnet to evaluate.")
flags.DEFINE_string("output_path", "out", "dir to write results to")
flags.DEFINE_string("prompt", "an elephant under the sea", "the prompt for text-to-image generation and text variation")
flags.DEFINE_string("img", "assets/space.jpg", "the image path for image-to-text generation and image variation")
flags.DEFINE_integer("n_samples", 1, "the number of samples to generate")
flags.DEFINE_integer("nrow", 4, "number of images displayed in each row of the grid")
flags.DEFINE_string("mode", None,
                    "type of generation, one of t2i / i2t / joint / i / t / i2t2i/ t2i2t\n"
                    "t2i: text to image\n"
                    "i2t: image to text\n"
                    "joint: joint generation of text and image\n"
                    "i: only generate image\n"
                    "t: only generate text\n"
                    "i2t2i: image variation, first image to text, then text to image\n"
                    "t2i2t: text variation, first text to image, the image to text\n"
                    )


def sample(argv):
    config = FLAGS.config
    config.nnet_path = FLAGS.nnet_path
    config.output_path = FLAGS.output_path
    config.prompt = FLAGS.prompt
    config.nrow = min(FLAGS.nrow, FLAGS.n_samples)
    config.img = FLAGS.img
    config.n_samples = FLAGS.n_samples
    config.mode = FLAGS.mode
    evaluate(config)


def generate():
    app.run(sample)


if __name__ == '__main__':
    generate()