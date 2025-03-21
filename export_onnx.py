from pathlib import Path
from nnsb.vpr_systems.eigenplaces import EigenPlaces
from onnxruntime.quantization import quantize_static, QuantFormat, QuantType
from onnxruntime.quantization import CalibrationDataReader
import numpy
import os
from PIL import Image


model = EigenPlaces()

# model.export_onnx(Path("weights/tmp/onnx/eigenplaces.onnx"))

images_path = Path("datasets/st_lucia/images/test/database")


def _preprocess_images(images_folder: str, height: int, width: int, size_limit=0):
    """
    Loads a batch of images and preprocess them
    parameter images_folder: path to folder storing images
    parameter height: image height in pixels
    parameter width: image width in pixels
    parameter size_limit: number of images to load. Default is 0 which means all images are picked.
    return: list of matrices characterizing multiple images
    """
    image_names = os.listdir(images_folder)
    if size_limit > 0 and len(image_names) >= size_limit:
        batch_filenames = [image_names[i] for i in range(size_limit)]
    else:
        batch_filenames = image_names
    unconcatenated_batch_data = []

    for image_name in batch_filenames:
        image_filepath = images_folder + "/" + image_name
        pillow_img = Image.new("RGB", (width, height))
        pillow_img.paste(Image.open(image_filepath).resize((width, height)))
        input_data = numpy.float32(pillow_img) - numpy.array(
            [123.68, 116.78, 103.94], dtype=numpy.float32
        )
        nhwc_data = numpy.expand_dims(input_data, axis=0)
        nchw_data = nhwc_data.transpose(0, 3, 1, 2)  # ONNX Runtime standard
        unconcatenated_batch_data.append(nchw_data)
    batch_data = numpy.concatenate(
        numpy.expand_dims(unconcatenated_batch_data, axis=0), axis=0
    )
    return batch_data


class Reader(CalibrationDataReader):
    def __init__(self, dataset_path, method):
        self.enum_data = None
        resize = method.resize
        self.nhwc_data = _preprocess_images(
            str(dataset_path), resize, resize, size_limit=100
        )
        self.input_name = "input.1"
        self.datasize = len(self.nhwc_data)
    
    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                [{self.input_name: nhwc_data} for nhwc_data in self.nhwc_data]
            )
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None
        

reader = Reader(images_path, model)


quantize_static(
    Path("weights/tmp/onnx/eigenplaces-infer.onnx"),
    Path("weights/tmp/onnx/eigenplaces_quant.onnx"),
    reader,
    weight_type=QuantType.QInt16,
    # quant_format=QuantFormat.QDQ,
)
