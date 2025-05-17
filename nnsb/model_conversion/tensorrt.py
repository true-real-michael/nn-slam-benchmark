import tqdm
from nnsb.method import Method
from torch.utils import data
from nnsb.dataset.dataset import Data
import torch


CALIBRATION_DATASET_N_BATCHES = 16


class TensorRTExportable(Method):
    def quantize_to_onnx(self, calibration_data_dir, calibration_data_name, onnx_path, resize):
        """
        Quantize model and export to ONNX.
        
        Args:
            calibration_data_dir: Dataset directory for calibration
            calibration_data_name: Name of the dataset for calibration
            onnx_path: Path to save the ONNX model
            resize: Image resize value
        
        Returns:
            bool: True if quantization was successful, False otherwise
        """
        try:
            from pytorch_quantization import nn as quant_nn
            from pytorch_quantization import quant_modules
            from pytorch_quantization import calib
            
            quant_modules.initialize()
            
            train_dataset = Data(calibration_data_dir, calibration_data_name, resize=resize)
            train_dataloader = data.DataLoader(
                train_dataset, batch_size=64, shuffle=True, drop_last=True
            )
        
            model = self.get_torch_backend().get_torch_module().cpu()
            
            for param in model.parameters():
                param.requires_grad = False
                
            model = model.cuda()
    
            with torch.no_grad():
                for name, module in model.named_modules():
                    if isinstance(module, quant_nn.TensorQuantizer):
                        if module._calibrator is not None:
                            module.disable_quant()
                            module.enable_calib()
                        else:
                            module.disable()
    
                for i, (image, _) in tqdm.tqdm(zip(range(CALIBRATION_DATASET_N_BATCHES), (train_dataloader)), total=CALIBRATION_DATASET_N_BATCHES):
                    model(image.cuda())
    
                for name, module in model.named_modules():
                    if isinstance(module, quant_nn.TensorQuantizer):
                        if module._calibrator is not None:
                            module.enable_quant()
                            module.disable_calib()
                        else:
                            module.enable()
                
                for name, module in model.named_modules():
                    if isinstance(module, quant_nn.TensorQuantizer):
                        if module._calibrator is not None:
                            if isinstance(module._calibrator, calib.MaxCalibrator):
                                module.load_calib_amax()
                            else:
                                module.load_calib_amax(method="max")
    
            quant_nn.TensorQuantizer.use_fb_fake_quant = True
    
            dummy_input = self.get_sample_input().cuda()
    
            input_names = ["actual_input_1"]
            output_names = ["output1"]
    
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                verbose=False,
                opset_version=13,
                do_constant_folding=False,
                input_names=input_names,
                output_names=output_names
            )
            
            del model
            torch.cuda.empty_cache()
            
            return True
            
        except Exception as e:
            print(f"Error during ONNX quantization: {e}")
            import traceback
            traceback.print_exc()
            return False
