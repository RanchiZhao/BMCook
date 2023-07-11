import types
import model_center
import math
import cpm_kernels.torch as ct
import torch
import bitsandbytes as bnb

class BMQuant:
    '''
    BMQuant enables quantization-aware training of PLMs by using `cpm-kernels`.
    '''

    @classmethod
    def quantize(cls, model, config, target_linear = None):
        '''
        Practitioners can turn on quantization by `is_quant` in the config, which will replace all linear layers with quantized linear layers. BMCook provides the simulation of 8-bit quantization.

        :param model: Model to quantize.
        :param config: Configuration of the quantization.
        '''
        quant_config = config.get('quantization')
        if not quant_config['is_quant']:
            return

        if quant_config['is_quant'] == 'int8':
            # fix cpm_kernel
            ct.gemm.GEMMInt8._backward = ct.gemm.GEMMInt8.backward
            def new_func(ctx, grad_f):
                if not grad_f.is_contiguous():
                    grad_f = grad_f.contiguous()
                return ct.gemm.GEMMInt8._backward(ctx, grad_f)
            ct.gemm.GEMMInt8.backward = new_func

            target_linear = model_center.layer.Linear if target_linear is None else target_linear

            for name, module in model.named_modules():
                if isinstance(module, target_linear):
                    if len(quant_config["quantized_module"]) != 0:
                        if not any([pattern in name for pattern in quant_config["quantized_module"]]):
                            continue
                    if target_linear != model_center.layer.Linear:
                        module.forward = types.MethodType(forward_int8_cpmlive, module)
                    else:
                        module.forward = types.MethodType(forward_in8, module)
                    module.quant = True
        #add here            
        elif quant_config['is_quant'] == 'int4':
            target_linear = model_center.layer.Linear if target_linear is None else target_linear
            for name, module in model.named_modules():
                if isinstance(module, target_linear):
                    if len(quant_config["quantized_module"]) != 0:
                        if not any([pattern in name for pattern in quant_config["quantized_module"]]):
                            continue
                    if target_linear != model_center.layer.Linear:
                        module.forward = types.MethodType(forward_int4_cpmlive, module)
                    else:
                        raise FutureWarning("not available, future work") 
                    module.quant = True


def forward_in8(module_self, x):
    if module_self.length_scale and module_self.length_scale_before:
        x = x / math.sqrt(module_self.dim_in)
    x = x.transpose(1, 2).contiguous()
    x = ct.bmm(module_self.weight.unsqueeze(0), False, x, False, int8=True)
    x = x.transpose(1, 2).contiguous()
    if module_self.length_scale and not module_self.length_scale_before:
        x = x / math.sqrt(module_self.dim_in)
    if module_self.bias is not None:
        x = x + module_self.bias
    return x

def forward_int8_cpmlive(module_self, x):
    if module_self.scale_before:
        x = x / math.sqrt(module_self.dim_in)
    x = x.transpose(1, 2).contiguous()
    x = ct.bmm(module_self.weight.unsqueeze(0), False, x, False, int8=True)
    x = x.transpose(1, 2).contiguous()
    if not module_self.scale_before:
        x = x / math.sqrt(module_self.dim_in)
    return x

#add here
def forward_int4_cpmlive(module_self, x: torch.Tensor,blocksize,compress_statistics,quant_type):
    #quant_state可能需要保存
    original_weights = module_self.weight.data
    w = module_self.weight.data.contiguous().half()
    w_4bit, quant_state = bnb.functional.quantize_4bit(w, blocksize=blocksize, compress_statistics=compress_statistics, quant_type=quant_type)

    if getattr(module_self.weight, 'quant_state', None) is None:
        print('quantization state not initialized. Please ensure that the model parameters you load include the quant_state attribute.')
    inp_dtype = x.dtype
    # dtype_dict = {
    #     'torch.float32': torch.float32,
    #     'torch.float16': torch.float16,
    # }
    # if module_self.compute_dtype is not None:
    #     if isinstance(module_self.compute_dtype, str):
    #         module_self.compute_dtype = dtype_dict[module_self.compute_dtype]
    #     x = x.to(dtype=module_self.compute_dtype)
    if module_self.scale_before:
        x = x / math.sqrt(module_self.dim_in)
    out = bnb.matmul_4bit(x, w_4bit.t(), bias=None, quant_state=quant_state)
    out = out.to(inp_dtype)
    if not module_self.scale_before:
        x = x / math.sqrt(module_self.dim_in)
    module_self.weight.data = original_weights
    return out