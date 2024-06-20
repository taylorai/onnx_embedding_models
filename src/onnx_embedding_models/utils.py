import os
import pathlib
import numpy as np

def dict_slice(d, idx):
    """
    Slice a dictionary of jagged arrays, returning a dictionary of singleton numpy arrays.
    """
    return {k: np.array(v[idx]).reshape(1, -1) for k, v in d.items()}

def get_cpu_info():
    # figure out which model to load. check if current cpu is avx2, avx512, avx512_vnni, or arm64
    try:
        cpu_info = print(pathlib.Path("/proc/cpuinfo").read_text())
        if "avx512" in cpu_info and "vnni" in cpu_info:
            return "avx512_vnni"
        elif "avx512" in cpu_info:
            return "avx512"
        elif "avx2" in cpu_info or "x86-64" in cpu_info:
            return "avx2"
        elif "aarch64" in cpu_info.lower():
            return "arm64"
        else:
            return None
    except Exception as e:
        print(f"Failed to read /proc/cpuinfo: {e}")
        try:
            # maybe we're on a mac, use sysctl
            cpu_info = os.popen("sysctl -a").read()
            if "intel" in cpu_info.lower():
                if "avx512" in cpu_info.lower() and "vnni" in cpu_info.lower():
                    return "avx512_vnni"
                elif "avx512" in cpu_info.lower():
                    return "avx512"
                elif "avx2" in cpu_info.lower():
                    return "avx2"
                else:
                    return None
            elif (
                "apple m1" in cpu_info.lower()
                or "apple m2" in cpu_info.lower()
                or "apple m3" in cpu_info.lower()
            ):
                return "arm64"
            else:
                return None
        except Exception as e:
            print(f"Failed to run sysctl -a: {e}")
            return None


def find_onnx_model(repo_path: str):
    cpu_type = get_cpu_info()
    
    # check if there's an 'onnx' folder in the repo
    if os.path.exists(os.path.join(repo_path, "onnx")):
        onnx_models = os.listdir(os.path.join(repo_path, "onnx"))
        onnx_models = [os.path.join(repo_path, "onnx", model) for model in onnx_models if model.endswith(".onnx")]

    # otherwise check for onnx models in the root of the repo
    else:
        onnx_models = os.listdir(repo_path)
        onnx_models = [os.path.join(repo_path, model) for model in onnx_models if model.endswith(".onnx")]

    # try to select the model based on the cpu type
    matching_cpu_models = [model for model in onnx_models if cpu_type is not None and cpu_type in model]
    if len(matching_cpu_models) > 0:
        return matching_cpu_models[0]
    
    # if none matching the cpu type, prefer a quantized model
    quantized_models = [model for model in onnx_models if "quantized" in model]
    if len(quantized_models) > 0:
        return quantized_models[0]
    
    # if no quantized model, just return the first model
    if len(onnx_models) > 0:
        return onnx_models[0]
    else:
        return None