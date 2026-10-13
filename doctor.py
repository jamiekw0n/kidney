"""Report dependencies and devices; never installs packages or downloads weights."""
import importlib
import json
import platform
import sys


def main():
    result={'python':sys.version,'platform':platform.platform(),'packages':{}}
    missing=[]
    for name in ('numpy','PIL','torch','torchvision','timm','tifffile','zarr','imagecodecs'):
        try:
            module=importlib.import_module(name)
            result['packages'][name]=getattr(module,'__version__','installed')
            if name=='torch':
                result['cuda_available']=module.cuda.is_available()
                result['torch_cuda_version']=module.version.cuda
                result['gpu_names']=[module.cuda.get_device_name(i) for i in range(module.cuda.device_count())]
        except Exception as exc:
            result['packages'][name]=f'{type(exc).__name__}: {exc}'
            missing.append(name)
    print(json.dumps(result,indent=2))
    return bool(missing)

if __name__=='__main__':
    sys.exit(main())
