
def get_img_size(dataset_name: str):

    if dataset_name == 'allweather':
        img_size = (480, 720)
    elif dataset_name == 'cityscapes':
        img_size = (1024, 2048)
    elif dataset_name in ['raindrop', 'snow100k', 'outdoor_rain']:
        img_size = (480, 720)
    elif dataset_name == 'synthetic_rain':
        img_size = (320, 480)
    elif dataset_name == 'ots':
        img_size = (640, 640)
    elif dataset_name == 'cdd11':
        img_size = (720,1080)
    else:
        raise NotImplementedError

    return img_size

def get_crop_ratio(dataset_name: str):
    
    if dataset_name == 'allweather':
        crop_ratio = (224.0/480.0, 224.0/720.0)
    elif dataset_name == 'cdd11':
        crop_ratio = (224.0/720, 224.0/1080.0)
    elif dataset_name == 'cityscapes':
        crop_ratio = (1.0/4.0, 1.0/8.0)
        # crop_ratio = (1.0/2.0, 1.0/4.0)
    elif dataset_name in ['raindrop', 'snow100k', 'synthetic_rain', 'outdoor_rain']:
        crop_ratio = (1.0/2.0, 1.0/3.0)
    elif dataset_name == 'ots':
        crop_ratio = (1.0, 1.0)
    else:
        raise NotImplementedError

    return crop_ratio


def get_dataset_root(dataset_name: str):
    
    if dataset_name == 'allweather':
        root = '/home/data/allweather/allweather'
    elif dataset_name == 'cityscapes':
        root = '/home/data/cityscapes/cityscapes'
    elif dataset_name == 'raindrop':
        root = '/home/data/weather/raindrop/RainDrop'
    elif dataset_name == 'snow100k':
        root = '/home/data/weather/snow/snow100k'
    elif dataset_name == 'synthetic_rain':
        root = '/home/data/weather/rain/synthetic_rain'
    elif dataset_name == 'outdoor_rain':
        root = '/home/data/weather/rain/outdoor_rain'
    elif dataset_name == 'ots':
        root = '/home/data/weather/haze/reside/OTS'
    elif dataset_name == 'cdd11':
        root = '/home/data/CDD11/cdd11'
    else:
        raise NotImplementedError

    return root

def get_no_val_dataset(): 
    return ['allweather', 'raindrop', 'snow100k', 'synthetic_rain', 'outdoor_rain', 'ots', 'cdd11']