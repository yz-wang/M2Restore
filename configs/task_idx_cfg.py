
_total_task_list = ['derain', 'dehaze', 'desnow', 'deraindrop', 'cdd11']

def get_task_info(dataset: str, type: str, **kwargs):
    _task_dict = {}
    if dataset == 'allweather':
        _task_dict['list'] = ['derain', 'deraindrop', 'desnow', 'dehaze']
        _task_dict['scale'] = '1+1+1'
        _task_dict['idx'] = {'derain': 0, 'deraindrop': 1, 'desnow': 2, 'dehaze': 3}
        _task_dict['task'] = {0: 'derain', 1: 'deraindrop', 2: 'desnow', 3: 'dehaze'}
    elif dataset == 'cdd11':
        _task_dict['list'] = ['derain', 'haze', 'low', 'snow', 'haze_rain', 'haze_snow', 'low_haze', 'low_haze_rain', 'low_haze_snow', 'low_rain', 'low_snow']
        _task_dict['scale'] = '1+1+1+1+1+1+1+1+1+1+1'
        _task_dict['idx'] = {'derain':0, 'haze':1, 'low':2, 'snow':3, 'haze_rain':4, 'haze_snow':5, 'low_haze':6, 'low_haze_rain':7, 'low_haze_snow':8, 'low_rain':9, 'low_snow':10}
        _task_dict['task'] = {0:'derain', 1:'haze', 2:'low', 3:'snow', 4:'haze_rain', 5:'haze_snow', 6:'low_haze', 7:'low_haze_rain', 8:'low_haze_snow', 9:'low_rain', 10:'low_snow'}
    elif dataset == 'cityscapes':
        _task_dict['list'] = ['derain', 'dehaze']
        _task_dict['scale'] = '1+1'
        _task_dict['idx'] = {'derain': 0, 'dehaze': 1}
        _task_dict['task'] = {0: 'derain', 1: 'dehaze'}
    elif dataset == 'raindrop':
        _task_dict['list'] = ['deraindrop']
        _task_dict['scale'] = '1'
        _task_dict['idx'] = {'deraindrop': 0}
        _task_dict['task'] = {0: 'deraindrop'}
    elif dataset == 'snow100k':
        _task_dict['list'] = ['desnow']
        _task_dict['scale'] = '1'
        _task_dict['idx'] = {'desnow': 0}
        _task_dict['task'] = {0: 'desnow'}
    elif dataset in ['synthetic_rain', 'outdoor_rain']:
        _task_dict['list'] = ['derain']
        _task_dict['scale'] = '1'
        _task_dict['idx'] = {'derain': 0}
        _task_dict['task'] = {0: 'derain'}
    elif dataset == 'ots':
        _task_dict['list'] = ['dehaze']
        _task_dict['scale'] = '1'
        _task_dict['idx'] = {'dehaze': 0}
        _task_dict['task'] = {0: 'dehaze'}
    else:
        raise NotImplementedError
    
    if type == 'list':
        return _task_dict['list']

    elif type == 'dict':
        return _task_dict['dict']
    
    elif type == 'scale':
        return _task_dict['scale']

    elif type == 'idx':  # task -> idx
        assert kwargs['task'] in _task_dict['idx'].keys(), \
            "{} is not in {} _task_dict".format(kwargs['task'], dataset)
        return _task_dict['idx'][kwargs['task']]

    elif type == 'task':  # idx -> task
        assert kwargs['idx'] in _task_dict['task'].keys(), \
            "{} is not in {} _task_dict".format(kwargs['idx'], dataset)
        return _task_dict['task'][kwargs['idx']]