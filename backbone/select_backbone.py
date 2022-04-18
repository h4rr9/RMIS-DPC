# from https://github.com/TengdaHan/DPC/blob/master/backbone/select_backbone.py
import resnet_2d3d


def select_resnet(
    network,
    track_running_stats=True,
):
    param = {'feature_size': 1024}
    if network == 'resnet18':
        model = resnet_2d3d.resnet18_2d3d_full(
            track_running_stats=track_running_stats)
        param['feature_size'] = 256
    elif network == 'resnet34':
        model = resnet_2d3d.resnet34_2d3d_full(
            track_running_stats=track_running_stats)
        param['feature_size'] = 256
    elif network == 'resnet50':
        model = resnet_2d3d.resnet50_2d3d_full(
            track_running_stats=track_running_stats)
    elif network == 'resnet101':
        model = resnet_2d3d.resnet101_2d3d_full(
            track_running_stats=track_running_stats)
    elif network == 'resnet152':
        model = resnet_2d3d.resnet152_2d3d_full(
            track_running_stats=track_running_stats)
    elif network == 'resnet200':
        model = resnet_2d3d.resnet200_2d3d_full(
            track_running_stats=track_running_stats)
    else:
        raise IOError('model type is wrong')

    return model,
