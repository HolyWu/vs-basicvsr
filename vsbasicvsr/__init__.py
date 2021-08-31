import math
import numpy as np
import os
import torch
import vapoursynth as vs
from .basicvsr_arch import BasicVSR as BVSR


def BasicVSR(clip: vs.VideoNode, model: int=0, radius: int=7, tile_x: int=0, tile_y: int=0, tile_pad: int=10,
             device_type: str='cuda', device_index: int=0, fp16: bool=False) -> vs.VideoNode:
    '''
    BasicVSR: The Search for Essential Components in Video Super-Resolution and Beyond

    Currently only x4 is supported.

    Parameters:
        clip: Clip to process. Only planar format with float sample type of 32 bit depth is supported.

        model: Model to use.
            0 = REDS
            1 = Vimeo-90K (BI)
            2 = Vimeo-90K (BD)

        radius: Temporal radius. Interval size is (radius * 2 + 1).

        tile_x, tile_y: Tile width and height respectively, 0 for no tiling.
            It's recommended that the input's width and height is divisible by the tile's width and height respectively.
            Set it to the maximum value that your GPU supports to reduce its impact on the output.

        tile_pad: Tile padding.

        device_type: Device type on which the tensor is allocated. Must be 'cuda' or 'cpu'.

        device_index: Device ordinal for the device type.

        fp16: fp16 mode for faster and more lightweight inference on cards with Tensor Cores.
    '''
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('BasicVSR: this is not a clip')

    if clip.format.id != vs.RGBS:
        raise vs.Error('BasicVSR: only RGBS format is supported')

    if model not in [0, 1, 2]:
        raise vs.Error('BasicVSR: model must be 0, 1, or 2')

    if radius < 1:
        raise vs.Error('BasicVSR: radius must be at least 1')

    device_type = device_type.lower()

    if device_type not in ['cuda', 'cpu']:
        raise vs.Error("BasicVSR: device_type must be 'cuda' or 'cpu'")

    if device_type == 'cuda' and not torch.cuda.is_available():
        raise vs.Error('BasicVSR: CUDA is not available')

    device = torch.device(device_type, device_index)
    if device_type == 'cuda':
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        if fp16:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)

    if model == 0:
        model_name = 'BasicVSR_REDS4.pth'
    elif model == 1:
        model_name = 'BasicVSR_Vimeo90K_BIx4.pth'
    else:
        model_name = 'BasicVSR_Vimeo90K_BDx4.pth'
    model_path = os.path.join(os.path.dirname(__file__), model_name)

    model = BVSR(num_feat=64, num_block=30)
    model.load_state_dict(torch.load(model_path)['params'], strict=True)
    model.eval()
    model = model.to(device)

    new_clip = clip.std.BlankClip(width=clip.width * 4, height=clip.height * 4)

    def basicvsr(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        imgs = [frame_to_tensor(f[0])]
        for i in range(1, radius + 1):
            imgs.insert(0, frame_to_tensor(clip.get_frame(max(n - i, 0))))
            imgs.append(frame_to_tensor(clip.get_frame(min(n + i, clip.num_frames - 1))))
        imgs = torch.stack(imgs)
        imgs = imgs.unsqueeze(0).to(device)
        if fp16:
            imgs = imgs.half()

        with torch.no_grad():
            if tile_x > 0 and tile_y > 0:
                output = tile_process(imgs, tile_x, tile_y, tile_pad, model)
            else:
                output = model(imgs)

        return tensor_to_frame(output, f[1])

    return new_clip.std.ModifyFrame(clips=[clip, new_clip], selector=basicvsr)


def frame_to_tensor(f: vs.VideoFrame) -> torch.Tensor:
    arr = np.stack([np.asarray(f.get_read_array(plane)) for plane in range(f.format.num_planes)])
    return torch.from_numpy(arr)


def tensor_to_frame(t: torch.Tensor, f: vs.VideoFrame) -> vs.VideoFrame:
    arr = t.data.squeeze().cpu().numpy()
    fout = f.copy()
    for plane in range(fout.format.num_planes):
        np.copyto(np.asarray(fout.get_write_array(plane)), arr[plane, :, :])
    return fout

def tile_process(img: torch.Tensor, tile_x: int, tile_y: int, tile_pad: int, model: BVSR) -> torch.Tensor:
    batch, _, channel, height, width = img.shape
    output_height = height * 4
    output_width = width * 4
    output_shape = (batch, channel, output_height, output_width)

    # start with black image
    output = img.new_zeros(output_shape)

    tiles_x = math.ceil(width / tile_x)
    tiles_y = math.ceil(height / tile_y)

    # loop over all tiles
    for y in range(tiles_y):
        for x in range(tiles_x):
            # extract tile from input image
            ofs_x = x * tile_x
            ofs_y = y * tile_y

            # input tile area on total image
            input_start_x = ofs_x
            input_end_x = min(ofs_x + tile_x, width)
            input_start_y = ofs_y
            input_end_y = min(ofs_y + tile_y, height)

            # input tile area on total image with padding
            input_start_x_pad = max(input_start_x - tile_pad, 0)
            input_end_x_pad = min(input_end_x + tile_pad, width)
            input_start_y_pad = max(input_start_y - tile_pad, 0)
            input_end_y_pad = min(input_end_y + tile_pad, height)

            # input tile dimensions
            input_tile_width = input_end_x - input_start_x
            input_tile_height = input_end_y - input_start_y

            input_tile = img[:, :, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

            # upscale tile
            output_tile = model(input_tile)

            # output tile area on total image
            output_start_x = input_start_x * 4
            output_end_x = input_end_x * 4
            output_start_y = input_start_y * 4
            output_end_y = input_end_y * 4

            # output tile area without padding
            output_start_x_tile = (input_start_x - input_start_x_pad) * 4
            output_end_x_tile = output_start_x_tile + input_tile_width * 4
            output_start_y_tile = (input_start_y - input_start_y_pad) * 4
            output_end_y_tile = output_start_y_tile + input_tile_height * 4

            # put tile into output image
            output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = \
                output_tile[:, :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile]

    return output
