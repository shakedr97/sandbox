import numpy as np
import matplotlib.pyplot as plt
import time
import random
from enum import Enum

# TODO:
# allow user to select the amplitude and mixture of noise types
# make correlation calculations faster
# clean up
# allow user to choose problem dimensions
# display all steps at the end

class DebugLevel(Enum):
    NONE = 0
    INFO = 1
    DEBUG = 2
    TRACE = 3

def step(width, point, high=1, low=0):
    return high if (-0.1 < point < width + 0.1) else low

def create_single_time_data(timestamp, step_width=10, step_length=20):
    data = np.linspace(1, 1000, 1000)
    data = data - (timestamp - (timestamp % step_length))
    data = [step(step_width, x) for x in data]
    return data

def create_sweep_data(time_offset, step_width=10, step_length=20):
    sweep_length = 1500
    data = []
    for t in range(sweep_length):
        data.append(create_single_time_data(t + time_offset, step_width, step_length))
    return data

def generate_frequency_response(smoothing_width=250, size=1000, scale=10):
    """
    randomly generate a mock frequency response.
    the response function is random, but continuous. this is acheived by smoothing a random sequence.\n
     * smoothing_width -> smoothing window width\n
     * size -> size of the sequence representing the frequency response\n
     * scale -> the max value of the frequency response to aim for, so that the response has values in the range [0, {scale}).
    """
    # random seed
    diffs = np.random.random(smoothing_width + size)
    response = []
    # 
    for i in range(size):
        point = 0
        for j in range(i, i + smoothing_width):
            point += diffs[j]
        response.append(point)
    
    # normalize reponse so max is half the selected scale and average is 0
    response = np.array(response)
    response = response - np.mean(response)
    response = response * scale / 2 / np.max(response)
    # slide response up so it is non-negative
    response = response - np.min(response)
    return response

def log_sum(a, b):
    return 10 * np.log10(np.pow(10, 0.1 * a) + np.pow(10, 0.1 * b))

def generate_random_noise(amplitude_factor=4):
    return generate_frequency_response() * amplitude_factor

def generate_spurious_freq_spurious_time_noise(width=20, likelihood=0.1, timescale=1500, freqscale=1000, amplitude_scale=10):
    time_sequence = generate_poisson_one_d_sequence(length=timescale, likelihood=likelihood)
    base_freq = np.random.randint(0, freqscale)
    freq_sequence = np.array([step(width, point) for point in range(-base_freq, freqscale - base_freq)])
    single_spur_data = time_sequence[:, np.newaxis] * freq_sequence[np.newaxis, :] * np.random.randint(1, 10)
    single_spur_data = 10 * np.log10(single_spur_data + 1e-9) 
    return single_spur_data * (amplitude_scale / 10)

def generate_poisson_one_d_sequence(length=1000, likelihood=0.1, low=0):
    sequence = []
    while len(sequence) < length:
        n = random.random()
        if n > likelihood:
            non_section_length = np.random.poisson((1-likelihood) * length / 10)
            for _ in range(non_section_length):
                sequence.append(low)
        else:
            section_length = np.random.poisson(likelihood * length / 10)
            for _ in range(section_length):
                sequence.append(1)
    return np.array(sequence[:length])

def generate_spurious_spectrum(num_spurs=30, amplitude_min=5, amplitude_max=40):
    data = None
    for _ in range(num_spurs):
            likelihood = np.random.randint(1, 20) / 20
            amplitude_scale = np.random.randint(amplitude_min, amplitude_max)
            if data is None:
                data = generate_spurious_freq_spurious_time_noise(likelihood=likelihood, amplitude_scale=amplitude_scale)
            else:
                data = log_sum(data, generate_spurious_freq_spurious_time_noise(likelihood=likelihood, amplitude_scale=amplitude_scale))
    return data

def generate_sudden_wide_noise(timescale=1500, freqscale=1000,amplitude_max=40, freq_hole_size=200):
    start_time = random.randint(1, timescale)
    length = random.randint(1, timescale)
    base_mask = [0] * start_time + [1] * length
    base_mask = base_mask + [0] * (timescale - len(base_mask))
    base_mask = np.array(base_mask[:timescale])
    freq_hole_start = random.randint(1, freqscale)
    freq = [1] * freq_hole_start + [0] * freq_hole_size + [1] * freqscale
    freq = np.array(freq[:freqscale])
    mask = base_mask[:, np.newaxis] * freq[np.newaxis, :]
    return np.log10(np.random.random((timescale, freqscale)) * mask * 10 + 1e-90) * amplitude_max

def create_mask(sum_zero_mask=False, point=100, step_size=10, step_length=20):
    mask = np.array(create_sweep_data(point, step_size, step_length))
    if not sum_zero_mask:
        return mask
    offset = step_length // 2

    up_mask = mask[offset:, :]
    extra = np.zeros((offset, mask.shape[1]))
    up_mask = np.concatenate([up_mask, extra])

    # move mask 5 points down
    down_mask = mask[:mask.shape[0]-offset, :]
    extra = np.zeros((offset, mask.shape[1]))
    down_mask = np.concatenate([extra, down_mask])

    mix_mask = np.logical_or(up_mask,down_mask)

    return 2 * mask - mix_mask
    
def get_sweep_start_time(data, sum_zero_mask=False, debug_level=DebugLevel.NONE):
    start = time.time()
    corr = []
    coarse_step = 15
    for i in range(-500, 0, coarse_step):
        mask = create_mask(sum_zero_mask=sum_zero_mask, point=i, step_size=coarse_step)
        if debug_level == DebugLevel.TRACE:
            plt.imshow(mask, cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.show()
        conv = np.multiply(data, mask)
        single_corr = np.sum(conv)
        corr.append(single_corr)
        print(i)
    stop = time.time()
    print(stop - start)
    plt.plot(corr)
    plt.show()

    max_index = np.argmax(corr)
    max_value = -500 + coarse_step * max_index
    step = 1
    start = time.time()
    corr = []
    start_value = max_value - coarse_step
    stop_value = max_value + coarse_step
    for i in range(start_value, stop_value, step):
        mask = create_mask(sum_zero_mask=sum_zero_mask, point=i, step_size=step)
        conv = np.multiply(data, mask)
        single_corr = np.sum(conv)
        corr.append(single_corr)
        print(i)
    stop = time.time()
    print(stop - start)
    plt.plot(corr)
    plt.show()

    max_index = np.argmax(corr)
    final_value = start_value + step * max_index
    return final_value

def get_percentile_data(data_freqs, data_amps, percentile=80):
    indexed_data = {}
    for point in range(len(data_freqs)):
        freq = data_freqs[point]
        amp = data_amps[point]
        if freq not in indexed_data.keys():
            indexed_data[freq] = []
        else:
            indexed_data[freq].append(amp)
    
    percentile_data = {}
    for freq in indexed_data:
        print(freq)
        indexed_data[freq].sort()
        percentile_index = len(indexed_data[freq]) * percentile // 100
        print(f'freq {freq} length: {len(indexed_data[freq])}, percentile_index: {percentile_index}')
        percentile_data[freq] = indexed_data[freq][percentile_index]
    return percentile_data


if __name__ == '__main__':
    demo = False
    if demo:
        noise = generate_spurious_spectrum()

        sudden_noise = generate_sudden_wide_noise()
        print(sudden_noise.shape)
        plt.imshow(sudden_noise, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.show()

        noise = log_sum(noise, sudden_noise)
        plt.imshow(noise, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.show()
    else:
        data = np.array(create_sweep_data(-319))
        response = generate_frequency_response()
        data = data * response[None, :]

        plt.imshow(data, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.show()

        print(data.shape)
        print(response.shape)
        # noise_response = generate_random_noise()
        # plt.plot(noise_response)
        # plt.show()
        # noise = np.random.rand(data.shape[0], data.shape[1]) * noise_response[None, :]
        noise = generate_spurious_spectrum()

        sudden_noise = generate_sudden_wide_noise()
        print(sudden_noise.shape)
        plt.imshow(sudden_noise, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.show()

        noise = log_sum(noise, sudden_noise)
        plt.imshow(noise, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.show()

        # data = data + noise
        data = log_sum(data, noise)
        plt.imshow(data, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.show()

        sweep_start = get_sweep_start_time(data, sum_zero_mask=False)
        print(sweep_start)
        sweep_start = get_sweep_start_time(data, sum_zero_mask=True)
        print(sweep_start)

        sweep_mask = np.array(create_sweep_data(sweep_start))
        sweep_freqs = []
        sweep_amps = []
        noise_freqs = []
        noise_amps = []
        for i in range(sweep_mask.shape[0]):
            for j in range(sweep_mask.shape[1]):
                point_freq = j
                point_amplitude = data[i][j]
                if sweep_mask[i][j]:
                    sweep_freqs.append(point_freq)
                    sweep_amps.append(point_amplitude)
                else:
                    noise_freqs.append(point_freq)
                    noise_amps.append(point_amplitude)
        
        plt.plot(sweep_freqs, sweep_amps, 'o', c='g', label='sweep', markersize=1)
        plt.plot(response)
        plt.show()
        plt.plot(noise_freqs, noise_amps, 'o', c='r', label='noise', markersize=1)
        plt.plot(response)
        plt.plot(sweep_freqs, sweep_amps, 'o', c='g', label='sweep', markersize=1)
        plt.show()

        # do "histograms"
        sweep_percentile = get_percentile_data(sweep_freqs, sweep_amps, percentile=10)
        noise_percentile = get_percentile_data(noise_freqs, noise_amps, percentile=90)
        diff = [sweep_percentile[k] - noise_percentile[k] for k in sweep_percentile.keys()]
        diff = [d if d > 0 else 0 for d in diff]
        plt.plot(sweep_percentile.keys(), sweep_percentile.values(), 'o', c='g', label='sweep', markersize=1)
        plt.plot(noise_percentile.keys(), noise_percentile.values(), 'o', c='r', label='sweep', markersize=1)
        plt.plot(sweep_percentile.keys(), diff)
        plt.show()
        
