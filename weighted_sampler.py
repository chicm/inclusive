import random
import numpy as np
from collections import Counter
import time
from utils import get_classes, get_train_val_meta, get_weights_by_counts
import settings
'''
void Resample(float* input, float* weights, float* outputs, int inputN, int outN)
{
    float sumWeights = 0.0f;
    for (int i = 0; i < inputN; ++i)
    {
        sumWeights += weights[i];
    }
    
    float sampleWidth = sumWeights / outN;
    std::default_random_engine generator;
    std::uniform_real_distribution<float> rnd(0, sampleWidth);
    int outputSampleIx = -1;
    float weightSoFar = -rnd(generator);
    for (int i = 0; i < outN; ++i)
    {
        // How far is this sample from the origin (minus offset)?       
        float sampleDist = i*sampleWidth;

        // Find which sample to output. Just walk up the samples until the sum
        // of the weights is > to the distance of the current sample
        while (sampleDist >= weightSoFar && outputSampleIx + 1 < inputN)
        {
            weightSoFar += weights[++outputSampleIx];
        }           
        outputs[i] = input[outputSampleIx]; 
    }
}
'''

def weighted_sample_v1(inputs, weights, sample_num):
    sum_weights = sum(weights)

    #sample_width = sum_weights / sample_num
    outputs = []

    for i in range(sample_num):
        #print(i)
        sample_distance = random.random() * sum_weights

        weight_sofar = 0
        out_index = -1
        while sample_distance >= weight_sofar and out_index+1 < len(inputs):
            weight_sofar +=  weights[out_index+1] 
            out_index +=1
        outputs.append(inputs[out_index])

    return outputs


def weighted_sample(inputs, weights, sample_num):
    #print(weights[:100])
    sum_weights = sum(weights)

    #sample_width = sum_weights / sample_num
    outputs = []
    samples = []
    
    for i in range(sample_num):
        samples.append(random.random() * sum_weights)
    samples = sorted(samples)

    #print(samples[:10])
    sample_index = 0
    cur_weights = 0

    for i, w in enumerate(weights):
        while sample_index < sample_num and cur_weights + w > samples[sample_index]:
            outputs.append(inputs[i])
            sample_index += 1
        cur_weights += w

    return outputs

def test_performance():
    inputs = [1,2,3,4,5,6,7,8] *1000
    weights = [1, 5, 10, 1, 20, 1, 5, 1]*1000
    bg = time.time()
    results = weighted_sample(inputs, weights, 100000)
    t = time.time() - bg
    print('time:', t)
    counts = Counter()
    counts.update(results)
    print(counts.most_common(8))

def test():
    inputs = [1,2,3,4,5,6,7,8]
    weights = [1, 5, 10, 1, 20, 1, 5, 1]
    bg = time.time()
    results = weighted_sample(inputs, weights, 1000)
    t = time.time() - bg
    print('time:', t)
    counts = Counter()
    counts.update(results)
    print(counts.most_common(8))


def get_weights(df, col_name, max_scale=100):
    counts = df[col_name].values
    counts = np.sqrt(counts)
    #print('counts1:', counts)
    v_min = np.min(counts)
    v_max = np.max(counts)
    #print(v_min, v_max)
    counts = (counts - v_min ) / (v_max - v_min)
    #print('counts2:', counts)
    counts = 1 - counts
    counts = ((counts * max_scale) + 1) / max_scale
    #print(counts[:10])
    #print('min:', np.min(counts))
    #print('counts3:', counts)
    return counts

def get_weighted_sample(df, sample_num):
    if random.random() < 0.5:
        w = get_weights_by_counts(df['rare_counts'].values, max_weight=500)
    else:
        w = get_weights_by_counts(df['avg_counts'].values, max_weight=500)

    return weighted_sample(df['ImageID'].values, w, sample_num)

def test_sampling():
    classes, stoi = get_classes('trainable', 0, 7172)
    train_meta, val_meta = get_train_val_meta('trainable', 0, 7172)
    print(train_meta.shape)
    print(train_meta.head())
    #train_meta = train_meta.iloc[:100]

    #w = get_weights(train_meta, 'rare_counts')
    w = get_weights_by_counts(train_meta['rare_counts'].values, max_weight=500)
    print('weights:', [int(x) for x in w.tolist()])
    print(w.shape)
    #w = [1]*100

    #sample = weighted_sample(train_meta['ImageID'].values, w, 50000)
    sample = get_weighted_sample(train_meta, 50000)
    counts = Counter()
    print(sample[:10])
    counts.update(sample)
    print(counts.most_common(100))
    common_ids = [x[0] for x in counts.most_common(50000)]

    df_selected = train_meta.set_index('ImageID')
    df_selected = df_selected.loc[sample]
    print(df_selected.head(20))

if __name__ == '__main__':
    #test()
    test_sampling()