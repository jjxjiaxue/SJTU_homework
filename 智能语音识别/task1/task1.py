#!/usr/bin/env python
# coding: utf-8

# In[2]:


def calEnergy(wave_data) :
    energy = []
    sum = 0
    for i in range(len(wave_data)) :
        sum = sum + (int(wave_data[i]) * int(wave_data[i]))
        if (i + 1) % 256 == 0 :
            energy.append(sum)
            sum = 0
        elif i == len(wave_data) - 1 :
            energy.append(sum)
    return energy



# 自定义函数，计算数值的符号。
def sgn(data):
    if data >= 0 :
        return 1
    else :
        return 0
#计算过零率
def calZeroCrossingRate(wave_data) :
    zeroCrossingRate = []
    sum = 0
    for i in range(len(wave_data)) :
        if i % 256 == 0:
            continue
        sum = sum + np.abs(sgn(wave_data[i]) - sgn(wave_data[i - 1]))
        if (i + 1) % 256 == 0 :
            zeroCrossingRate.append(float(sum) / 255)
            sum = 0
        elif i == len(wave_data) - 1 :
            zeroCrossingRate.append(float(sum) / 255)
    return zeroCrossingRate



# 利用短时能量，短时过零率，使用双门限法进行端点检测
def endPointDetect(wave_data, energy, zeroCrossingRate) :
    sum = 0
    energyAverage = 0
    for en in energy :
        sum = sum + en
    energyAverage = sum / len(energy)

    sum = 0
    for en in energy[:5] :
        sum = sum + en
    ML = sum / 5                        
    MH = energyAverage / 4              #较高的能量阈值
    ML = (ML + MH) / 4    #较低的能量阈值
    sum = 0
    for zcr in zeroCrossingRate[:5] :
        sum = float(sum) + zcr             
    Zs = sum / 5                     #过零率阈值

    A = []
    B = []
    C = []

    # 首先利用较大能量阈值 MH 进行初步检测
    flag = 0
    for i in range(len(energy)):
        if len(A) == 0 and flag == 0 and energy[i] > MH :
            A.append(i)
            flag = 1
        elif flag == 0 and energy[i] > MH and i - 21 > A[len(A) - 1]:
            A.append(i)
            flag = 1
        elif flag == 0 and energy[i] > MH and i - 21 <= A[len(A) - 1]:
            A = A[:len(A) - 1]
            flag = 1

        if flag == 1 and energy[i] < MH :
            A.append(i)
            flag = 0
    #print("较高能量阈值，计算后的浊音A:" + str(A))

    # 利用较小能量阈值 ML 进行第二步能量检测
    for j in range(len(A)) :
        i = A[j]
        if j % 2 == 1 :
            while i < len(energy) and energy[i] > ML :
                i = i + 1
            B.append(i)
        else :
            while i > 0 and energy[i] > ML :
                i = i - 1
            B.append(i)
    #print("较低能量阈值，增加一段语言B:" + str(B))

    # 利用过零率进行最后一步检测
    for j in range(len(B)) :
        i = B[j]
        if j % 2 == 1 :
            while i < len(zeroCrossingRate) and zeroCrossingRate[i] >= 3 * Zs :
                i = i + 1
            C.append(i)
        else :
            while i > 0 and zeroCrossingRate[i] >= 3 * Zs :
                i = i - 1
            C.append(i)
    #print("过零率阈值，最终语音分段C:" + str(C))
    return C



# In[3]:


def load_label_data():
    dev_path='dev_label.txt'
    dev_label=read_label_from_file(dev_path,0.032,0.008)
    return dev_label
from pathlib import Path
def load_label_data2():
    pred_path='pred_label.txt'
    pred_label=read_label_from_file(pred_path,0.032,0.008)
    return pred_label


def parse_vad_label(line, frame_size: float = 0.032, frame_shift: float = 0.008):
    """Parse VAD information in each line, and convert it to frame-wise VAD label.

    Args:
        line (str): e.g. "0.2,3.11 3.48,10.51 10.52,11.02"
        frame_size (float): frame size (in seconds) that is used when
                            extarcting spectral features
        frame_shift (float): frame shift / hop length (in seconds) that
                            is used when extarcting spectral features
    Returns:
        frames (List[int]): frame-wise VAD label

    Examples:
        >>> label = parse_vad_label("0.3,0.5 0.7,0.9")
        [0, ..., 0, 1, ..., 1, 0, ..., 0, 1, ..., 1]
        >>> print(len(label))
        110

    NOTE: The output label length may vary according to the last timestamp in `line`,
    which may not correspond to the real duration of that sample.

    For example, if an audio sample contains 1-sec silence at the end, the resulting
    VAD label will be approximately 1-sec shorter than the sample duration.

    Thus, you need to pad zeros manually to the end of each label to match the number
    of frames in the feature. E.g.:
        >>> feature = extract_feature(audio)    # frames: 320
        >>> frames = feature.shape[1]           # here assumes the frame dimention is 1
        >>> label = parse_vad_label(vad_line)   # length: 210
        >>> import numpy as np
        >>> label_pad = np.pad(label, (0, np.maximum(frames - len(label), 0)))[:frames]
    """
    frame2time = lambda n: n * frame_shift + frame_size / 2
    frames = []
    frame_n = 0
    for time_pairs in line.split():
        start, end = map(float, time_pairs.split(","))
        assert end > start, (start, end)
        while frame2time(frame_n) < start:
            frames.append(0)
            frame_n += 1
        while frame2time(frame_n) <= end:
            frames.append(1)
            frame_n += 1
    return frames


def prediction_to_vad_label(
    prediction,
    frame_size: float = 0.032,
    frame_shift: float = 0.008,
    threshold: float = 0.5,
):
    """Convert model prediction to VAD labels.

    Args:
        prediction (List[float]): predicted speech activity of each **frame** in one sample
            e.g. [0.01, 0.03, 0.48, 0.66, 0.89, 0.87, ..., 0.72, 0.55, 0.20, 0.18, 0.07]
        frame_size (float): frame size (in seconds) that is used when
                            extarcting spectral features
        frame_shift (float): frame shift / hop length (in seconds) that
                            is used when extarcting spectral features
        threshold (float): prediction values that are higher than `threshold` are set to 1,
                            and those lower than or equal to `threshold` are set to 0
    Returns:
        vad_label (str): converted VAD label
            e.g. "0.31,2.56 2.6,3.89 4.62,7.99 8.85,11.06"

    NOTE: Each frame is converted to the timestamp according to its center time point.
    Thus the converted labels may not exactly coincide with the original VAD label, depending
    on the specified `frame_size` and `frame_shift`.
    See the following exmaple for more detailed explanation.

    Examples:
        >>> label = parse_vad_label("0.31,0.52 0.75,0.92")
        >>> prediction_to_vad_label(label)
        '0.31,0.53 0.75,0.92'
    """
    frame2time = lambda n: n * frame_shift + frame_size / 2
    speech_frames = []
    prev_state = False
    start, end = 0, 0
    end_prediction = len(prediction) - 1
    for i, pred in enumerate(prediction):
        state = pred > threshold
        if not prev_state and state:
            # 0 -> 1
            start = i
        elif not state and prev_state:
            # 1 -> 0
            end = i
            speech_frames.append(
                "{:.2f},{:.2f}".format(frame2time(start), frame2time(end))
            )
        elif i == end_prediction and state:
            # 1 -> 1 (end)
            end = i
            speech_frames.append(
                "{:.2f},{:.2f}".format(frame2time(start), frame2time(end))
            )
        prev_state = state
    return " ".join(speech_frames)


##############################################
# Examples of how to use the above functions #
##############################################
def read_label_from_file(
    path="data/dev_label.txt", frame_size: float = 0.032, frame_shift: float = 0.008
):
    """Read VAD information of all samples, and convert into
    frame-wise labels (not padded yet).

    Args:
        path (str): Path to the VAD label file.
        frame_size (float): frame size (in seconds) that is used when
                            extarcting spectral features
        frame_shift (float): frame shift / hop length (in seconds) that
                            is used when extarcting spectral features
    Returns:
        data (dict): Dictionary storing the frame-wise VAD
                    information of each sample.
            e.g. {
                "1031-133220-0062": [0, 0, 0, 0, ... ],
                "1031-133220-0091": [0, 0, 0, 0, ... ],
                ...
            }
    """
    data = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for linenum, line in enumerate(f, 1):
            sps = line.strip().split(maxsplit=1)
            if len(sps) == 1:
                print(f'Error happened with path="{path}", id="{sps[0]}", value=""')
            else:
                k, v = sps
            if k in data:
                raise RuntimeError(f"{k} is duplicated ({path}:{linenum})")
            data[k] = parse_vad_label(v, frame_size=frame_size, frame_shift=frame_shift)
    return data


# In[4]:


dev_label=load_label_data()
print(len(dev_label))


# In[5]:


import os
filepath="/Users/zhangshijie/Desktop/vad/wavs/dev"
filenames=os.listdir(filepath)


# In[10]:


import os
import wave
import numpy as np
f=open("pred_label.txt","w").close()
f=open("ground_truth.txt","w").close()
filepath="/Users/zhangshijie/Desktop/vad/wavs/test"
filenames=os.listdir(filepath)
j=0
for filename in filenames:
    print(j,filename)
    if(filename=='.DS_Store'):
        continue
    j+=1
    f = wave.open(filepath+'/'+filename,"rb")
    params = f.getparams()
# nframes 采样点数目
    nchannels, sampwidth, framerate, nframes = params[:4]
# readframes() 按照采样点读取数据
    str_data = f.readframes(nframes)            # str_data 是二进制字符串

# 以上可以直接写成 str_data = f.readframes(f.getnframes())

# 转成二字节数组形式（每个采样点占两个字节）
    wave_data = np.fromstring(str_data, dtype = np.short)
    #print( "采样点数目：" + str(len(wave_data)))          #输出应为采样点数目
    f.close()
    energy = calEnergy(wave_data)
    zeroCrossingRate = calZeroCrossingRate(wave_data)
    C=endPointDetect(f, energy, zeroCrossingRate)
    print(len(wave_data)/framerate)
    if(len(C)==1):
        print(filename,"error A!error A!error A!error A!")
        #plot("")
        continue;
    if(len(C)>2):
        if(C[0]==C[2]):
            print(filename,"error B!error B!error B!error B!")
            continue
    for i in range(len(C)):
        C[i]=C[i]*256/framerate
    #print(C)
    real_filename = filename.split('.')[0]
    if(len(C)%2==0):
        with open("pred_label.txt","a") as f:
            f.write(real_filename+" ")
            print(C)
       
            
            for i in range(0,len(C),2):
                f.write(str(C[i])+","+str(C[i+1])+" ")
            f.write('\n')
    else:
        with open("pred_label.txt","a") as f:
            f.write(real_filename+" ")
            print(C)
       
            
            for i in range(0,len(C)-1,2):
                f.write(str(C[i])+","+str(C[i+1])+" ")
            f.write(str(C[i])+","+str(len(wave_data)/framerate)+" ")
            f.write('\n')


# In[65]:


print(len(C))


# In[67]:



prediction=load_label_data2()
print(len(prediction))


# In[73]:



for key,value in prediction.items():
    n=min(len(prediction[key]),len(dev_label[key]))
    prediction[key]=prediction[key][:n]
    dev_label[key]=dev_label[key][:n]
x=prediction['834-130871-0052'];
y=dev_label['834-130871-0052'];
for key,value in prediction.items():
    x=x+prediction[key]
    y=y+dev_label[key]


# In[72]:


for key,value in prediction.items():
    n=min(len(prediction[key]),len(dev_label[key]))
    prediction[key]=prediction[key][:n]
    dev_label[key]=dev_label[key][:n]


# In[75]:


from sklearn import metrics


def compute_eer(target_scores, nontarget_scores):
    """Calculate EER following the same way as in Kaldi.

    Args:
        target_scores (array-like): sequence of scores where the
                                    label is the target class
        nontarget_scores (array-like): sequence of scores where the
                                    label is the non-target class
    Returns:
        eer (float): equal error rate
        threshold (float): the value where the target error rate
                           (the proportion of target_scores below
                           threshold) is equal to the non-target
                           error rate (the proportion of nontarget_scores
                           above threshold)
    """
    assert len(target_scores) != 0 and len(nontarget_scores) != 0
    tgt_scores = sorted(target_scores)
    nontgt_scores = sorted(nontarget_scores)

    target_size = float(len(tgt_scores))
    nontarget_size = len(nontgt_scores)
    target_position = 0
    for target_position, tgt_score in enumerate(tgt_scores[:-1]):
        nontarget_n = nontarget_size * target_position / target_size
        nontarget_position = int(nontarget_size - 1 - nontarget_n)
        if nontarget_position < 0:
            nontarget_position = 0
        if nontgt_scores[nontarget_position] < tgt_score:
            break
    threshold = tgt_scores[target_position]
    eer = target_position / target_size
    return eer, threshold


def get_metrics(prediction, label):
    """Calculate several metrics for a binary classification task.

    Args:
        prediction (array-like): sequence of probabilities
            e.g. [0.1, 0.4, 0.35, 0.8]
        labels (array-like): sequence of class labels (0 or 1)
            e.g. [0, 0, 1, 1]
    Returns:
        auc: area-under-curve
        eer: equal error rate
    """  # noqa: H405, E261
    assert len(prediction) == len(label), (len(prediction), len(label))
    fpr, tpr, thresholds = metrics.roc_curve(label, prediction, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    # from scipy.optimize import brentq
    # from scipy.interpolate import interp1d
    # fnr = 1 - tpr
    # eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    eer, thres = compute_eer(
        [pred for i, pred in enumerate(prediction) if label[i] == 1],
        [pred for i, pred in enumerate(prediction) if label[i] == 0],
    )
    return auc, eer


if __name__ == "__main__":
    # 第一个参数为模型预测输出（可以是概率，也可以是二值分类结果）
    # 第二个参数为数据对应的标签
    print(get_metrics(x,y))
    # 注意：计算最终指标时，应将整个数据集（而不是在每个样本上单独计算）的所有语音帧预测结果合并在一个list中，
    # 对应的标签也合并在一个list中，然后再调用get_metrics来计算指标
k=0
for i in range(len(x)):
    if(x[i]==y[i]):
        k+=1;
print("accuracy =" , k/len(x))


# In[ ]:




