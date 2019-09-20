import librosa
import pandas as pd
import numpy as np
import csv
from IPython.display import clear_output
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import soundfile as sf
from Indices import *
import glob
from compute_indice import *

#% cd /media/david/DavidDD/2017
os.chdir("/media/david/DavidDD/2017")  ##direccion de disco duro

header = ["filename","chroma_stft","spec_cent","spec_bw", "ADI", "ACI", "TE", "ESM", "NDSI", "P", "M", "mba","bnf" ,"md","fm", "we", "rms", "cf",
                        "ADIm1", "ADIm2", "ADIm3", "ADIm4", "ADIm5", "ADIm6", "ADIm7", "ADIm8",
                       "ADIm9", "ADIm10", "ADIm11"]
import os
os.getcwd()
genres=glob.glob(f''+'*')
genres

tipo_ventana = "hann"
sobreposicion = 0
tamano_ventana = 512
nfft= 512
bio_band = (2000, 8000)
tech_band = (200, 1500)


grabadora=glob.glob(f''+'*')

'''Gneres es la grabadora, fecha es la fecha dentro de la grabadora y filename es el nombre de la grabacion'''

for g in grabadora:
    clear_output()
    print(g+'+++++++++++++++++++++++++++++++++++++++++++++          ')
    for fecha in glob.glob(g+'/*/'):
        print(fecha+'********************************************')
        file = open(fecha+'vari20sep.csv', 'w', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(header)
        for num, filename in enumerate(glob.glob(fecha+'/*.wav'), start=1):
            try:
                print(filename)
                print("grabacion {} of {}".format(num,len(glob.glob(fecha+'/*.wav'))))
                x, Fs = sf.read(filename)
                audio = x
                nmin = len(audio) // (60 * Fs)
                ## indices acusticos
                f, t, s = signal.spectrogram(audio, Fs, window=tipo_ventana, nperseg=nmin * tamano_ventana,
                                         mode="magnitude", \
                                         noverlap=sobreposicion, nfft=nmin * nfft)
                nmin = len(audio) // (60 * Fs)                
                ADIM=list(ADIm(s, Fs, 1000)[:11])
                adi=ADI(s, 10000, 1000, -50)
                aci=ACItf(audio, Fs, 5, s)
                #beta=beta(s, f)
                te=temporal_entropy(audio, Fs)
                sme=spectral_maxima_entropy(s, f, 482, 8820)
                ndsi=NDSI(s, f, bio_band, tech_band)
                r=rho(s, f, bio_band, tech_band)
                me=median_envelope(audio, Fs, 16)
                #number_of_peaks(s, f, 10 * nmin)
                mba=mid_band_activity(s, f, 450, 3500)
                bnf=np.mean(background_noise_freq(s))
                md=musicality_degree(audio, Fs, tamano_ventana, nfft, tipo_ventana, sobreposicion)
                fm=frequency_modulation(s)
                we=wiener_entropy(audio, tamano_ventana, nfft, tipo_ventana, sobreposicion)
                rmss=rms(audio)
                cf=crest_factor(audio, rmss)
                chroma_stft = np.mean(librosa.feature.chroma_stft(y=x, sr=Fs))
                spec_cent = np.mean(librosa.feature.spectral_centroid(y=x, sr=Fs))
                spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=x, sr=Fs))
                to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(spec_cent)} {np.mean(spec_bw)} {adi} {aci} {te} {sme} {ndsi} {r} {me} {mba} {bnf} {md} {fm} {we} {rmss} {cf}' 
                for i in ADIM: 
                    to_append += f' {i}'
                
                file = open(fecha+'vari20sep.csv', 'a', newline='')

                with file:
                    writer = csv.writer(file)
                    writer.writerow(to_append.split())
                
            except:
                print("grabacion con error:"+filename)
