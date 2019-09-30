
import librosa
import pandas as pd
import numpy as np
import csv
from IPython.display import clear_output
import os
import warnings
warnings.filterwarnings('ignore')
import soundfile as sf
from Indices import *
import glob
import time

#·························DB path·····················
#% cd /media/david/DavidDD/2017
os.chdir("/media/david/DavidDD/bioacustica/2017")  ##direccion de disco duro


#···········Arguments for function-·····························
tipo_ventana = "hann"
sobreposicion = 0
tamano_ventana = 512
nfft= 512
bio_band = (2000, 8000)
tech_band = (200, 1500)
grabadora=glob.glob('*/')

'''Gneres es la grabadora, fecha es la fecha dentro de la grabadora y filename es el nombre de la grabacion'''

#············Declare Datarray··································
header = ["filename", "ACI", "NDSI",  "M", "mba","bnf" ,"fm", "we", "rms", "cf",
                        "ADIm1", "ADIm2", "ADIm3", "ADIm4", "ADIm5", "ADIm6", "ADIm7", "ADIm8",
                       "ADIm9", "ADIm10", "ADIm11","PSD"]


for g in grabadora[3:]:        				#grabadora[1:]
    clear_output()
    print(g+'+++++++++++++++++++++++++++++++++++++++++++++          ')
    for fecha in glob.glob(g+'/*/'):
    
        df = pd.DataFrame(columns=header)
        print(fecha+'********************************************')
        file = open('**'+'30sep.csv', 'w', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(header)
        for num, filename in enumerate(glob.glob(fecha+'/*.wav'), start=1):
            try:
                print(filename)
                print("grabacion {} of {}".format(num,len(glob.glob(fecha+'/*.wav'))))
                start_time = time.time()
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
                f2, p = signal.welch(audio, Fs, nperseg=tamano_ventana, window=tipo_ventana,
                                nfft=nfft, noverlap=sobreposicion)                




                df=df.append({"filename":filename, "ACI"=aci,"NDSI":ndsi, "M":me, "mba":mba,"bnf":bnf ,"fm":fm, "we":we, "rms":rmss, "cf":cf,
                        "ADIm1":ADIM[0], "ADIm2":ADIM[1], "ADIm3":ADIM[2], "ADIm4":ADIM[3], "ADIm5":ADIM[4], "ADIm6":ADIM[5], "ADIm7":ADIM[6], "ADIm8":ADIM[7],
                       "ADIm9":ADIM[8], "ADIm10":ADIM[9], "ADIm11":ADIM[10], "PSD":np.mean(p)}, ignore_index=True )

                df.to_csv(fecha+'variablessegment.csv')

                print("--- %s seconds ---" % (time.time() - start_time))

            except:
                print("grabacion con error:"+filename)






