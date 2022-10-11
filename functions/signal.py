# -*- coding: utf-8 -*-
"""
Created on Tue Aug 03 10:43:00 2021

@author: Gudrun Flach
"""
import numpy as np
from scipy.io import wavfile
from pylab import *
from scipy import *
from scipy.signal import butter, freqz, lfilter
import matplotlib.pyplot as plt

class Sweep:
    def __init__(self):
        self.fS, self.Data = 10000, np.ones(10000)
        self.Data = self.Data/max(max(self.Data), abs(min(self.Data)))*0.8
        self.Duration = len(self.Data)/self.fS
        self.lin = True
        
    def plot(self):        
        plt.figure(figsize=(15,4))
        t = np.linspace(0, self.Duration, len(self.Data))
        plt.plot(t, self.Data)
        plt.title('Sweepsignal')
        plt.grid(True)
        plt.xlabel('t in s')
        plt.axis([0,max(t),-1,1])

    def plot_selection(self, start=0, stop=0):
        plt.figure(figsize=(15,4))
        t = np.linspace(start, stop, int(stop*self.fS)-int(start*self.fS))
        plt.plot(t, self.Data[int(start*self.fS):int(stop*self.fS)])
        plt.title('Ausschnitt Sweepsignal')
        plt.grid(True)
        plt.xlabel('t in s', x=1)
        plt.axis([start,stop,-1,1])
    
    def plot_spektrum(self, start=0, stop=0):
        l = len(self.Data[int(start*self.fS):int(stop*self.fS)])
        bs = np.abs(np.fft.fft(self.Data[int(start*self.fS):int(stop*self.fS)]))/l
        f = np.linspace(0,self.fS/2,l//2)
        plt.subplot(212)
        plt.plot(f, bs[0:len(bs)//2])
        plt.title('Amplitudenspektrum Ausschnitt Sweepsignal')
        plt.grid(True)
        plt.xlabel('f in Hz', x=1)
        
    def create_sweep(self, start_f, stop_f, dur, fs, lin=True):
        self.fS = fs
        self.lin = lin
        self.Duration = dur
        t = np.arange(0,dur,1/fs)
        if self.lin == True:
            self.Data = np.sin(np.pi*((stop_f-start_f)/dur*t + 2*start_f)*t)
        else:
            a = np.log(stop_f/start_f)/dur
            self.Data = np.sin(2*np.pi*(start_f/a*(np.exp(a*t)-1)))
        
class Noise:
    def __init__(self):
        self.dur = 3
        self.mu = 0
        self.fS, self.Data = 10000, np.random.normal(0, 0.2, size=self.dur*10000) 
        self.Data_filtered = self.Data
        self.a = self.b = np.zeros(9)
    
    def create_Noise(self, mu, var, dur, fs):
        self.fS = fs
        self.dur = dur
        self.mu = mu
        self.Data = np.random.normal(mu, var, size=self.dur*fs)
                             
    def plot_Noise(self):
        plt.figure(figsize=(15,4))
        t = np.linspace(0, self.dur, len(self.Data))
        plt.plot(t, self.Data)
        plt.title('Rauschsignal (ungefiltert)')
        plt.grid(True)
        plt.xlabel('t in s')
        plt.axis([0,max(t),self.mu-1,self.mu+1])
        

    def filt_noise(self, ampl):
        def butterworth_bandpass(low, high, order):
            nyq = 0.5 * self.fS
            lowcut = low / nyq
            highcut = high / nyq
            self.b, self.a = butter(order, [lowcut, highcut], btype='band')
        butterworth_bandpass(900, 1100, 4)
        self.Data_filtered = lfilter(self.b, self.a, self.Data)
        smax = np.max(np.abs(self.Data_filtered))
        self.Data_filtered = ampl/smax * self.Data_filtered + self.mu

    def plot_Noise_filt(self, smax):
        plt.figure(figsize=(15,4))
        t = np.linspace(0, self.dur, len(self.Data))
        plt.plot(t, self.Data, label = 'Rauschsignal (ungefiltert)')
        plt.plot(t, self.Data_filtered, label = 'Rauschsignal (gefiltert und verstärkt)')
        plt.legend()
        plt.grid(True)
        plt.xlabel('t in s')
        plt.axis([0,max(t),self.mu-smax,self.mu+smax])
        
    def show_spec(self, sig1, sig2):
        sig = sig1 + sig2
        plt.figure(figsize=(15,4))
        f = np.linspace(0, self.fS/2, int(len(sig)/2))
        spec1 = np.abs(fft.fft(sig1))[0:int(len(sig1)/2)]
        spec2 = np.abs(fft.fft(sig2))[0:int(len(sig2)/2)]
        plt.stem(f, spec1, 'r', basefmt=" ", markerfmt=" ", use_line_collection=True, label='Rauschen')
        plt.stem(f, spec2, basefmt=" ", markerfmt=" ", use_line_collection=True, label='Sinuston')
        plt.grid(True)
        plt.legend()
        plt.xlabel('f in Hz')
        plt.title('Signalspektren')

        
        
         
        
        
        