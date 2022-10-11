from ipywidgets import interact, fixed, Layout
import IPython.display as ipd
import numpy as np
import ipywidgets as widgets
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import get_window
from matplotlib import gridspec

class Onselect():

    def __init__(self, ax2, line2, x, y):
        #self.coords = {}
        self.start = 0
        self.stop = 0
        self.ax2 = ax2
        self.line2 = line2
        self.x = x
        self.y = y

    def __call__(self, xmin, xmax):
        indmin, indmax = np.searchsorted(self.x, (xmin, xmax))
        indmax = min(len(self.x)-1, indmax)
        thisx = self.x[indmin:indmax]
        thisy = self.y[indmin:indmax]
        self.start = indmin
        self.stop = indmax
        self.line2.set_data(thisx, thisy)
        self.ax2.set_xlim(thisx[0], thisx[-1])
        self.ax2.set_ylim(thisy.min(), thisy.max())
        fig.canvas.draw()

class Test_sig():
    """
    Klasse zum Erzeugen eines Sinustestsignals
    Frequenz und Amplitude sin über Slider einstellbar
    G. Flach, 19.08.2021
    """
    def __init__(self, fs, dur):
        self.Tsig = np.zeros(fs*dur)
        self.dur = dur
        self.fs = fs
        
    def create_Tsig(self):
        style = {'description_width': 'initial'}
        def gen_sig(sig_f, sig_amp, dur, fs):
            t = np.linspace(0, dur, dur*fs)
            self.Tsig = sig_amp * np.sin(2*np.pi*sig_f*t)
            ipd.display(ipd.Audio(self.Tsig, rate = fs))

        sig_f = widgets.IntSlider( value=1000, min=500, max=1500, step=1, description='Frequenz:')
        sig_amp = widgets.FloatSlider(value=1.0, min=0.1, max=2.0, step=0.1, description='Amplitude:', style=style)
        interact(gen_sig, sig_f=sig_f, sig_amp=sig_amp, dur=fixed(self.dur), fs=fixed(self.fs))
        
class Signal():
    """
    Klasse zum Darstellen eines Sprachsignals
    Schalldruck-Zeit-Funktion
    Auswahl von Signalabschnitten
    Spektrogramm
    G. Flach, 19.08.2021
    """
    def __init__(self, name='tagesschau'):
        self.fS = 44100
        self.dur = 1
        self.Data = np.zeros(self.fS*self.dur)
        self.name = name
        self.t = np.linspace(0, self.dur, len(self.Data))
        self.read_sig(self.name)
        self.zf = 'boxcar'
    
    def read_sig(self, name):
        file_name = 'sound/' + name + '.wav'
        self.name = file_name
        self.fS, self.Data = wavfile.read(file_name)
        self.Data = self.Data/np.max(np.abs(self.Data))
        self.dur = len(self.Data)/self.fS
        self.t = np.linspace(0, self.dur, len(self.Data))
        
    def show_tsig_int(self, name):
        """
        Signaldarstellung mit auswählbaren Parametern
        - Signal
        """
        self.read_sig(name)
        plt.figure(figsize=(12,4))
        self.t = np.linspace(0, self.dur, len(self.Data))
        plt.plot(self.t, self.Data)
        plt.title('normierte Schalldruck-Zeit-Funktion des Sprachsignals')
        plt.grid(True)
        plt.xlabel('t in s', x=1)
        plt.axis([0, self.dur, np.min(self.Data), np.max(self.Data)])
    
    def show_tsig(self):
        """
        Parameterauswahl zur Signaldarstellung 
        """
        style = {'description_width': 'initial'}
        sig = widgets.Dropdown(options=['sinus', 'sweep', 'noise', 'berlin', 'tagesschau', 'roboter', 'a', 'e', 'o'], value='tagesschau', description='Signal:', style=style)
        interact(self.show_tsig_int, name=sig)

    def sel_parts(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
        t = np.linspace(0, self.dur, len(self.Data))
        ax1.plot(t, self.Data)
        ax1.grid(True)
        ax1.set_ylim(np.min(self.Data), np.max(self.Data))
        ax1.set_title('Originalsignal: ' + self.name + ' - Auswahl mit Mauszeiger')
        line2, = ax2.plot(t, self.Data)
        ax2.grid(True)
        ax2.set_ylim(np.min(self.Data), np.max(self.Data))
        ax2.set_xlabel('t in s')
        return ax1, ax2, line2, t, self.Data, self.fS
    
    def show_energy(self, zflen, zfol, zf='boxcar', sig='tagesschau'):
        """
        Berechnung des Verlaufs der Signalenergie mit auswählbaren Parametern
        - Signal
        - Zeitfensterlänge
        - Zeitfensterform
        - Fortsetzrate
        """
        size = zflen
        step = zfol
        self.zf = zf
        self.read_sig(sig)
        t = [np.array(self.t[i : i + size]) for i in range(0, len(self.t)-size, step)]
        t = np.array(t)
        v = [np.array(self.Data[i : i + size]) for i in range(0, len(self.Data)-size, step)]
        v = np.array(v) * get_window(zf, size)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,6))
        ax1.plot(self.t, self.Data)
        ax1.set_title('Schalldruck-Zeit-Funktion')
        ax1.grid(True)
        ax1.set_xlabel('t in s', x=1)
        ax1.axis([0, self.dur, -1, 1])
        a1 = np.array([])
        for k in range(v.shape[0]):
            a = np.ones(len(t[k][:step]))
            a1 = np.append(a1, a * np.sum(v[k]**2))
        ax2.plot(self.t[0:len(a1)], a1/np.max(a1))
        ax2.set_title('normierte Energie-Zeit-Funktion des Sprachsignals')
        ax2.grid(True)
        ax2.set_xlabel('t in s', x=1)
        ax2.axis([0, self.dur, 0, 1])
        plt.tight_layout()
        
    def energy(self):
        """
        Parameterauswahl zur Berechnung des Verlaufs der Signalenergie 
        """
        style = {'description_width': 'initial'}
        zf_len = widgets.Dropdown(options=[128, 256, 512, 1024, 2048, 4096, 8192], value=512, description='Zeitfensterlänge(ATW):', style=style)
        ol = widgets.Dropdown(options=[64, 128, 256, 512, 1024, 2048, 4096],value=256, description='Fortsetzrate(ATW):', style=style)
        sig = widgets.Dropdown(options=['sinus', 'sweep', 'noise', 'berlin', 'tagesschau', 'roboter', 'a', 'e', 'o'], value='tagesschau', description='Signal:', style=style)
        zf = widgets.Dropdown(options=['boxcar', 'hamming', 'hanning', 'bartlett'], value='boxcar', description='Zeitfensterform:', style=style)
        interact(self.show_energy, zflen=zf_len, zfol=ol, zf=zf, sig=sig)

    
    def show_spec(self, NFFT=256, noverlap=128):
        """
        Darstellung des Kurzzeitspektrums
        """
        plt.figure(figsize=(12,4))
        plt.specgram(self.Data, Fs=self.fS, cmap='Blues', NFFT=NFFT, noverlap=noverlap)
        plt.ylabel('f in Hz')
        plt.xlabel('t in s')
        plt.title('Spektrogramm des Sprachsignals')
        plt.show()
        
    def show_all_akf(self, zflen, zfol):
        """
        Darstellung des Gesamtresultats der AKF (ähnlich dem Kurzzeitspektrum)
        """
        size = zflen
        step = zfol
        bins = 40
        t = [np.array(self.t[i : i + size]) for i in range(0, len(self.t)-size, step)]
        t = np.array(t)
        v = [np.array(self.Data[i : i + size]) for i in range(0, len(self.Data)-size, step)]
        v = np.array(v) * get_window(self.zf, size)
        pic = []
        for k in range(v.shape[0]):
            y = np.correlate(v[k], v[k], mode='full')
            res = self.build_bins(y[size-1:2*size], size, bins)
            pic = np.append(pic, res)
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.imshow(pic.reshape(bins,v.shape[0]).T, origin='lower', extent=[0,self.dur,0,1], aspect='auto')
        ax.set_xlabel('t in s')
        ax.set_ylabel('Verschiebung in ZF-len')
        plt.show()
        
    def build_bins(self, y, x1, x2):
        """
        erzeugt "Bins" für die Darstellung der AKF des Gesamtsignals
        """
        lbin = int(x1/x2)
        res = np.zeros(x2)
        for i in range(x2):
            res[i] = np.sum(y[lbin*i:lbin*(i+1)])
        return res
    
    def show_akf(self):
        """
        Darstellung der Lage eines ZF, des gezoomten ZF-Inhalts und der Autokorrelationsfunktion
        """
        style = {'description_width': 'initial'}
        def akf(zf_len, start):
            nr = int(start * self.fS/zf_len)
            t = self.t[nr*zf_len:(nr+1)*zf_len]
            x = self.Data[nr*zf_len:(nr+1)*zf_len] * get_window(self.zf, len(t))

            fig = plt.figure(figsize=(12, 6))
            gs = gridspec.GridSpec(nrows=2, ncols=2)
            ax0 = fig.add_subplot(gs[0, :])
            ax0.plot(self.t, self.Data)
            ax0.plot([start, start],[-1,1],'r')
            ax0.plot([start+0.025, start+0.025],[-1,1],'r')
            ax0.grid(True)
            ax0.set_title('Zeitfunktion')
            ax0.set_xlabel('t in s')

            ax1 = fig.add_subplot(gs[1, 0])
            ax1.plot(t,x)
            ax1.axis([np.min(t), np.max(t), -1, 1])
            ax1.grid(True)
            ax1.set_title('ausgewähltes Zeitfenster')
            ax1.set_xlabel('t in s')

            r = np.arange(0,1,1/zf_len)
            y = np.correlate(x, x, mode='full')
            ax2 = fig.add_subplot(gs[1, 1])
            ax2.plot(r[0:len(y[zf_len-1:2*zf_len])], y[zf_len-1:2*zf_len])
            ax2.grid(True)
            ax2.set_title('Autokorrelation (Zeitfenster)')
            ax2.set_xlabel('Verschiebung in Zeitfensterlängen')

            plt.tight_layout()
            plt.show()

        zf_len = int(0.025 * self.fS)
        start = widgets.FloatSlider(value=0.0, min=0.0, max=self.dur, step=0.025, description='Startzeit:', style=style, readout_format='.3f', layout=Layout(width='80%'))
        interact(akf, zf_len=fixed(zf_len), start=start)
        
    def zero_cross_int(self, name, zflen):
        """
        Berechnung des Verlaufs der Nulldurchgangsrate mit auswählbaren Parametern
        - Signal
        - Zeitfensterlänge
        """
        size = zflen
        step = size
        self.read_sig(name)
        t = [np.array(self.t[i : i + size]) for i in range(0, len(self.t)-size, step)]
        t = np.array(t)
        v = [np.array(self.Data[i : i + size]) for i in range(0, len(self.Data)-size, step)]
        v = np.array(v)
        zr = np.ones(size)
        for k in range(v.shape[0]):
            zero_crosses = np.nonzero(np.diff(v[k] > 0))[0].size
            zr = np.append(zr, np.ones(size)*zero_crosses)
        fig, ax = plt.subplots(figsize=(12, 4))
        
        ax.plot(self.t, zr[0:len(self.t)])
        ax.set_title('Nulldurchgangsrate pro Zeitfenster')
        ax.set_xlabel('t in s')
        ax.set_ylabel('Anzahl')
        ax.grid(True)
        plt.show()
        
    def zero_cross(self):
        """
        Parameterauswahl zur Berechnung des Verlaufs der Nulldurchgangsrate 
        """
        style = {'description_width': 'initial'}
        zf_len = widgets.Dropdown(options=[128, 256, 512, 1024, 2048, 4096, 8192], value=512, description='Zeitfensterlänge(ATW):', style=style)
        sig = widgets.Dropdown(options=['sinus', 'sweep', 'noise', 'berlin', 'tagesschau', 'roboter', 'a', 'e', 'o'], value='tagesschau', description='Signal:', style=style)
        interact(self.zero_cross_int, name=sig, zflen=zf_len)

    def show_stfft(self, zflen, zf_fr, zf, sig):
        """
        Berechnung des Spektrogramms mit auswählbaren Parametern
        - Signal
        - Zeitfensterlänge
        - Zeitfensterform
        - Fortsetzrate
        """
        size = zflen
        step = zf_fr
        self.read_sig(sig)
        #t = [np.array(self.t[i : i + size]) for i in range(0, len(self.t)-size, step)]
        #t = np.array(t)
        v = [np.array(self.Data[i : i + size]) for i in range(0, len(self.Data)-size, step)]
        v = np.array(v)
        f = np.linspace(0,self.fS/2, int(size/2))
        sk = []
        for k in range(v.shape[0]):
            #print(len(np.abs(np.fft.rfft(v[k]))))
            sk = np.append(sk, np.abs(np.fft.rfft(v[k])))    
        #print(np.min(sk), np.max(sk))
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.imshow(sk.reshape(v.shape[0], int(size/2)+1).T, origin='lower', extent=[0,self.dur,0,self.fS/2], aspect='auto', cmap ='Blues')
        ax.set_xlabel('t in s')
        ax.set_ylabel('f in Hz')
        ax.set_title('Spektrogramm')
        plt.tight_layout()
        
    def shorttime_fft(self):
        """
        Parameterauswahl zur Berechnung des Spektrogramms 
        """
        style = {'description_width': 'initial'}
        zf_len = widgets.Dropdown(options=[128, 256, 512, 1024, 2048, 4096, 8192], value=512, description='Zeitfensterlänge(ATW):', style=style)
        ol = widgets.Dropdown(options=[64, 128, 256, 512, 1024, 2048, 4096],value=256, description='Fortsetzrate(ATW):', style=style)
        sig = widgets.Dropdown(options=['sinus', 'sweep', 'noise', 'berlin', 'tagesschau', 'roboter', 'a', 'e', 'o'], value='tagesschau', description='Signal:', style=style)
        zf = widgets.Dropdown(options=['boxcar', 'hamming', 'hanning', 'bartlett'], value='boxcar', description='Zeitfensterform:', style=style)
        interact(self.show_stfft, zflen=zf_len, zf_fr=ol, zf=zf, sig=sig)
        

def demo_windowing():
    style = {'description_width': 'initial'}
    sig_f1 = widgets.FloatSlider(value=10, min=8, max=12, step=0.1, description='Frequenz f1(Hz):', style=style)
    sig_f2 = widgets.FloatSlider(value=10, min=5, max=20, step=0.1, description='Frequenz f2(Hz):', style=style)
    sig_a1 = widgets.FloatSlider(value=1, min=0, max=5, step=0.1, description='Amplitude a1:', style=style)
    sig_a2 = widgets.FloatSlider(value=0, min=0, max=1, step=0.01, description='Amplitude a2:', style=style)
    sig_dur = widgets.IntSlider(value=1, min=1, max=3, step=1, description='Signaldauer (s):', style=style)
    sig_fs = widgets.IntSlider(value=500, min=100, max=2000, step=100, description='Abtastfrequenz (Hz):', style=style)
    wf = widgets.Dropdown(options=['boxcar', 'hamming', 'blackman'], value='boxcar', description='Zeitfensterform:', style=style)
    interact(win_demo, tmax=sig_dur, f1=sig_f1, a1=sig_a1, f2=sig_f2, a2=sig_a2, fs=sig_fs, wf=wf)
    
def win_demo(tmax, f1, a1, f2, a2, fs, wf):
    t = np.arange(0, tmax, step=1/fs)
    m = t.size
    s = a1 * np.sin(2 * np.pi * f1 * t) + a2 * np.sin(2 * np.pi * f2 * t)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15,10))
    ax1.stem(t, s, markerfmt=' ', use_line_collection=True)
    ax1.set_title("Zeitfunktion, {} samples, sampling rate {} Hz".format(m, 1/(t[1] - t[0])))
    ax1.set_xlabel('t in s')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True)
    n = 8192
    w = np.fft.rfft(s * get_window(wf, m), n=n)
    freqs = np.fft.rfftfreq(n, d=t[1] - t[0])
    ax2.plot(freqs, 20*np.log10(np.abs(w)/np.max(np.abs(w))), ':')
    n = len(t)
    w = np.fft.rfft(s * get_window(wf, m), n=n)
    freqs = np.fft.rfftfreq(n, d=t[1] - t[0])
    ax2.stem(freqs, 20*np.log10(np.abs(w)/np.max(np.abs(w))), markerfmt=' ', use_line_collection=True)
    ax2.grid(True)
    ax2.set_title('Fensterspektrum ' + wf + ' und Frequenzkomponenten')
    ax2.set_xlabel('f in Hz')
    ax2.set_ylabel('Dämpfung in dB')
    ax2.set_ylim(-125, 25)
    ax2.set_xlim(0, int(2 * max(f1, f2)))
    plt.tight_layout()
                                
