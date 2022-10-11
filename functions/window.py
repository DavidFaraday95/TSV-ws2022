import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import get_window


def compute_mainlobe_width(spectrum):
    """
    computes mainlobe width from spectrum
    
    assumes the mainlobe starts at 0, that spectrum size is odd, and that 
    the spectrum is real-valued (half of the frequencies)
    
    returns the number of samples of full mainlobe (not just half)
    """
    abs_spectrum = np.abs(spectrum)
    current_value = abs_spectrum[0]
    for ind, next_value in enumerate(abs_spectrum):
        if next_value > current_value:
            break
        else:
            current_value = next_value        
    return 2 * ind - 1

def compute_sidelobe_level(spectrum):
    """
    computes sidelobe level from spectrum

    assumes the mainlobe starts at 0, that spectrum size is odd, and that 
    the spectrum is real-valued (half of the frequencies)
    
    returns the level of sidelobes in dB 
    """
    mainlobe_width = compute_mainlobe_width(spectrum)
    ind = int((mainlobe_width - 1) / 2)
    abs_spectrum = np.abs(spectrum)
    return 20 * np.log10(abs_spectrum[ind:].max() / abs_spectrum.max())

def window_tf():
    """
    describes window function in time and frequency domain
    """
    fig, axs = plt.subplots(2, 3, figsize=(15,10))
    # Generieren eines Hamming-Fensters mit 512 Werten
    N = 512
    n = np.arange(N)
    w = get_window('hamming', N)
    axs[0,0].plot(n,w)
    axs[0,0].set_title('Fensterfunktion')
    axs[0,0].set_xlabel('n')

    # Berechnen der reellen FT (256 Koeffizienten
    w_fft = np.fft.rfft(w)
    axs[0,1].plot(np.abs(w_fft))
    axs[0,1].set_title('Koeffizienten der reellen FT')
    axs[0,1].set_xlabel('n')

    #Auswahl der ersten 20 Koeffizienten
    axs[0,2].plot(np.abs(w_fft[0:20]))
    axs[0,2].set_xlabel('n')
    axs[0,2].set_title('Koeffizienten 0 - 20 der reellen FT')

    N = 4096
    w_fft = np.fft.rfft(w, N)
    axs[1,0].plot(np.abs(w_fft[0:160]))
    axs[1,0].set_title('K 0 - 160 der RFT mit zero-padding')
    axs[1,0].set_xlabel('n')
    axs[1,1].plot(20*np.log10(np.abs(w_fft[0:160]) / np.abs(w_fft).max() + np.finfo(float).eps))
    axs[1,1].set_title('K 0 - 160 der RFT mit ZP (logarithmisch)')
    axs[1,1].set_xlabel('n')
    axs[1,2].stem(20*np.log10(np.abs(w_fft[0:160]) / np.abs(w_fft).max() + np.finfo(float).eps))
    axs[1,2].set_title('K 0 - 160 der RFT mit ZP (logarithmisch) als ATW')
    axs[1,2].set_xlabel('n')
    fig.suptitle("Übertragungsfunktion einer Fensterfunktion", fontsize="x-large")
    plt.tight_layout()
    return n, w, w_fft
    
def main_side(n, w, w_fft):
    width = compute_mainlobe_width(w_fft)
    level = compute_sidelobe_level(w_fft)

    fig, axs = plt.subplots(1, 2, figsize=(15,5))
    fig.suptitle("Breite Hauptband, Dämpfung Nebenband", fontsize="x-large")
    axs[0].plot(n,w)
    axs[0].set_title('Zeitfenster')
    axs[0].grid(True)
    axs[1].plot(20*np.log10(np.abs(w_fft[0:160]) / np.abs(w_fft).max() + np.finfo(float).eps))

    ylim_range  = axs[1].get_ylim()
    axs[1].vlines((width - 1) / 2 * 512 / 4096 * 8, ylim_range[0], ylim_range[1], 'r', lw=2)

    xlim_range  = axs[1].get_xlim()
    axs[1].hlines(level, xlim_range[0], xlim_range[1], 'r', lw=2)
    axs[1].set_title('Übertragungsfunktion')
    axs[1].grid(True)
    
def show_wind():
    for window in ['boxcar', 'hanning', 'hamming', 'blackman', 'blackmanharris']:
        m = 513
        w = get_window(window, m)
        n = 4096
        w_fft = np.fft.rfft(w, n)
        freqs = np.fft.rfftfreq(n, d=1/m)
        plt.figure(figsize=(15, 5))
        plt.subplot(121)
        plt.plot(w)
        plt.xlabel("n")
        plt.ylabel("Amplitude")
        plt.title("{} window".format(window))
        #plt.xlim(0, t.size)
        plt.ylim(-0.025, 1.025)
        plt.grid(True)
        plt.subplot(122)
        plt.plot(freqs, 20*np.log10(np.abs(w_fft) / np.abs(w_fft).max() + np.finfo(float).eps))
        plt.xlim(0, 25)
        plt.ylim(-120, 1)
        width = compute_mainlobe_width(w_fft)
        width_bins = width * m / n
        level = compute_sidelobe_level(w_fft)
        ylim_range = plt.ylim()
        plt.vlines((width - 1) / 2 * m / n, ylim_range[0], ylim_range[1], 'r', lw=2)
        xlim_range = plt.xlim()
        plt.hlines(level, xlim_range[0], xlim_range[1], 'r', lw=2)
        plt.title("{} window\nmainlobe width = {:.0f} bins, sidelobe level = {:.0f} dB".format(window,
                                                                           width_bins, 
                                                                           level))
        plt.grid(True)
        plt.xlabel('frequency bin #')